"""VICTR: pi0/pi05 extended with retrieved-chunk context conditioning (paper Sec III-E).

JAX/openpi port of viktr/policy/pi05_context.py's mechanism (lerobot/PyTorch). The two
backbones turn out to share the same block-causal attention convention already:
openpi's `embed_prefix` returns an `ar_mask` where a `True` entry starts a fresh
causal block (see pi0.py's `make_attn_mask` docstring) -- exactly lerobot's `att_mask`
bit. So no new masking primitive is needed here, only a place to build extra
image+text blocks and concatenate them in front of the query's own prefix, farthest
chunk first: the farthest chunk sees only itself, each nearer chunk also sees every
chunk before it, the query prefix (its own fresh block) sees every chunk, and the
query's action-expert suffix sees everything. Context chunks never see the query.
"""

import dataclasses
from typing import Literal

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at


class Pi0Victr(pi0.Pi0):
    """pi0.Pi0 + a learned per-neighbor-rank embedding for context blocks."""

    def __init__(self, config: "Pi0VictrConfig", rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self.victr_config = config
        width = _gemma_width(config)
        num_ranks = max(config.num_context_chunks, 1)
        self.neighbor_rank_embedding = nnx.Embed(num_ranks, width, rngs=rngs)

    @at.typecheck
    def embed_context_chunks(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """One image+text block per retrieved chunk (obs.context_images/context_tokens,
        farthest-to-nearest -- see openpi.policies.yor_retrieval.RetrievalContextInputs),
        each tagged with a learned neighbor_rank_embedding and starting a fresh causal
        block (mirrors embed_prefix, parameterized by rank instead of camera name)."""
        num_chunks = obs.context_images.shape[1]
        all_tokens, all_masks, all_ar = [], [], []
        for rank in range(num_chunks):
            tokens = []
            input_mask = []
            ar_mask = []
            num_frames = obs.context_images.shape[2]
            for frame in range(num_frames):
                image_tokens, _ = self.PaliGemma.img(obs.context_images[:, rank, frame], train=False)
                tokens.append(image_tokens)
                input_mask.append(
                    einops.repeat(obs.context_image_masks[:, rank, frame], "b -> b s", s=image_tokens.shape[1])
                )
                ar_mask += [False] * image_tokens.shape[1]

            text_tokens = self.PaliGemma.llm(obs.context_tokens[:, rank], method="embed")
            tokens.append(text_tokens)
            input_mask.append(obs.context_tokens_mask[:, rank])
            ar_mask += [False] * text_tokens.shape[1]

            block_tokens = jnp.concatenate(tokens, axis=1)
            block_mask = jnp.concatenate(input_mask, axis=1)
            block_ar = jnp.array(ar_mask)
            block_ar = block_ar.at[0].set(True)  # fresh causal block per chunk

            # .reshape(-1): nnx.Embed on a 0-d index can return a (1, width) leading-dim
            # array rather than a pure (width,) vector; flattening first guarantees
            # rank_emb[None, None, :] broadcasts to (1, 1, width) -- not (1, 1, 1, width),
            # which silently prepended a spurious batch axis to block_tokens.
            rank_emb = self.neighbor_rank_embedding(jnp.asarray(rank)).reshape(-1)
            block_tokens = block_tokens + rank_emb[None, None, :].astype(block_tokens.dtype)

            all_tokens.append(block_tokens)
            all_masks.append(block_mask)
            all_ar.append(block_ar)

        return jnp.concatenate(all_tokens, axis=1), jnp.concatenate(all_masks, axis=1), jnp.concatenate(all_ar, axis=0)

    @at.typecheck
    def embed_prefix_with_context(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        query_tokens, query_mask, query_ar = self.embed_prefix(obs)
        if obs.context_images is None:
            return query_tokens, query_mask, query_ar
        query_ar = query_ar.at[0].set(True)  # query prefix also starts a fresh block, after all context chunks
        ctx_tokens, ctx_mask, ctx_ar = self.embed_context_chunks(obs)
        return (
            jnp.concatenate([ctx_tokens, query_tokens], axis=1),
            jnp.concatenate([ctx_mask, query_mask], axis=1),
            jnp.concatenate([ctx_ar, query_ar], axis=0),
        )

    @at.typecheck
    def _blended_action_target(
        self, actions: _model.Actions, observation: _model.Observation
    ) -> _model.Actions:
        """Action-interpolation training target (notes/action_interpolation.md;
        Pi0VictrConfig.use_action_interpolation's docstring) -- the continuous
        analogue of Pi0FastVictr.interpolate_actions' discrete token-distribution
        blend. Convex-blends the query's own true action with the top-1 retrieved
        chunk's actual action (obs.nearest_action), weighted by
        obs.exp_lamda_distance: weight=0 (nearest chunk far away) reproduces the
        query's own action unmodified, matching interpolate_actions' same weight=0
        fallback. Unlike the discrete case -- which only ever touches the CE loss,
        never the tokens the model is conditioned on -- this blended value becomes
        the actual flow-matching target `compute_loss` builds `x_t`/`u_t` from, so a
        high-weight blend also changes what noisy `x_t` the network is trained to
        condition on, not just the regression target it's scored against. This is
        the simpler of two defensible translations (the other being: keep x_t tied
        to the true action and only blend u_t) -- flagged here as a real design
        choice, not a forced one."""
        weight = observation.exp_lamda_distance[..., None, None]
        return (1 - weight) * actions + weight * observation.nearest_action

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        target_actions = actions
        if self.victr_config.use_action_interpolation:
            target_actions = self._blended_action_target(actions, observation)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * target_actions
        u_t = noise - target_actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_with_context(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @at.typecheck
    def _blend_velocity(
        self,
        v_t: at.Float[at.Array, "b ah ad"],
        x_t: at.Float[at.Array, "b ah ad"],
        time: at.Float[at.Array, ""],
        observation: _model.Observation,
    ) -> at.Float[at.Array, "b ah ad"]:
        """Sample-time counterpart of _blended_action_target above -- there is no
        ground-truth action to blend toward at inference, only the current noisy
        iterate x_t, so this blends the *velocity field* instead of the target
        directly (the direct analogue of Pi0FastVictr._blend_decode_step blending
        per-step logits rather than a target token).

        For the same linear interpolant compute_loss uses (x_t = t*noise +
        (1-t)*target), the implied velocity toward *any* hypothetical target y,
        given only the current (x_t, t), is (x_t - y) / t -- exactly u_t = noise - y
        when x_t truly lies on that line (algebraically: x_t - y = t*(noise - y)).
        Using y = obs.nearest_action gives the "what velocity would point this same
        x_t at the neighbor's action instead" counterpart to the model's own
        predicted v_t, blended by the same obs.exp_lamda_distance weight
        (weight=0 reproduces v_t unmodified). `time` never reaches exactly 0 inside
        sample_actions' while_loop (cond stops at time < -dt/2, i.e. the smallest
        time fed here is dt = 1/num_steps), so this division is always well-defined
        at this call site -- not a general-purpose safe-divide."""
        weight = observation.exp_lamda_distance[:, None, None]
        v_neighbor = (x_t - observation.nearest_action) / time
        return (1 - weight) * v_t + weight * v_neighbor

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_with_context(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            if self.victr_config.use_action_interpolation:
                v_t = self._blend_velocity(v_t, x_t, time, observation)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0


def _gemma_width(config: pi0_config.Pi0Config) -> int:
    from openpi.models import gemma as _gemma

    return _gemma.get_config(config.paligemma_variant).width


@dataclasses.dataclass(frozen=True)
class Pi0VictrConfig(pi0_config.Pi0Config):
    """Pi0Config + in-context retrieval parameters (mirrors viktr/policy/
    configuration_victr.py's VictrConfig). model_type/pi05 behavior, defaults, and
    ModelTransformFactory dispatch are all inherited unchanged from Pi0Config -- only
    `create()` differs, building a Pi0Victr instead of a Pi0."""

    # k: number of retrieved chunks conditioning each query (paper Sec III-E).
    num_context_chunks: int = 1
    # L: frames per retrieved chunk. Should match the chunk_size used to build the
    # ChunkDictionary (viktr.retrieval.chunk_dictionary) the chunks came from.
    context_chunk_size: int = 10
    # K: frames subsampled per chunk, embedded through the shared SigLIP vision tower.
    context_frames_per_chunk: int = 1
    # Max token length of each chunk's "Task/State/Action" text summary.
    context_text_max_length: int = 64
    # Which retrieval metric selected the context chunks (informational here --
    # retrieval itself happens upstream, in openpi.policies.yor_retrieval.
    # RetrievalContextInputs; wired through so the TrainConfig/wandb config can log
    # which arm is running).
    retrieval_metric: Literal["vision", "value", "vision_value"] = "vision"

    # RICL-style action interpolation (arXiv:2508.02062; notes/action_interpolation.md),
    # adapted for a flow-matching action head -- the continuous analogue of
    # pi0_fast_victr.Pi0FastVictrConfig's own use_action_interpolation (which blends
    # discrete FAST-token logits; not applicable here, a flow-matching head has no
    # vocabulary/logits to blend). PROTOTYPE: the model-side computation below assumes
    # obs.exp_lamda_distance/obs.nearest_action are already populated -- unlike the
    # FAST case, no data-pipeline wiring (yor_retrieval.py extras function,
    # LeRobotYorVictrCanonicalDataConfig fields, a precompute-script flag) exists yet
    # to actually populate them; this is model-architecture prototyping only, not yet
    # runnable end-to-end.
    #
    # When set, Pi0Victr.compute_loss/sample_actions blend the top-1 retrieved chunk's
    # own native action (obs.nearest_action) into the flow-matching target/velocity,
    # weighted by obs.exp_lamda_distance = exp(-lamda * normalized top-1 vision
    # distance) -- same weighting convention as the FAST case (weight=0 when the
    # nearest chunk is far away falls back to the query's own action/prediction
    # unmodified), just applied to a continuous target instead of a token
    # distribution. See Pi0Victr.compute_loss/sample_actions for the exact formulas
    # and why they differ (loss-time blends the actual regression target; sample-time
    # blends the per-step velocity field, since there is no ground-truth action to
    # blend toward at inference time).
    use_action_interpolation: bool = False
    lamda: float = 10.0

    def __post_init__(self):
        super().__post_init__()
        if self.num_context_chunks < 0:
            raise ValueError(f"num_context_chunks must be >= 0, got {self.num_context_chunks}")
        if self.context_frames_per_chunk < 1:
            raise ValueError(f"context_frames_per_chunk must be >= 1, got {self.context_frames_per_chunk}")
        if self.use_action_interpolation and self.retrieval_metric != "vision":
            raise ValueError("use_action_interpolation requires retrieval_metric='vision'")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Victr":
        return Pi0Victr(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        k = max(self.num_context_chunks, 1)
        context_image_spec = jax.ShapeDtypeStruct(
            [batch_size, k, self.context_frames_per_chunk, *_model.IMAGE_RESOLUTION, 3], jnp.float32
        )
        context_image_mask_spec = jax.ShapeDtypeStruct([batch_size, k, self.context_frames_per_chunk], bool)
        context_tokens_spec = jax.ShapeDtypeStruct([batch_size, k, self.context_text_max_length], jnp.int32)
        context_tokens_mask_spec = jax.ShapeDtypeStruct([batch_size, k, self.context_text_max_length], bool)
        with at.disable_typechecking():
            observation_spec = dataclasses.replace(
                observation_spec,
                context_images=context_image_spec,
                context_image_masks=context_image_mask_spec,
                context_tokens=context_tokens_spec,
                context_tokens_mask=context_tokens_mask_spec,
            )
            if self.use_action_interpolation:
                observation_spec = dataclasses.replace(
                    observation_spec,
                    exp_lamda_distance=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                    nearest_action=jax.ShapeDtypeStruct(
                        [batch_size, self.action_horizon, self.action_dim], jnp.float32
                    ),
                )
        return observation_spec, action_spec
