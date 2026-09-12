"""VICTR context-chunk conditioning ported onto the pi0-FAST (discrete, autoregressive)
backend (openpi.models.pi0_fast.Pi0FAST) instead of pi0/pi0.5's continuous
flow-matching backend (openpi.models.pi0_victr.Pi0Victr). This is our native
equivalent of the RICL baseline (https://arxiv.org/abs/2508.02062, retrieval-
conditioned in-context learning on a FAST-tokenized VLA) -- built directly on our own
retrieval-context/data pipeline instead of adapting third_party/ricl_openpi's
DROID-specific fork (which hardcodes DROID's raw h5+JPG preprocessing, 3 fixed camera
names, and an 8-dim action space that don't match icl-dataset).

embed_context_chunks is ported near-verbatim from pi0_victr.Pi0Victr -- it only calls
self.PaliGemma.img/.llm, which Pi0FAST exposes with the same img() signature (see
Pi0FAST.embed_inputs). The one call that differs is text-token embedding: gemma_fast's
llm module uses `embed_only=True`, where pi0.py's gemma module (used by Pi0Victr) uses
`method="embed"` -- see gemma_fast.Module.__call__ vs gemma.Module.__call__.

Unlike Pi0Victr (separate embed_prefix + flow-matching embed_suffix, with a 1-D,
batch-shared ar_mask), Pi0FAST builds one unified autoregressive sequence via
embed_inputs, whose ar_mask is already per-batch-item int32 (token_ar_mask varies in
length per example, since it marks the prompt/action boundary within padded text).
embed_inputs_with_context therefore broadcasts embed_context_chunks' shared 1-D ar_mask
to match before concatenating, and forces query_ar[:, 0] = 1 (a fresh causal block,
mirroring Pi0Victr's query_ar.at[0].set(True)) so the query's own block sits at a
higher cumulative ar_mask than every context chunk and can attend to all of them (see
pi0_fast.make_attn_mask's docstring for the cumsum-block convention).

compute_loss/sample_actions are copied from pi0_fast.Pi0FAST verbatim except for
swapping self.embed_inputs(observation) for self.embed_inputs_with_context(observation)
-- every other line (one-hot CE targets, KV-cache prefill/decode) only reads the
resulting token embeddings/masks positionally, so it's agnostic to whether a context
prefix was prepended. Context chunks are prepended (not appended), so slicing the last
N positions off the concatenated sequence (targets.shape[1] for the loss, action_horizon
for continuous models) still isolates exactly the query's own logits regardless of how
many context tokens precede them.
"""

import dataclasses
from typing import Literal

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_fast
import openpi.models.gemma_fast as _gemma_fast
from openpi.shared import array_typing as at


class Pi0FastVictr(pi0_fast.Pi0FAST):
    """pi0_fast.Pi0FAST + a learned per-neighbor-rank embedding for context blocks."""

    def __init__(self, config: "Pi0FastVictrConfig", rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self.victr_config = config
        width = _gemma_fast.get_config(config.paligemma_variant)["width"]
        num_ranks = max(config.num_context_chunks, 1)
        self.neighbor_rank_embedding = nnx.Embed(num_ranks, width, rngs=rngs)

    @at.typecheck
    def embed_context_chunks(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """One image+text block per retrieved chunk (obs.context_images/context_tokens,
        farthest-to-nearest -- see openpi.policies.yor_retrieval.RetrievalContextInputs),
        each tagged with a learned neighbor_rank_embedding and starting a fresh causal
        block (mirrors embed_inputs, parameterized by rank instead of camera name)."""
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

            text_tokens = self.PaliGemma.llm(obs.context_tokens[:, rank], embed_only=True)
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

    @staticmethod
    def _dynamic_window(arr: at.Array, start: at.Int[at.Array, " b"], width: int) -> at.Array:
        """Fixed-`width` slice of `arr` along axis 1, starting at each batch item's own
        dynamic `start` index. jax.lax.dynamic_slice clamps an out-of-range start to the
        array's valid range rather than raising, so this never reads out of bounds even
        if `start + width` would otherwise overflow -- see interpolate_actions callers."""
        return jax.vmap(lambda a, s: jax.lax.dynamic_slice_in_dim(a, s, width, axis=0))(arr, start)

    @at.typecheck
    def interpolate_actions(
        self,
        logits: at.Float[at.Array, "b w v"],
        nearest_action_tokens: at.Int[at.Array, "b w"],
        weight: at.Float[at.Array, "b w"],
    ) -> at.Float[at.Array, "b w v"]:
        """RICL-style (arXiv:2508.02062) discrete action interpolation: per-position
        convex blend of the model's own predicted action-token distribution with the
        top-1 retrieved chunk's actual FAST action tokens. `weight` is
        exp(-lamda * normalized vision distance) with the nearest chunk's own padding
        mask (and, for the decode-step caller, an in-range mask) already folded in by
        the caller -- weight=0 at any position falls back to the model's own
        prediction unmodified. Returns blended probabilities (not log-probabilities),
        matching third_party/ricl_openpi's own convention: compute_loss clips+logs the
        result itself; sample_actions uses it in place of logits directly (see that
        method for why this is fine at temperature=0, the only mode either baseline
        actually serves with).

        Avoids materializing a (b, w, vocab) one-hot for the target: with vocab
        ~257k, batch 128, and w~191, that one-hot (plus its blended-output copy
        and their gradient buffers) was allocating ~100GB and OOMing a single
        H200 outright. `w * onehot(target) + (1-w) * softmax(logits)` is exactly
        `(1-w) * softmax(logits)` everywhere except a single `+w` at the target
        index per position, so a scatter-add onto the (much cheaper) softmax
        output is mathematically identical without ever building the one-hot."""
        w = weight[..., None]
        blended = (1 - w) * jax.nn.softmax(logits, axis=-1)
        b, win = nearest_action_tokens.shape
        return blended.at[jnp.arange(b)[:, None], jnp.arange(win)[None, :], nearest_action_tokens].add(weight)

    def _blend_decode_step(
        self,
        last_logit: at.Float[at.Array, "b 1 v"],
        observation: _model.Observation,
        step_index: at.Int[at.Array, ""],
    ) -> at.Float[at.Array, "b 1 v"]:
        """interpolate_actions for a single autoregressive decode step (sample_actions).
        step_index counts postfix positions from 0 (the first generated token) -- decode
        always starts exactly at the query's postfix boundary (prefix is fully
        prefilled first), so unlike compute_loss's batched case, no per-example
        token_loss_mask lookup is needed to locate it."""
        window = observation.nearest_action_tokens.shape[1]
        clamped = jnp.clip(step_index, 0, window - 1)
        token = jax.lax.dynamic_slice_in_dim(observation.nearest_action_tokens, clamped, 1, axis=1)
        mask = jax.lax.dynamic_slice_in_dim(observation.nearest_action_tokens_mask, clamped, 1, axis=1)
        in_range = (step_index < window).astype(last_logit.dtype)
        weight = observation.exp_lamda_distance[:, None] * mask.astype(last_logit.dtype) * in_range
        return self.interpolate_actions(last_logit, token, weight)

    @at.typecheck
    def embed_inputs_with_context(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        query_tokens, query_mask, query_ar = self.embed_inputs(obs)
        if obs.context_images is None:
            return query_tokens, query_mask, query_ar
        query_ar = query_ar.at[:, 0].set(1)  # query sequence also starts a fresh block, after all context chunks
        ctx_tokens, ctx_mask, ctx_ar = self.embed_context_chunks(obs)
        ctx_ar = jnp.broadcast_to(ctx_ar.astype(query_ar.dtype), (query_ar.shape[0], ctx_ar.shape[0]))
        return (
            jnp.concatenate([ctx_tokens, query_tokens], axis=1),
            jnp.concatenate([ctx_mask, query_mask], axis=1),
            jnp.concatenate([ctx_ar, query_ar], axis=1),
        )

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        # Compute inputs: one big forward pass of context + prefix + suffix at once
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs_with_context(observation)
        attn_mask = pi0_fast.make_attn_mask(input_mask, ar_mask)

        # Compute one-hot targets: we predict *next* token, so shift the input tokens by one.
        # (observation.tokenized_prompt is unaffected by context chunks -- they're only
        # prepended to the token *embedding* sequence, not to the query's own raw token
        # ids -- so this target/loss-mask construction is unchanged from Pi0FAST.)
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.PaliGemma.llm.module.vocab_size,
        )

        # Each input predicts *next* token, so we don't input the last token.
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )

        # Only decode logits for the target tokens to save memory -- context chunks
        # precede the query, so the trailing targets.shape[1] positions are exactly the
        # query's own logits regardless of how many context tokens came before them.
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1] :],
        )

        # Compute CE loss on token targets
        assert observation.token_loss_mask is not None, "Token loss mask is required"
        loss_mask = observation.token_loss_mask[:, 1:]

        logp = jax.nn.log_softmax(logits, axis=-1)
        token_pplx = jnp.sum(targets * logp, axis=-1)

        if self.victr_config.use_action_interpolation:
            # Action interpolation (notes/action_interpolation.md): blend the query's
            # own predicted action-token distribution with the top-1 retrieved chunk's
            # actual FAST action tokens over a fixed-width window starting at the
            # query's own postfix boundary (loss_mask's first True position) -- see
            # interpolate_actions' docstring for why this differs from RICL's own
            # fixed 50/50 prefix/postfix tokenizer split.
            #
            # token_pplx (like the non-interpolation branch above) only ever needs the
            # scalar log-prob at each position's own target token id -- sum(targets *
            # logp) is just a one-hot gather. The previous version built a full (b,
            # window, vocab) blended distribution (interpolate_actions' one-hot-based
            # blend, then log(clip(...)) over the whole vocab) purely to re-gather that
            # same scalar afterwards via the one-hot multiply below. That intermediate
            # (plus logits_window itself and the one-hot target it was built from) was
            # 3-4 extra ~27GB buffers on top of the model's own full-sequence
            # targets/logp (~31GB each), which OOM'd a single H200 outright (jobs
            # 16561972/16612787: RESOURCE_EXHAUSTED on a single ~100-120GB allocation).
            # Gathering the target log-prob directly via take_along_axis +
            # log_softmax(x)[i] = x[i] - logsumexp(x) needs only logits_window itself
            # (already unavoidable) plus (b, window)-sized scalars -- no second
            # (b, window, vocab) array at all.
            window = observation.nearest_action_tokens.shape[1]
            postfix_start = jnp.argmax(loss_mask, axis=-1)
            logits_window = self._dynamic_window(logits, postfix_start, window)
            target_ids_window = self._dynamic_window(observation.tokenized_prompt[:, 1:], postfix_start, window)
            weight = observation.exp_lamda_distance[:, None] * observation.nearest_action_tokens_mask.astype(
                logits.dtype
            )
            logp_at_target_window = jnp.take_along_axis(
                jax.nn.log_softmax(logits_window, axis=-1), target_ids_window[..., None], axis=-1
            )[..., 0]
            is_nearest_match = (observation.nearest_action_tokens == target_ids_window).astype(logits.dtype)
            epsilon = 1e-9
            blended_at_target = (1 - weight) * jnp.exp(logp_at_target_window) + weight * is_nearest_match
            token_pplx_window = jnp.log(jnp.clip(blended_at_target, epsilon, 1 - epsilon))
            token_pplx = jax.vmap(lambda full, win, s: jax.lax.dynamic_update_slice_in_dim(full, win, s, axis=0))(
                token_pplx, token_pplx_window, postfix_start
            )
        return -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        # embed inputs (context chunks + query)
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs_with_context(observation)
        prefix_attn_mask = pi0_fast.make_attn_mask(prefix_mask, prefix_ar_mask)

        # left to right align all input token sequences
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = pi0_fast.left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # first fill KV cache with a forward pass of the prefix
        # pad attention mask to set the size of the KV cache (prefill_size + max_decoding_steps)
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
        )

        # prepare decoding -- final logit decodes the first token
        last_logit = prefix_logits[:, -1:]
        if self.victr_config.use_action_interpolation:
            last_logit = self._blend_decode_step(last_logit, observation, jnp.array(0))
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

        def step(carry):
            rng, last_logit, output_tokens, cache, _, step = carry

            # Sample token from last logit
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = pi0_fast.put_along_last_axis(
                output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token
            )

            # Check for early stopping --> stop if all batch elements have EOS token
            has_eos = jnp.any(token == pi0_fast.PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            # Decode one step
            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
            )
            if self.victr_config.use_action_interpolation:
                last_logit = self._blend_decode_step(last_logit, observation, step + 1)

            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # Use lax.while_loop so we can jit the full decoding loop.
        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
        )
        return output_tokens


@dataclasses.dataclass(frozen=True)
class Pi0FastVictrConfig(pi0_fast.Pi0FASTConfig):
    """Pi0FASTConfig + in-context retrieval parameters -- the pi0-FAST-backed
    counterpart to pi0_victr.Pi0VictrConfig (see that class's docstring). model_type/
    ModelTransformFactory dispatch are inherited unchanged from Pi0FASTConfig (still
    PI0_FAST -- the stock TokenizeFASTInputs/ExtractFASTActions pipeline applies
    unmodified) -- only `create()`/`inputs_spec()` differ, adding the context fields
    and building a Pi0FastVictr instead of a Pi0FAST."""

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

    # RICL-style action interpolation (arXiv:2508.02062; notes/action_interpolation.md;
    # Pi0FastVictr.interpolate_actions) -- blends the model's own predicted action
    # tokens with the top-1 retrieved chunk's actual action tokens, weighted by
    # exp(-lamda * vision distance). Requires retrieval_metric="vision" (RICL's own
    # mechanism has no value-distance analog) and openpi.policies.yor_retrieval.
    # RetrievalContextInputs to be configured with a matching use_action_interpolation
    # (see LeRobotYorVictrDataConfig) -- this flag alone only controls the model side.
    use_action_interpolation: bool = False
    lamda: float = 10.0
    # Fixed token budget for the nearest chunk's FASTTokenizer.tokenize_action_only
    # target (and the matching window sliced out of the query's own postfix) -- must
    # be >= the longest actual "Action: <FAST tokens> |" span this action space
    # produces, same convention as Pi0KiConfig's max_action_tokens.
    max_action_tokens: int = 192

    def __post_init__(self):
        if self.num_context_chunks < 0:
            raise ValueError(f"num_context_chunks must be >= 0, got {self.num_context_chunks}")
        if self.context_frames_per_chunk < 1:
            raise ValueError(f"context_frames_per_chunk must be >= 1, got {self.context_frames_per_chunk}")
        if self.use_action_interpolation and self.retrieval_metric != "vision":
            raise ValueError("use_action_interpolation requires retrieval_metric='vision'")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FastVictr":
        return Pi0FastVictr(self, rngs=nnx.Rngs(rng))

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
                window = self.max_action_tokens - 1  # tokenize_action_only's own len, minus its leading bos
                observation_spec = dataclasses.replace(
                    observation_spec,
                    exp_lamda_distance=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                    nearest_action_tokens=jax.ShapeDtypeStruct([batch_size, window], jnp.int32),
                    nearest_action_tokens_mask=jax.ShapeDtypeStruct([batch_size, window], bool),
                )
        return observation_spec, action_spec
