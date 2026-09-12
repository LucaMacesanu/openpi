"""Knowledge Insulation (KI) + subtask prediction: pi0/pi05 with the flow-matching
loss's gradient kept off the PaliGemma backbone, which instead keeps training via two
discrete next-token cross-entropy losses through its own LM head -- one on
FAST-tokenized ground-truth actions (ki_loss), one on the active subtask's text
(subtask_loss). JAX/openpi port of the lerobot patch to
lerobot/policies/pi05/modeling_pi05.py's compute_ki_loss/compute_subtask_loss
(third_party/lerobot, patched at $SCRATCH/lerobot-src), currently training
yor-pi05-ablation-ki-subtask.yaml.

lerobot's PyTorch patch found that merely detaching the prefix *input* embeddings
before the fused two-stream forward pass does not insulate the backbone -- PyTorch
still back-props into the backbone's *weights* through that op even when the input
itself has no grad. Its fix: prefill the prefix's KV cache under torch.no_grad(), then
run a suffix-only continuation that cross-attends into that ungraphed cache. The same
subtlety holds in JAX (stop_gradient on an *input* to y=f(params, x) does not stop
gradient to params), but the fix is simpler here: reuse pi0.py's own sample_actions
prefill/continuation pattern for training, with jax.lax.stop_gradient applied to the
*output* KV cache. Since the two-expert Gemma module (openpi.models.gemma.Module) keeps
PaliGemma and the action expert as entirely separate per-layer params, this stop_gradient
insulates only PaliGemma -- the action expert still trains normally from the (unstopped)
flow-matching loss.

ki_loss/subtask_loss get full gradient into PaliGemma via a *separate*, ordinary
forward pass over [prefix, target_tokens] with the action expert absent entirely (not
insulated, not reused from the flow branch) -- so the prefix effectively runs through
PaliGemma's layers twice per step, the JAX-side price for correct insulation (matches
the 2x-compute tradeoff the lerobot patch's own comments call out).
"""

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at


class Pi0Ki(pi0.Pi0):
    """pi0.Pi0 with an insulated flow-matching loss + discrete FAST-action/subtask
    auxiliary losses. Adds no new learned params (unlike Pi0Victr) -- only the loss
    computation differs, so the plain CheckpointWeightLoader works unmodified."""

    def __init__(self, config: "Pi0KiConfig", rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self.ki_config = config

    @staticmethod
    def _next_token_ce(logits: at.Array, targets: at.Array, mask: at.Array) -> at.Array:
        """Masked next-token CE within one target span: logits[:, i] predicts
        targets[:, i+1] (targets[:, 0], typically bos, is never predicted -- it's the
        span's own first "given" token, matching the lerobot reference's convention)."""
        logits = logits[:, :-1, :]
        targets = targets[:, 1:]
        m = mask[:, 1:].astype(jnp.float32)
        logp = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        return jnp.sum(nll * m) / jnp.maximum(jnp.sum(m), 1.0)

    def _discrete_losses(
        self,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_ar_mask: at.Bool[at.Array, " s"],
        fast_tokens: at.Int[at.Array, "b fl"],
        fast_mask: at.Bool[at.Array, "b fl"],
        subtask_tokens: at.Int[at.Array, "b sl"] | None,
        subtask_mask: at.Bool[at.Array, "b sl"] | None,
    ) -> tuple[at.Float[at.Array, ""], at.Float[at.Array, ""] | None]:
        """ki_loss (+ subtask_loss, if given) via a SINGLE pure-PaliGemma forward pass
        over [prefix, fast_action_tokens, subtask_tokens] (not two separate passes --
        each extra full-prefix-length forward pass roughly doubles this branch's
        activation memory, and running three total passes per step (this + the
        insulated flow branch's own prefill) was enough to OOM a single H200 even at
        half the global batch size). fast_action and subtask each start a fresh
        causal block (attends to the prefix + itself only) and, critically, the
        SUBTASK block's visibility into the fast_action block is explicitly masked
        OFF after the fact -- pi0.make_attn_mask's cumulative block-causal convention
        would otherwise let subtask (the later block) attend to fast_action (the
        earlier one), letting subtask-prediction "cheat" by reading the ground-truth
        action instead of learning from the prefix alone, changing what the auxiliary
        loss actually trains (and diverging from lerobot's genuinely-separate-forward-
        passes reference behavior)."""
        prefix_len = prefix_tokens.shape[1]
        fast_len = fast_tokens.shape[1]
        fast_embs = self.PaliGemma.llm(fast_tokens, method="embed")
        fast_ar = jnp.ones((fast_len,), dtype=jnp.bool_)

        if subtask_tokens is not None:
            subtask_embs = self.PaliGemma.llm(subtask_tokens, method="embed")
            subtask_ar = jnp.ones((subtask_tokens.shape[1],), dtype=jnp.bool_)
            tokens = jnp.concatenate([prefix_tokens, fast_embs, subtask_embs], axis=1)
            input_mask = jnp.concatenate([prefix_mask, fast_mask, subtask_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, fast_ar, subtask_ar], axis=0)
        else:
            tokens = jnp.concatenate([prefix_tokens, fast_embs], axis=1)
            input_mask = jnp.concatenate([prefix_mask, fast_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, fast_ar], axis=0)

        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        if subtask_tokens is not None:
            attn_mask = attn_mask.at[
                :, prefix_len + fast_len :, prefix_len : prefix_len + fast_len
            ].set(False)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (lm_out, _), _ = self.PaliGemma.llm([tokens, None], mask=attn_mask, positions=positions)
        # Decode only the target span (fast_action + subtask), not the whole sequence:
        # decode() projects through the full 257k-token vocab table, so running it over
        # the ~350-token prefix too (which is never used -- only fast/subtask logits are
        # needed) was materializing a needless (batch, prefix_len+fast_len+subtask_len,
        # 257152) tensor, tens of GB by itself and the dominant cause of an OOM this
        # slicing avoids.
        target_hidden = lm_out[:, prefix_len:, :]
        logits = self.PaliGemma.llm(target_hidden, method="decode").astype(jnp.float32)

        fast_logits = logits[:, :fast_len, :]
        ki_loss = self._next_token_ce(fast_logits, fast_tokens, fast_mask)

        subtask_loss = None
        if subtask_tokens is not None:
            subtask_logits = logits[:, fast_len:, :]
            subtask_loss = self._next_token_ce(subtask_logits, subtask_tokens, subtask_mask)

        return ki_loss, subtask_loss

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        # --- discrete losses: full gradient into PaliGemma, live prefix, one shared pass ---
        subtask_tokens = observation.subtask_tokens if self.ki_config.use_subtask_prediction else None
        subtask_mask = observation.subtask_tokens_mask if self.ki_config.use_subtask_prediction else None
        ki_loss, subtask_loss = self._discrete_losses(
            prefix_tokens, prefix_mask, prefix_ar_mask,
            observation.fast_action_tokens, observation.fast_action_tokens_mask,
            subtask_tokens, subtask_mask,
        )
        aux_loss = self.ki_config.ki_loss_weight * ki_loss
        if subtask_loss is not None:
            aux_loss = aux_loss + self.ki_config.subtask_loss_weight * subtask_loss

        # --- flow-matching loss: insulated from PaliGemma via a stop-gradiented KV cache ---
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions_prefix)
        kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)

        suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_rep = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask_rep, suffix_attn_mask], axis=-1)
        positions_suffix = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (_prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions_suffix,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        return flow_loss + aux_loss

    # sample_actions: inherited unchanged from pi0.Pi0 -- KI only changes the training
    # loss, not inference (no context blocks to add, unlike Pi0Victr).


@dataclasses.dataclass(frozen=True)
class Pi0KiConfig(pi0_config.Pi0Config):
    """Pi0Config + Knowledge Insulation / subtask-prediction parameters (mirrors
    lerobot's PI05Config KI fields). model_type/pi05 behavior, defaults, and
    ModelTransformFactory dispatch are all inherited unchanged from Pi0Config -- only
    `create()`/`inputs_spec()` differ."""

    # Local dir or HF repo id of a FAST tokenizer fit on this dataset's own action
    # distribution (nyu-finger-robot/tools/fixes/fit_fast_tokenizer.py).
    fast_tokenizer_path: str = ""
    ki_loss_weight: float = 1.0
    # Padded size of the [bos, "Action: ", <FAST tokens>, "|"] span. Separate budget
    # from max_token_len, which bounds only the prefix prompt.
    max_action_tokens: int = 256

    use_subtask_prediction: bool = True
    # Path to a subtasks.jsonl side-table (episode_index -> subtask_names/
    # subtask_start_frames/subtask_end_frames), e.g. from SARM-style VLM annotation.
    subtask_annotations_path: str = ""
    subtask_loss_weight: float = 1.0
    max_subtask_tokens: int = 32

    def __post_init__(self):
        super().__post_init__()
        if not self.fast_tokenizer_path:
            raise ValueError("fast_tokenizer_path must be set for Pi0KiConfig.")
        if self.use_subtask_prediction and not self.subtask_annotations_path:
            raise ValueError("subtask_annotations_path must be set when use_subtask_prediction=True.")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Ki":
        return Pi0Ki(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        fast_tokens_spec = jax.ShapeDtypeStruct([batch_size, self.max_action_tokens], jnp.int32)
        fast_mask_spec = jax.ShapeDtypeStruct([batch_size, self.max_action_tokens], bool)
        with at.disable_typechecking():
            observation_spec = dataclasses.replace(
                observation_spec,
                fast_action_tokens=fast_tokens_spec,
                fast_action_tokens_mask=fast_mask_spec,
            )
            if self.use_subtask_prediction:
                subtask_tokens_spec = jax.ShapeDtypeStruct([batch_size, self.max_subtask_tokens], jnp.int32)
                subtask_mask_spec = jax.ShapeDtypeStruct([batch_size, self.max_subtask_tokens], bool)
                observation_spec = dataclasses.replace(
                    observation_spec,
                    subtask_tokens=subtask_tokens_spec,
                    subtask_tokens_mask=subtask_mask_spec,
                )
        return observation_spec, action_spec
