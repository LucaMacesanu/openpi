"""Dynamic Mixture-of-Experts action expert for Pi0.5.

Implements the DynMoE approach from "DynMoE: Dynamic Mixture of Experts for Continual Learning".

Key differences from moe.py (standard top-k MoE):
  1. Top-Any gating: per-token sigmoid thresholds replace fixed top-k.
     k varies per token; a straight-through estimator (STE) makes it differentiable.
  2. Diverse-and-Simple auxiliary loss: orthogonality + magnitude regularization
     on expert representation matrix W_g; replaces router z-loss.
  3. Dynamic expert add/remove: experts_mask tracks active experts. Between tasks,
     adaptive_update_experts() prunes dead experts and seeds new ones from
     token embeddings that activated no expert.

Parameter layout (inside the scan, leading depth=18 dim from nn.scan):
    mlp_1/expert_{k}/gating_einsum          (18, 2, 1024, 4096)
    mlp_1/expert_{k}/linear                 (18, 4096, 1024)
    mlp_1/expert_{k}/gating_einsum_lora_a   (18, 2, 1024, rank)   -- if lora
    mlp_1/expert_{k}/gating_einsum_lora_b   (18, 2, rank, 4096)
    mlp_1/expert_{k}/linear_lora_a          (18, 4096, rank)
    mlp_1/expert_{k}/linear_lora_b          (18, rank, 1024)
    mlp_1/router/sim_matrix                 (18, 1024, max_experts)  lecun_normal
    mlp_1/router/gates                      (18, max_experts)        zeros
    mlp_1/router/experts_mask               (18, max_experts)        binary, frozen
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Sequence
from typing import TypeAlias

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.gemma as _gemma
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

logger = logging.getLogger("openpi")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DynMoEConfig:
    """Configuration for the DynMoE action expert.

    Args:
        num_experts: Number of initially active experts (<=max_experts).
        max_experts: Pre-allocated capacity. Extra slots start masked out.
            Paper default for vision-language tasks: 4.
        aux_loss_coeff: Weight of the diverse-and-simple gating loss.
    """

    num_experts: int = 2
    max_experts: int = 4
    aux_loss_coeff: float = 1e-3


# ---------------------------------------------------------------------------
# Helpers (mirrored from gemma.py)
# ---------------------------------------------------------------------------


def _name(name: str, i: int) -> str:
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate


# ---------------------------------------------------------------------------
# Auxiliary loss
# ---------------------------------------------------------------------------


def _diverse_and_simple_gate_loss(sim_matrix, experts_mask):
    """Diverse-and-Simple gating loss from DynMoE paper (Eq. 8).

    L = ||W_g^T W_g - I_K||_F^2  +  (1/K) * sum_e ||w_{g,e}||_2

    Args:
        sim_matrix: (D, max_experts) — expert representation matrix W_g.
        experts_mask: (max_experts,) — binary mask, 1 for active experts.
    Returns:
        Scalar loss. float32.
    """
    # Zero out inactive expert columns.
    w = sim_matrix * experts_mask[None, :]  # (D, max_experts)

    # Normalize active columns to unit vectors.
    col_norms = jnp.linalg.norm(w, axis=0, keepdims=True) + 1e-8  # (1, max_experts)
    w_norm = w / col_norms  # (D, max_experts)

    # Gram matrix of normalized columns: W_g^T W_g.
    gram = jnp.einsum("de,df->ef", w_norm, w_norm)  # (max_experts, max_experts)

    # Only compare active-active pairs.
    m2 = jnp.outer(experts_mask, experts_mask)  # (max_experts, max_experts)
    identity = jnp.eye(gram.shape[0])

    # Diversity: match Gram to identity (orthogonal experts).
    diversity_loss = jnp.sum(jnp.square(gram - identity) * m2)

    # Simplicity: bounded column norms of the un-normalized matrix.
    k_active = jnp.sum(experts_mask) + 1e-8
    simplicity_loss = jnp.sum(jnp.linalg.norm(sim_matrix, axis=0) * experts_mask) / k_active

    return diversity_loss + simplicity_loss


# ---------------------------------------------------------------------------
# DynMoE Gate
# ---------------------------------------------------------------------------


class DynMoEGate(nn.Module):
    """Top-Any gate using cosine similarity and per-expert sigmoid thresholds.

    Params:
        sim_matrix  (D, max_experts) — expert representation matrix W_g
        gates       (max_experts,)   — per-expert learnable thresholds G, init=0
        experts_mask (max_experts,)  — binary mask; frozen from optimizer
    """

    features: int      # token embedding dimension D
    num_experts: int   # initially active experts
    max_experts: int   # pre-allocated capacity

    def setup(self):
        self.sim_matrix = self.param(
            "sim_matrix",
            nn.initializers.lecun_normal(in_axis=0, out_axis=1),
            (self.features, self.max_experts),
        )
        self.gates = self.param(
            "gates",
            nn.initializers.zeros,
            (self.max_experts,),
        )
        n_active = self.num_experts
        n_total = self.max_experts

        def _mask_init(key, shape):  # noqa: ARG001
            return jnp.concatenate([
                jnp.ones(n_active, dtype=jnp.float32),
                jnp.zeros(n_total - n_active, dtype=jnp.float32),
            ])

        self.experts_mask = self.param("experts_mask", _mask_init, (self.max_experts,))

    def __call__(self, x, deterministic: bool):
        """
        Args:
            x: (B, T, D) token embeddings, any dtype.
            deterministic: if True, apply top-1 fallback for tokens with k=0.
        Returns:
            gate_hard:  (B, T, max_experts) binary, dtype=x.dtype
            weights:    (B, T, max_experts) normalized routing weights, dtype=x.dtype
            sim_matrix: (D, max_experts) float32, for aux loss computation
        """
        dtype = x.dtype

        mask = self.experts_mask  # (max_experts,)

        # Cosine similarity: normalize token embeddings and expert columns.
        x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)  # (B, T, D)
        # Zero out inactive expert columns before normalizing.
        w_masked = self.sim_matrix * mask[None, :]  # (D, max_experts)
        w_norm = w_masked / (jnp.linalg.norm(w_masked, axis=0, keepdims=True) + 1e-8)
        cos_sim = jnp.einsum("...d,de->...e", x_norm, w_norm)  # (B, T, max_experts)

        # Activation probabilities; mask inactive experts to 0.
        probs = jax.nn.sigmoid(cos_sim) * mask  # (B, T, max_experts)

        # Learnable per-expert threshold; apply mask.
        threshold = jax.nn.sigmoid(self.gates) * mask  # (max_experts,)
        diff = probs - threshold  # (B, T, max_experts)

        # Straight-through estimator: forward=(diff > 0) i.e. 0 or 1, backward=identity.
        # Using (diff > 0) instead of sign(diff) is critical: sign gives {-1, 0, +1}, so
        # weights = probs * gate_hard would be negative for non-selected experts, making
        # sum(weights) potentially ≤ 0 and causing division instability → NaN.
        gate_binary = (diff > 0).astype(dtype)  # 0.0 or 1.0
        gate_hard = diff + jax.lax.stop_gradient(gate_binary - diff)  # (B, T, max_experts)

        # Top-1 fallback for tokens with k=0 (DynMoE paper Eq. 7).
        # Always applied via jnp.where to avoid Python-branch-on-traced-value (TracerBoolConversionError).
        # When any expert is selected (any_selected=True), gate_hard is unchanged.
        # When no expert is selected (any_selected=False), use the top-1 argmax.
        any_selected = jnp.any(gate_hard > 0, axis=-1, keepdims=True)  # (B, T, 1)
        top1_mask = jax.nn.one_hot(
            jnp.argmax(probs, axis=-1), self.max_experts, dtype=dtype
        )  # (B, T, max_experts)
        gate_hard = jnp.where(any_selected, gate_hard, top1_mask)

        # Routing weights: prob * gate_hard (gate_hard is 0 or 1, so weights ≥ 0).
        weights = probs * gate_hard  # (B, T, max_experts)
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-8)

        return gate_hard.astype(dtype), weights.astype(dtype), self.sim_matrix


# ---------------------------------------------------------------------------
# DynMoE Feed-Forward
# ---------------------------------------------------------------------------


class DynMoEFeedForward(nn.Module):
    """Sparse DynMoE feed-forward layer.

    Replaces a single lora.FeedForward with max_experts experts + a DynMoEGate.
    All max_experts are always computed (same as standard MoE — no sparse compute).
    Returns (output, aux_loss, gate_hard) so the block can accumulate routing stats.
    """

    features: int
    hidden_dim: int
    dyn_moe_config: DynMoEConfig
    lora_config: lora.LoRAConfig | None = None

    def setup(self):
        for k in range(self.dyn_moe_config.max_experts):
            setattr(
                self,
                f"expert_{k}",
                lora.FeedForward(
                    features=self.features,
                    hidden_dim=self.hidden_dim,
                    lora_config=self.lora_config,
                ),
            )
        self.router = DynMoEGate(
            features=self.features,
            num_experts=self.dyn_moe_config.num_experts,
            max_experts=self.dyn_moe_config.max_experts,
            name="router",
        )

    def __call__(self, x, deterministic: bool):
        """
        Returns:
            output:     (B, T, D) dtype=x.dtype
            aux_loss:   scalar float32
            gate_hard:  (B, T, max_experts) dtype=x.dtype
        """
        dtype = x.dtype

        gate_hard, weights, sim_matrix = self.router(x, deterministic)

        # Compute all expert outputs: (B, T, max_experts, D).
        expert_outputs = jnp.stack(
            [getattr(self, f"expert_{k}")(x) for k in range(self.dyn_moe_config.max_experts)],
            axis=-2,
        )

        # Weighted sum over active experts.
        output = jnp.einsum("...e,...ed->...d", weights, expert_outputs).astype(dtype)

        aux_loss = _diverse_and_simple_gate_loss(sim_matrix, self.router.experts_mask)

        assert output.dtype == dtype
        return output, aux_loss, gate_hard


# ---------------------------------------------------------------------------
# DynMoE Block
# ---------------------------------------------------------------------------


class DynMoEBlock(nn.Module):
    """Transformer block where action expert (index 1) FFW is replaced by DynMoE."""

    configs: tuple[_gemma.Config, ...]
    dyn_moe_config: DynMoEConfig

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = _gemma.Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = _gemma.RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        # Pre-allocate routing stat accumulators (shape fixed for scan compatibility).
        action_width = self.configs[1].width
        max_e = self.dyn_moe_config.max_experts
        aux_loss_scalar = jnp.zeros(())
        activation_counts = jnp.zeros((max_e,))          # per-expert token counts
        unrouted_sum = jnp.zeros((action_width,))         # sum of unrouted token embeddings
        unrouted_count = jnp.zeros(())                    # number of unrouted tokens

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = _gemma.RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                if i == 1:
                    x_pre_ffw = x  # save pre-FFW embedding for unrouted stats
                    x, aux_loss_scalar, gate_hard = DynMoEFeedForward(  # noqa: PLW2901
                        features=config.width,
                        hidden_dim=config.mlp_dim,
                        dyn_moe_config=self.dyn_moe_config,
                        lora_config=config.lora_configs.get("ffn"),
                        name=_name("mlp", i),
                    )(x, deterministic)
                    # Routing stats (summed over batch and sequence).
                    activation_counts = jnp.sum(gate_hard > 0, axis=(0, 1)).astype(jnp.float32)
                    any_sel = jnp.any(gate_hard > 0, axis=-1)           # (B, T) bool
                    unrouted_mask = (~any_sel).astype(x_pre_ffw.dtype)   # (B, T)
                    unrouted_sum = jnp.sum(
                        x_pre_ffw * unrouted_mask[..., None], axis=(0, 1)
                    )
                    unrouted_count = jnp.sum(unrouted_mask).astype(jnp.float32)
                else:
                    x = lora.FeedForward(  # noqa: PLW2901
                        features=config.width,
                        hidden_dim=config.mlp_dim,
                        name=_name("mlp", i),
                        lora_config=config.lora_configs.get("ffn"),
                    )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        # Scan output: (carry=xs, y=(kv_cache, aux_loss, act_counts, unrouted_sum, unrouted_count))
        # Routing stats are zero when expert-1 tokens are None (prefix-only pass).
        return xs, (kv_cache, aux_loss_scalar, activation_counts, unrouted_sum, unrouted_count)


# ---------------------------------------------------------------------------
# DynMoE Module
# ---------------------------------------------------------------------------

KVCache: TypeAlias = _gemma.KVCache


class DynMoEModule(nn.Module):
    """Transformer with DynMoE action expert; mirrors gemma.Module."""

    configs: Sequence[_gemma.Config]
    embed_dtype: str
    dyn_moe_config: DynMoEConfig

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = _gemma.Embedder(
            vocab_size=_gemma.PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        block_cls = nn.remat(
            DynMoEBlock,
            prevent_cse=False,
            static_argnums=(5,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dyn_moe_config=self.dyn_moe_config,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [_gemma.RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache, at.Float[at.Array, ""]]:
        """Returns (outputs, kv_cache, total_aux_loss)."""
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, (kv_cache, aux_losses, _act_counts, _unrouted_sum, _unrouted_count) = self.layers(
            embedded, kv_cache, positions, mask, adarms_cond, deterministic
        )

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        outputs = [
            f(e, a)[0] if e is not None else e
            for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ]

        total_aux_loss = jnp.mean(aux_losses)
        return outputs, kv_cache, total_aux_loss

    def compute_routing_stats(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ):
        """Run forward pass and return per-layer routing statistics.

        Returns:
            activation_counts: (depth, max_experts) — tokens activating each expert per layer
            unrouted_sum:      (depth, D) — sum of pre-FFW embeddings for unrouted tokens
            unrouted_count:    (depth,) — number of unrouted tokens per layer
        """
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        _, (_, _, activation_counts, unrouted_sum, unrouted_count) = self.layers(
            embedded, kv_cache, positions, mask, adarms_cond, deterministic
        )

        return activation_counts, unrouted_sum, unrouted_count

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None
                for u, c in zip(use_adarms, self.configs, strict=True)
            ],
        )


# ---------------------------------------------------------------------------
# Adaptive expert update (called between tasks, outside JIT)
# ---------------------------------------------------------------------------


def adaptive_update_experts(model, data_iter, num_record_batches: int = 50) -> None:
    """Update DynMoE router params after completing a training task.

    Runs num_record_batches forward passes to collect routing statistics, then:
      - Deactivates experts that never activated any token (sets experts_mask[e]=0).
      - If any tokens were unrouted (k=0 for all experts), seeds a new expert from
        their mean embedding and activates its slot (sets experts_mask[new_e]=1,
        initializes sim_matrix[:, new_e], resets gates[new_e]=0).

    The model's NNX params are updated in-place.

    Args:
        model: Pi0DynMoE NNX model. Must expose .embed_prefix, .embed_suffix,
               .PaliGemma.llm (ToNNX-wrapped DynMoEModule).
        data_iter: Iterable of (Observation, Actions) batches from the just-trained task.
        num_record_batches: Number of batches used for statistics collection.
    """
    import itertools

    from openpi.models.pi0 import make_attn_mask

    model.eval()

    # ---- 1. Collect routing stats ----------------------------------------
    total_act_counts = None   # (depth, max_experts)
    total_unrouted_sum = None  # (depth, D)
    total_unrouted_count = None  # (depth,)

    n_batches = 0
    for batch in itertools.islice(data_iter, num_record_batches):
        obs, actions = batch

        # Use zero actions/timestep so routing stats reflect observation distribution.
        noise = jnp.zeros_like(actions)
        time = jnp.full(actions.shape[0], 0.5)

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(obs, noise, time)

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        act_counts, unrouted_sum, unrouted_count = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            method="compute_routing_stats",
        )

        act_counts = jax.device_get(act_counts)
        unrouted_sum = jax.device_get(unrouted_sum)
        unrouted_count = jax.device_get(unrouted_count)

        if total_act_counts is None:
            total_act_counts = act_counts
            total_unrouted_sum = unrouted_sum
            total_unrouted_count = unrouted_count
        else:
            total_act_counts = total_act_counts + act_counts
            total_unrouted_sum = total_unrouted_sum + unrouted_sum
            total_unrouted_count = total_unrouted_count + unrouted_count
        n_batches += 1

    if n_batches == 0 or total_act_counts is None:
        logger.warning("adaptive_update_experts: no batches processed, skipping update.")
        return

    depth, max_experts = total_act_counts.shape

    # ---- 2. Extract current router params from NNX model ----------------
    # Router params live at path: PaliGemma/llm/.../mlp_1/router/{sim_matrix,gates,experts_mask}
    # After scan they have leading depth dim: (depth, ...).
    # Use state.to_pure_dict() to get a plain nested dict of raw arrays,
    # then flatten for key lookup.
    import flax.traverse_util

    import flax.nnx as nnx

    state = nnx.state(model)
    # to_pure_dict() extracts the raw array values from VariableState wrappers,
    # giving a plain nested dict that flatten_dict can handle.
    flat_state = flax.traverse_util.flatten_dict(state.to_pure_dict(), sep="/")

    # Identify router param keys (one set per layer; scan stacks them → leading dim).
    sim_matrix_key = next(
        (k for k in flat_state if k.endswith("mlp_1/router/sim_matrix")), None
    )
    gates_key = next(
        (k for k in flat_state if k.endswith("mlp_1/router/gates")), None
    )
    mask_key = next(
        (k for k in flat_state if k.endswith("mlp_1/router/experts_mask")), None
    )

    if sim_matrix_key is None or gates_key is None or mask_key is None:
        logger.warning(
            "adaptive_update_experts: could not find router params in NNX state. "
            "Keys found: %s", [k for k in flat_state if "router" in k]
        )
        return

    # Shape: (depth, D, max_experts), (depth, max_experts), (depth, max_experts)
    sim_matrix = np.array(flat_state[sim_matrix_key])
    gates = np.array(flat_state[gates_key])
    experts_mask = np.array(flat_state[mask_key])

    # Guard: if params are NaN (training diverged), skip the update.
    if np.any(np.isnan(experts_mask)) or np.any(np.isnan(sim_matrix)) or np.any(np.isnan(gates)):
        logger.warning(
            "adaptive_update_experts: router params contain NaN (training likely diverged). "
            "Skipping expert update."
        )
        return

    # ---- 3. Remove / add experts per layer --------------------------------
    before_counts = experts_mask.sum(axis=1)

    for l in range(depth):
        layer_mask = experts_mask[l]   # (max_experts,)
        layer_act = total_act_counts[l]  # (max_experts,)

        # Remove experts that never activated any token.
        for e in range(max_experts):
            if layer_mask[e] > 0 and layer_act[e] == 0:
                layer_mask[e] = 0.0

        # Add a new expert from unrouted token embeddings.
        if total_unrouted_count[l] > 0:
            inactive_slots = np.where(layer_mask == 0)[0]
            if len(inactive_slots) > 0:
                new_e = int(inactive_slots[0])
                mean_embed = total_unrouted_sum[l] / (total_unrouted_count[l] + 1e-8)
                norm = np.linalg.norm(mean_embed)
                if norm > 1e-8:
                    sim_matrix[l, :, new_e] = mean_embed / norm
                    gates[l, new_e] = 0.0
                    layer_mask[new_e] = 1.0

        experts_mask[l] = layer_mask

    after_counts = experts_mask.sum(axis=1)
    for l in range(depth):
        if before_counts[l] != after_counts[l]:
            logger.info(
                "Layer %d: experts %d → %d active",
                l, int(before_counts[l]), int(after_counts[l]),
            )

    # ---- 4. Write updated params back into model --------------------------
    flat_state[sim_matrix_key] = sim_matrix.astype(flat_state[sim_matrix_key].dtype)
    flat_state[gates_key] = gates.astype(flat_state[gates_key].dtype)
    flat_state[mask_key] = experts_mask.astype(flat_state[mask_key].dtype)

    new_pure_dict = flax.traverse_util.unflatten_dict(flat_state, sep="/")

    # replace_by_pure_dict updates the VariableState values in-place using a
    # plain nested dict of raw arrays (the inverse of to_pure_dict).
    state.replace_by_pure_dict(new_pure_dict)
    nnx.update(model, state)
    logger.info("adaptive_update_experts: router params updated for %d layers.", depth)
