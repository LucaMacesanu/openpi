"""Mixture-of-Experts action expert for Pi0.5.

Provides MoEConfig, MoEFeedForward, MoEBlock, and MoEModule.

MoEModule is a drop-in replacement for gemma.Module where the action expert's
(index 1) feed-forward layer is replaced with a sparse top-k MoE FFW layer.
Each expert is a full lora.FeedForward with optional LoRA adapters.

Parameter layout (inside the scan, so all shapes have a leading depth=18 dim):
    mlp_1/expert_{k}/gating_einsum          (18, 2, 1024, 4096)
    mlp_1/expert_{k}/linear                 (18, 4096, 1024)
    mlp_1/expert_{k}/gating_einsum_lora_a   (18, 2, 1024, rank)   -- if lora
    mlp_1/expert_{k}/gating_einsum_lora_b   (18, 2, rank, 4096)
    mlp_1/expert_{k}/linear_lora_a          (18, 4096, rank)
    mlp_1/expert_{k}/linear_lora_b          (18, rank, 1024)
    mlp_1/router/kernel                     (18, 1024, num_experts)
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import TypeAlias

import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.gemma as _gemma
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MoEConfig:
    """Configuration for the MoE action expert."""

    num_experts: int = 4
    top_k: int = 2
    router_z_loss_coeff: float = 1e-3
    load_balance_loss_weight: float = 1e-2


# ---------------------------------------------------------------------------
# Helpers (mirrored from gemma.py to avoid modifying that file)
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


def _router_z_loss(router_logits):
    """ST-MoE router z-loss: mean(log(sum(exp(logits)))^2)."""
    log_z = jax.nn.logsumexp(router_logits, axis=-1)  # (B, T)
    return jnp.mean(jnp.square(log_z))


def _load_balance_loss(router_logits):
    """Switch-style auxiliary loss for top-1 routing.

    This couples the hard top-1 dispatch fractions with the router's softmax
    probabilities so over-used experts are pushed down and under-used experts
    are pushed up.
    """
    router_logits = router_logits.astype(jnp.float32)
    probs = jax.nn.softmax(router_logits, axis=-1)
    selected_experts = jnp.argmax(probs, axis=-1)
    expert_usage = jnp.mean(
        jax.nn.one_hot(selected_experts, router_logits.shape[-1], dtype=jnp.float32),
        axis=tuple(range(selected_experts.ndim)),
    )
    mean_router_probs = jnp.mean(probs, axis=tuple(range(probs.ndim - 1)))
    return router_logits.shape[-1] * jnp.sum(expert_usage * mean_router_probs)


def _empty_router_metrics(num_experts: int):
    return {
        "expert_usage": jnp.zeros((num_experts,), dtype=jnp.float32),
        "router_prob_mean": jnp.zeros((), dtype=jnp.float32),
        "router_prob_variance": jnp.zeros((), dtype=jnp.float32),
        "router_entropy": jnp.zeros((), dtype=jnp.float32),
        "router_logits_mean": jnp.zeros((), dtype=jnp.float32),
        "router_logits_std": jnp.zeros((), dtype=jnp.float32),
        "router_logits_max": jnp.zeros((), dtype=jnp.float32),
        "router_z_loss": jnp.zeros((), dtype=jnp.float32),
        "load_balance_loss": jnp.zeros((), dtype=jnp.float32),
    }


def _router_metrics(router_logits):
    """Lightweight scalar/vector summaries of the full router distribution."""
    router_logits = router_logits.astype(jnp.float32)
    probs = jax.nn.softmax(router_logits, axis=-1)
    selected_experts = jnp.argmax(probs, axis=-1)
    expert_usage = jnp.mean(
        jax.nn.one_hot(selected_experts, router_logits.shape[-1], dtype=jnp.float32),
        axis=tuple(range(selected_experts.ndim)),
    )
    entropy = -jnp.sum(probs * jnp.log(jnp.maximum(probs, 1e-9)), axis=-1)

    return {
        "expert_usage": expert_usage * 100.0,
        "router_prob_mean": jnp.mean(probs),
        "router_prob_variance": jnp.var(probs),
        "router_entropy": jnp.mean(entropy),
        "router_logits_mean": jnp.mean(router_logits),
        "router_logits_std": jnp.std(router_logits),
        "router_logits_max": jnp.max(router_logits),
        "router_z_loss": _router_z_loss(router_logits),
        "load_balance_loss": _load_balance_loss(router_logits),
    }


# ---------------------------------------------------------------------------
# MoE Feed-Forward
# ---------------------------------------------------------------------------


class MoEFeedForward(nn.Module):
    """Sparse MoE feed-forward layer.

    Replaces a single lora.FeedForward with N experts + a linear router.
    Returns (output, router_logits) so the caller can compute z-loss.
    """

    features: int
    hidden_dim: int
    moe_config: MoEConfig
    lora_config: lora.LoRAConfig | None = None

    def setup(self):
        for k in range(self.moe_config.num_experts):
            setattr(
                self,
                f"expert_{k}",
                lora.FeedForward(
                    features=self.features,
                    hidden_dim=self.hidden_dim,
                    lora_config=self.lora_config,
                ),
            )
        # Router: zero-initialized so initial routing is uniform.
        self.router = nn.Dense(
            self.moe_config.num_experts,
            use_bias=False,
            kernel_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        # x: (B, T, D) — may be bfloat16 or float32
        dtype = x.dtype

        # Router may upcast to float32; keep logits at full precision for numerical
        # stability, but cast weights/one-hot back to x.dtype so the output
        # (which feeds back into the scan carry) stays in the original dtype.
        router_logits = self.router(x)  # (B, T, num_experts)

        # Top-K gating
        top_k_logits, top_k_indices = jax.lax.top_k(router_logits, self.moe_config.top_k)
        top_k_weights = jax.nn.softmax(top_k_logits, axis=-1).astype(dtype)  # (B, T, top_k)

        # Compute all expert outputs: stack to (B, T, num_experts, D)
        expert_outputs = jnp.stack(
            [getattr(self, f"expert_{k}")(x) for k in range(self.moe_config.num_experts)],
            axis=-2,
        )

        # Gather top-k expert outputs via one-hot indexing
        one_hot = jax.nn.one_hot(top_k_indices, self.moe_config.num_experts, dtype=dtype)  # (B, T, top_k, E)
        selected = jnp.einsum("...ke,...ed->...kd", one_hot, expert_outputs)                # (B, T, top_k, D)
        output = jnp.einsum("...k,...kd->...d", top_k_weights, selected)                    # (B, T, D)

        assert output.dtype == dtype
        return output, router_logits


# ---------------------------------------------------------------------------
# MoE Block
# ---------------------------------------------------------------------------


class MoEBlock(nn.Module):
    """Transformer block where action expert (index 1) FFW is replaced by MoE."""

    configs: tuple[_gemma.Config, ...]
    moe_config: MoEConfig

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

        out = []
        gates = []
        z_loss_scalar = jnp.zeros(())
        load_balance_loss_scalar = jnp.zeros(())
        router_metrics = _empty_router_metrics(self.moe_config.num_experts)
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = _gemma.RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                if i == 1:
                    x, router_logits = MoEFeedForward(  # noqa: PLW2901
                        features=config.width,
                        hidden_dim=config.mlp_dim,
                        moe_config=self.moe_config,
                        lora_config=config.lora_configs.get("ffn"),
                        name=_name("mlp", i),
                    )(x)
                    z_loss_scalar = _router_z_loss(router_logits)
                    load_balance_loss_scalar = _load_balance_loss(router_logits)
                    router_metrics = _router_metrics(router_logits)
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

        # Return (carry=xs, y=(kv_cache, losses, router_metrics)) for nn.scan compatibility.
        # Losses are 0.0 when expert 1 tokens are None (prefix-only pass).
        return xs, (kv_cache, z_loss_scalar, load_balance_loss_scalar, router_metrics)


# ---------------------------------------------------------------------------
# MoE Module
# ---------------------------------------------------------------------------

KVCache: TypeAlias = _gemma.KVCache


class MoEModule(nn.Module):
    """Transformer with MoE action expert; mirrors gemma.Module."""

    configs: Sequence[_gemma.Config]
    embed_dtype: str
    moe_config: MoEConfig

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
            MoEBlock,
            prevent_cse=False,
            static_argnums=(5,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            moe_config=self.moe_config,
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
        return_moe_metrics: bool = False,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache, at.Float[at.Array, ""], at.Float[at.Array, ""]]:
        """Returns (outputs, kv_cache, total_z_loss, total_load_balance_loss)."""
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, (kv_cache, z_losses, load_balance_losses, router_metrics) = self.layers(
            embedded, kv_cache, positions, mask, adarms_cond, deterministic
        )

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        outputs = [
            f(e, a)[0] if e is not None else e
            for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ]

        # z_losses has shape (depth,) — mean over layers
        total_z_loss = jnp.mean(z_losses)
        total_load_balance_loss = jnp.mean(load_balance_losses)

        if return_moe_metrics:
            return outputs, kv_cache, total_z_loss, total_load_balance_loss, router_metrics

        return outputs, kv_cache, total_z_loss, total_load_balance_loss

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
