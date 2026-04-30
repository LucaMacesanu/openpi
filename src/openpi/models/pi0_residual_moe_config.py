"""Pi0ResidualMoEConfig — configuration for the pi0 residual-MoE model."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

import openpi.models.moe as moe
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_residual_moe import Pi0ResidualMoE


@dataclasses.dataclass(frozen=True)
class Pi0ResidualMoEConfig(Pi0Config):
    """pi0 config with a dense action FFN plus a routed residual expert branch."""

    pi05: bool = False
    moe_config: moe.ResidualMoEConfig = dataclasses.field(default_factory=moe.ResidualMoEConfig)

    def __post_init__(self):
        object.__setattr__(self, "pi05", False)
        object.__setattr__(self, "discrete_state_input", False)
        super().__post_init__()

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0ResidualMoE:
        from openpi.models.pi0_residual_moe import Pi0ResidualMoE

        return Pi0ResidualMoE(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze dense LLM weights while keeping router/residual trainable."""
        return nnx.Any(
            nnx.All(
                nnx_utils.PathRegex(".*llm.*"),
                nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
            ),
            nnx.All(
                nnx_utils.PathRegex(".*llm.*_1.*"),
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
                nnx.Not(nnx_utils.PathRegex(".*router.*")),
                nnx.Not(nnx_utils.PathRegex(".*expert_[0-9]+.*")),
            ),
        )
