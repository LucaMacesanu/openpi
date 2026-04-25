"""Pi05MoEConfig — configuration for the pi0.5-specific MoE model."""

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
    from openpi.models.pi05_moe import Pi05MoE


@dataclasses.dataclass(frozen=True)
class Pi05MoEConfig(Pi0Config):
    """pi0.5 config with a sparse MoE action expert.

    This config is intentionally pi0.5-specific: the suffix path always uses
    the pi0.5 action-only token stream plus adaRMS timestep conditioning.
    """

    pi05: bool = True
    moe_config: moe.MoEConfig = dataclasses.field(default_factory=moe.MoEConfig)

    def __post_init__(self):
        object.__setattr__(self, "pi05", True)
        object.__setattr__(self, "discrete_state_input", True)
        super().__post_init__()

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi05MoE:
        from openpi.models.pi05_moe import Pi05MoE

        return Pi05MoE(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze LLM base weights while leaving MoE router/adapters trainable.

        Note that this intentionally mirrors the existing pi0.5 MoE behavior:
        parameters outside `.*llm.*` such as vision and suffix projection layers
        remain trainable.
        """
        return nnx.Any(
            nnx.All(
                nnx_utils.PathRegex(".*llm.*"),
                nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
            ),
            nnx.All(
                nnx_utils.PathRegex(".*llm.*_1.*"),
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
                nnx.Not(nnx_utils.PathRegex(".*router.*")),
            ),
        )
