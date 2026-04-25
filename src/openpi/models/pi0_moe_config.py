"""Pi0MoEConfig — configuration for the pi0-specific MoE model."""

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
    from openpi.models.pi0_moe import Pi0MoE


@dataclasses.dataclass(frozen=True)
class Pi0MoEConfig(Pi0Config):
    """pi0 config with a sparse MoE action expert.

    This is intentionally pi0-specific: preprocessing and suffix construction
    remain on the dense pi0 path, and the MoE router sees the full suffix
    hidden representation (state + timestep-mixed action hidden states).
    """

    pi05: bool = False
    moe_config: moe.MoEConfig = dataclasses.field(default_factory=moe.MoEConfig)

    def __post_init__(self):
        object.__setattr__(self, "pi05", False)
        object.__setattr__(self, "discrete_state_input", False)
        super().__post_init__()

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0MoE:
        from openpi.models.pi0_moe import Pi0MoE

        return Pi0MoE(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze LLM base weights while leaving MoE router/adapters trainable.

        Note that this mirrors the existing MoE freeze policy: parameters
        outside `.*llm.*` such as vision and pi0 suffix projection layers
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
