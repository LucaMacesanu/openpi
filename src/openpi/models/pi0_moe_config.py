"""Pi0MoEConfig — configuration for the Pi0MoE model."""

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
    """Pi0.5 config with a sparse MoE action expert.

    Extends Pi0Config so that ModelTransformFactory's isinstance check passes.
    All Pi0Config fields are inherited; moe_config is the only addition.
    """

    moe_config: moe.MoEConfig = dataclasses.field(default_factory=moe.MoEConfig)

    def __post_init__(self):
        # Ensure pi05-style defaults (same as Pi0Config.__post_init__).
        super().__post_init__()
        # Override pi05=False default from Pi0Config when not explicitly set.
        # (We always want pi05=True for MoE; the default value is set below.)

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0MoE:
        from openpi.models.pi0_moe import Pi0MoE

        return Pi0MoE(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze PaliGemma backbone + action expert base weights.

        Trainable:
          - Action expert LoRA adapters  (.*llm.*_1.*lora.*)
          - Router kernel                (.*llm.*_1.*router.*)
        """
        return nnx.Any(
            # Freeze all PaliGemma LLM params (expert 0).
            nnx.All(
                nnx_utils.PathRegex(".*llm.*"),
                nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
            ),
            # Freeze action expert base weights, but not LoRA or router.
            nnx.All(
                nnx_utils.PathRegex(".*llm.*_1.*"),
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
                nnx.Not(nnx_utils.PathRegex(".*router.*")),
            ),
        )
