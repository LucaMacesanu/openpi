"""Pi0DynMoEConfig — configuration for the Pi0DynMoE model."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

import openpi.models.dyn_moe as dyn_moe
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_dyn_moe import Pi0DynMoE


@dataclasses.dataclass(frozen=True)
class Pi0DynMoEConfig(Pi0Config):
    """Pi0.5 config with a DynMoE (Dynamic Mixture-of-Experts) action expert.

    Extends Pi0Config so that ModelTransformFactory's isinstance check passes.
    All Pi0Config fields are inherited; dyn_moe_config is the only addition.

    Freeze filter:
      - PaliGemma backbone: frozen
      - Action expert base weights: frozen
      - Action expert LoRA adapters: trainable
      - Router (sim_matrix, gates): trainable
      - experts_mask: frozen from optimizer (mutated by adaptive_update_experts)
    """

    dyn_moe_config: dyn_moe.DynMoEConfig = dataclasses.field(
        default_factory=dyn_moe.DynMoEConfig
    )

    @override
    def create(self, rng: at.KeyArrayLike) -> Pi0DynMoE:
        from openpi.models.pi0_dyn_moe import Pi0DynMoE

        return Pi0DynMoE(self, rngs=nnx.Rngs(rng))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze PaliGemma backbone + action expert base weights.

        Trainable:
          - Action expert LoRA adapters  (.*llm.*_1.*lora.*)
          - Router sim_matrix and gates  (.*llm.*_1.*router.*)

        Frozen by optimizer (but mutable by adaptive_update_experts):
          - experts_mask                 (matched by .*router.* above)
          Note: experts_mask receives no gradient updates because it is
          binary and mutated directly. The router pattern allows sim_matrix
          and gates to train normally; experts_mask stays constant until
          adaptive_update_experts modifies it between tasks.
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
