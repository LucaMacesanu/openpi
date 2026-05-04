"""DynMoEWeightLoader — loads pi05_base weights and fans out action expert FFW to DynMoE experts.

Fan-out strategy
----------------
Identical to MoEWeightLoader, with two differences:

1. Fan-out count is max_experts (not num_experts) — pre-populate all pre-allocated
   expert slots so inactive slots have valid weight values from the start.

2. Router params (sim_matrix, gates, experts_mask) are NOT inserted here.
   They are filled from the model's own initialization via the _merge_params
   fallback at the bottom (sim_matrix: lecun_normal; gates: zeros;
   experts_mask: binary with num_experts active slots).

Usage
-----
    DynMoEWeightLoader(
        base_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_experts=2,
        max_experts=4,
    )
"""

from __future__ import annotations

import dataclasses
import re

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download
import openpi.training.weight_loaders as weight_loaders


@dataclasses.dataclass(frozen=True)
class DynMoEWeightLoader(weight_loaders.WeightLoader):
    """Loads pi05_base weights and fans out each action-expert FFW layer to max_experts slots.

    All max_experts expert slots (including initially inactive ones) are
    pre-populated with a copy of the base action-expert weights so that
    adaptive_update_experts() can activate any slot cleanly.

    Args:
        base_loader: A CheckpointWeightLoader pointing at the pi05_base checkpoint.
        num_experts: Number of initially active experts (must match DynMoEConfig.num_experts).
        max_experts: Total pre-allocated expert slots (must match DynMoEConfig.max_experts).
    """

    base_loader: weight_loaders.CheckpointWeightLoader
    num_experts: int = 2
    max_experts: int = 4

    def load(self, params: at.Params) -> at.Params:
        # Load the raw base checkpoint (np.ndarray, no LoRA filling yet).
        base_params = _model.restore_params(
            download.maybe_download(self.base_loader.params_path),
            restore_type=np.ndarray,
        )

        flat_base = flax.traverse_util.flatten_dict(base_params, sep="/")
        flat_model = flax.traverse_util.flatten_dict(params, sep="/")

        result: dict[str, np.ndarray] = {}

        # Pattern: a top-level mlp_1 parameter, i.e. the path ends in
        # .../mlp_1/<leaf_name> with no further "/" inside <leaf_name>.
        mlp1_leaf_re = re.compile(r"(.*)/mlp_1/([^/]+)$")

        for k, v in flat_base.items():
            m = mlp1_leaf_re.match(k)
            if m:
                # Fan out to ALL max_experts slots (including inactive ones).
                # Inactive slots are masked out during routing but have valid
                # weights so that adaptive_update_experts() can activate them.
                prefix, leaf = m.group(1), m.group(2)
                for ek in range(self.max_experts):
                    new_key = f"{prefix}/mlp_1/expert_{ek}/{leaf}"
                    if new_key in flat_model:
                        tgt_dtype = flat_model[new_key].dtype
                        result[new_key] = v.astype(tgt_dtype)
            elif k in flat_model:
                # All other params (PaliGemma backbone, norms, embedder, …).
                tgt_dtype = flat_model[k].dtype
                result[k] = v.astype(tgt_dtype) if v.dtype != tgt_dtype else v

        # Fill every key not yet in result from the model's reference params.
        # This covers:
        #   - mlp_1/router/sim_matrix    (lecun_normal from model init)
        #   - mlp_1/router/gates         (zeros from model init)
        #   - mlp_1/router/experts_mask  (binary from model init)
        #   - mlp_1/expert_k/lora_*      (zeros from model init)
        #   - action_in_proj / time_mlp_* (random from model init)
        for k, v in flat_model.items():
            if k not in result:
                result[k] = v

        return flax.traverse_util.unflatten_dict(result, sep="/")
