"""Data-loading transform for Reward-Aligned Behavior Cloning (RA-BC, notes/
reward_aligned_bc.md at the viktr repo root -- design doc summarizing
third_party/opensarm/2509.25358 Sec 3.2).

Per-frame RA-BC weights are precomputed offline by
scripts/precompute_rabc_weights.py (viktr-side venv, since that's where the
phi/progress-value sources live) into one <episode_index>.npy (T,) float32 curve per
episode. This module just does the O(1) training-time lookup, mirroring
yor_retrieval.py's _load_precomputed_episode/_robodopamine_curve convention -- no
runtime math, all of Eq. 6-9 already baked into the precomputed curve.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path

import numpy as np

from openpi import transforms


@functools.lru_cache(maxsize=None)
def _load_weight_curve(weights_dir: str, episode_index: int) -> np.ndarray:
    return np.load(Path(weights_dir) / f"{episode_index}.npy", mmap_mode="r")


@dataclasses.dataclass(frozen=True)
class RabcWeightInputs(transforms.DataTransformFn):
    """Must run BEFORE openpi.policies.yor_policy.YorInputs in data_transforms.inputs:
    needs the item's raw episode_index/frame_index keys, which YorInputs replaces."""

    weights_dir: str

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        curve = _load_weight_curve(self.weights_dir, int(data["episode_index"]))
        weight = np.float32(curve[int(data["frame_index"])])
        return {**data, "rabc_weight": weight}
