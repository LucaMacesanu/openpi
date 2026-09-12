"""Pre-decodes and caches the primary training-camera frames (observation.images.
{zed,fish0,fish1}) for the expanded 1,784-episode icl-dataset subset used by
yor_icl_pi05_expanded_frozen_vision, as a new image-mode (use_videos=False)
LeRobotDataset copy under third_party/openpi/assets/.

Motivation: job 16476252 (yor_icl_pi05_expanded_frozen_vision_full) was killed by the
cluster's admin reaper for sustained <80% GPU util. freeze_filter=".*img.*" cuts
backward-pass GPU compute per step, but live torchcodec/pyav decode of 3 cameras per
sample stays fixed -- see logs/gpu-util-16476252.csv (avg 59%) vs. job 16351039 (same
data, no freeze, avg 93.6%). This script pays the decode cost once, offline, instead of
once per epoch.

Every frame is resized to 224x224 via openpi.shared.image_tools.resize_with_pad -- the
SAME function openpi.transforms.ResizeImages calls at train time, and a no-op when
applied again to an already-224x224 input (ratio = max(w/w, h/h) == 1.0 -> zero
resize/pad -- verified against model.py's IMAGE_RESOLUTION). Also reuses
openpi.policies.yor_policy._parse_image, the exact CHW-float[0,1] -> HWC-uint8
conversion YorInputs already applies to every live-decoded frame. Together these
guarantee the cache is bit-identical to what today's live-decode path already produces,
just computed once -- not a reimplementation of either conversion.

A single shared LeRobotDataset does not support concurrent multi-process writers (see
lerobot.datasets.dataset_writer's episode-index/parquet bookkeeping), so each of
--num-shards workers decodes a disjoint subset of episodes -- greedily load-balanced by
frame count, longest-processing-time-first (same pattern as
precompute_retrieval_context.py's _assign_tasks_to_shards, weighted by episode length
instead of pool file size) -- into its OWN independent image-mode LeRobotDataset under
--shard-dir. A shard whose output directory already exists is skipped entirely (delete
it by hand to force a redo). A separate --merge pass (run once, after every shard has
finished) combines all shard datasets into the final target via
lerobot.datasets.dataset_tools.merge_datasets; --merge overwrites the final dir if
it already exists, so it's safe to rerun.

Usage (one shard of N -- see shells/slurm/victr_precompute_frame_cache_job.sh):
    third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_frame_cache.py \
        --icl-dataset-root /scratch/lim2045/icl_ws/icl-dataset \
        --episodes-json third_party/openpi/assets/yor_icl_expanded_episodes.json \
        --shard-dir outputs/frame_cache_expanded_frozen_vision/shards \
        --num-shards 16 --shard-index 0

Then, once every shard above has completed:
    third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/precompute_frame_cache.py \
        --icl-dataset-root /scratch/lim2045/icl_ws/icl-dataset \
        --episodes-json third_party/openpi/assets/yor_icl_expanded_episodes.json \
        --shard-dir outputs/frame_cache_expanded_frozen_vision/shards \
        --num-shards 16 --merge \
        --out-dir third_party/openpi/assets/yor_icl_pi05_expanded_frozen_vision_cache_224
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openpi.policies.yor_policy import _parse_image  # noqa: E402
from openpi.shared.image_tools import resize_with_pad  # noqa: E402

from lerobot.datasets.dataset_tools import merge_datasets  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

CAMERA_KEYS = ["observation.images.zed", "observation.images.fish0", "observation.images.fish1"]
NON_IMAGE_FEATURE_KEYS = [
    "observation.state",
    "raw_action.left_ee",
    "raw_action.right_ee",
    "raw_action.gripper",
    "raw_action.base_vel",
    "raw_action.lift_cmd",
    "action",
]
RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224


def _assign_episodes_to_shards(
    episode_lengths: dict[int, int], episode_order: list[int], num_shards: int
) -> list[list[int]]:
    """Greedy longest-processing-time-first bin-packing of episodes across shards,
    weighted by frame count -- see this module's docstring."""
    order = sorted(episode_order, key=lambda ep: episode_lengths[ep], reverse=True)
    bins: list[list[int]] = [[] for _ in range(num_shards)]
    bin_totals = [0] * num_shards
    for ep in order:
        j = min(range(num_shards), key=lambda b: bin_totals[b])
        bins[j].append(ep)
        bin_totals[j] += episode_lengths[ep]
    return bins


def _target_features(source_features: dict) -> dict:
    features = {}
    for key, ft in source_features.items():
        if key in CAMERA_KEYS:
            features[key] = {**ft, "dtype": "image", "shape": (RESIZE_HEIGHT, RESIZE_WIDTH, 3)}
        elif key in NON_IMAGE_FEATURE_KEYS:
            features[key] = ft
        # else: bookkeeping features (timestamp/frame_index/episode_index/index/
        # task_index) -- DEFAULT_FEATURES already adds these in LeRobotDataset.create.
    return features


def precompute_shard(
    icl_dataset_root: str, episodes_json: Path, shard_dir: Path, num_shards: int, shard_index: int
) -> None:
    all_episodes = json.loads(episodes_json.read_text())
    source = LeRobotDataset(repo_id="icl-dataset", root=icl_dataset_root, episodes=all_episodes, tolerance_s=1e-3)
    # meta.episodes' dataset_from_index/to_index are ABSOLUTE indices into the full,
    # unfiltered dataset, but source[i] (episode-filtered) indexes by RELATIVE position
    # -- see DatasetReader.get_item's docstring. abs_to_rel bridges the two.
    abs_to_rel = source.absolute_to_relative_idx

    episode_lengths = {}
    for ep in all_episodes:
        from_idx = source.meta.episodes["dataset_from_index"][ep]
        to_idx = source.meta.episodes["dataset_to_index"][ep]
        episode_lengths[ep] = to_idx - from_idx

    shards = _assign_episodes_to_shards(episode_lengths, all_episodes, num_shards)
    my_episodes = shards[shard_index]
    total_frames = sum(episode_lengths[ep] for ep in my_episodes)
    print(
        f"[shard {shard_index}/{num_shards}] {len(my_episodes)} episodes, "
        f"{total_frames} frames assigned",
        flush=True,
    )

    out_dir = shard_dir / f"shard{shard_index:03d}"
    if out_dir.exists():
        print(f"[shard {shard_index}] {out_dir} already exists, skipping", flush=True)
        return

    tmp_dir = shard_dir / f"shard{shard_index:03d}.partial"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    target = LeRobotDataset.create(
        repo_id=f"icl-dataset-frozen-vision-224-shard{shard_index:03d}",
        fps=source.meta.fps,
        features=_target_features(source.meta.features),
        root=tmp_dir,
        robot_type=source.meta.robot_type,
        use_videos=False,
        image_writer_threads=4,
    )

    t0 = time.time()
    for i, ep in enumerate(my_episodes):
        from_idx = source.meta.episodes["dataset_from_index"][ep]
        to_idx = source.meta.episodes["dataset_to_index"][ep]
        for abs_idx in range(from_idx, to_idx):
            item = source[abs_to_rel[abs_idx]]
            # HF datasets serializes shape-(1,) numeric features as bare scalars (see
            # dataset_writer.py's save_episode comment on datasets.Value) -- restore the
            # (1,) shape add_frame's validate_frame expects (matters for
            # raw_action.lift_cmd; a no-op reshape for the already-1D features).
            frame = {key: np.atleast_1d(np.asarray(item[key])) for key in NON_IMAGE_FEATURE_KEYS}
            for cam in CAMERA_KEYS:
                image = _parse_image(item[cam])  # (H, W, 3) uint8 -- exact live-path conversion
                frame[cam] = np.asarray(resize_with_pad(image, RESIZE_HEIGHT, RESIZE_WIDTH))
            frame["task"] = item["task"]
            target.add_frame(frame)
        target.save_episode()
        elapsed = time.time() - t0
        print(
            f"[shard {shard_index}] episode {i + 1}/{len(my_episodes)} "
            f"(source ep {ep}, {to_idx - from_idx} frames) done, {elapsed:.0f}s elapsed",
            flush=True,
        )

    target.finalize()
    tmp_dir.rename(out_dir)
    print(f"[shard {shard_index}] done: {out_dir}", flush=True)


def merge_shards(shard_dir: Path, num_shards: int, out_dir: Path) -> None:
    shard_paths = [shard_dir / f"shard{i:03d}" for i in range(num_shards)]
    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing shard(s), run precompute_shard for these first: {missing}")

    shard_datasets = [
        LeRobotDataset(repo_id=f"icl-dataset-frozen-vision-224-shard{i:03d}", root=shard_paths[i])
        for i in range(num_shards)
    ]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    merge_datasets(shard_datasets, output_repo_id="icl-dataset-frozen-vision-224", output_dir=out_dir)
    print(f"[merge] {sum(d.meta.total_episodes for d in shard_datasets)} episodes merged into {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--icl-dataset-root", required=True)
    parser.add_argument("--episodes-json", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.merge:
        if args.out_dir is None:
            parser.error("--out-dir is required with --merge")
        merge_shards(args.shard_dir, args.num_shards, args.out_dir)
    else:
        if args.shard_index is None:
            parser.error("--shard-index is required unless --merge is passed")
        args.shard_dir.mkdir(parents=True, exist_ok=True)
        precompute_shard(
            args.icl_dataset_root, args.episodes_json, args.shard_dir, args.num_shards, args.shard_index
        )


if __name__ == "__main__":
    main()
