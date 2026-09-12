"""One-time resize of a VICTR retrieval pool's chunk images to 224x224.

Pools built by scripts/build_retrieval_pool.py store chunk images at icl-dataset's
native camera resolution (480x640) -- but openpi.policies.yor_retrieval.
build_context_block resizes every chunk image down to 224x224 on EVERY access (via
resize_with_pad_torch), so the extra resolution is pure waste: for the expanded task
set (31 tasks), this makes third_party/openpi/assets/victr_icl_pool_expanded 176GB on
disk (one task's pool alone: 36GB), which was the real cause behind
scripts/precompute_retrieval_context.py's repeated OOMs (job 16350893 et al) --
loading even one such pool costs tens of GB of RSS per process, and the original
interleaved episode-level sharding has many shard processes independently load their
own full copy of the same big pool at around the same time.

This script resizes once, using the EXACT SAME resize_with_pad_torch(images, 224, 224)
call build_context_block makes at read time -- so a chunk that goes through this
script and then through build_context_block again is resized twice with the same
target size, which is a no-op the second time (verified: resize_with_pad_torch is
idempotent when input size already equals the target). Output is therefore guaranteed
bit-identical to today's live-resize behavior, just computed once instead of on every
access. ~6.1x fewer pixels (480*640 / 224*224) shrinks pools roughly 6x.

Only the primary camera is resized (that's all build_retrieval_pool.py ever stores in
a chunk -- build_context_block only reads chunk.images[primary_camera] too, see
load_episode_arrays' docstring in viktr/data/icl_dataset.py).

Processes one task at a time (loads, resizes, writes, frees) so peak memory is bounded
by the single largest task's pool, not the whole pool directory -- run under a SLURM
job with generous memory (see victr_resize_pool_job.sh), not on the login node (no
cgroup memory accounting there -- a prior unrelated script was silently OOM-killed
running directly on the login node with no traceback).

Usage:
    third_party/openpi/.venv/bin/python3 third_party/openpi/scripts/resize_pool_images.py \
        --pool-dir third_party/openpi/assets/victr_icl_pool_expanded \
        --out-dir third_party/openpi/assets/victr_icl_pool_expanded_224 \
        --primary-camera observation.images.zed
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openpi.shared.image_tools import resize_with_pad_torch  # noqa: E402


def resize_pool_file(in_path: Path, out_path: Path, primary_camera: str, height: int, width: int) -> None:
    with open(in_path, "rb") as f:
        pool = pickle.load(f)

    for chunk in pool["chunks"]:
        assert set(chunk["images"]) == {primary_camera}, (
            f"expected only {primary_camera!r} stored per chunk, got {list(chunk['images'])} "
            "-- build_retrieval_pool.py is documented to only ever store the primary camera; "
            "this script only resizes that key and would silently drop any others"
        )
        images = chunk["images"][primary_camera]  # (L, H, W, 3) uint8
        resized = resize_with_pad_torch(torch.from_numpy(images), height, width).numpy()
        chunk["images"] = {primary_camera: resized}

    with open(out_path, "wb") as f:
        pickle.dump(pool, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--primary-camera", default="observation.images.zed")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    in_paths = sorted(args.pool_dir.glob("*.pkl"))
    print(f"[resize_pool_images] {len(in_paths)} pool files under {args.pool_dir}")

    for i, in_path in enumerate(in_paths):
        out_path = args.out_dir / in_path.name
        if out_path.exists():
            print(f"[{i + 1}/{len(in_paths)}] {in_path.name}: already done, skipping")
            continue
        in_size_gb = in_path.stat().st_size / 1e9
        print(f"[{i + 1}/{len(in_paths)}] {in_path.name} ({in_size_gb:.1f}GB)...", flush=True)
        resize_pool_file(in_path, out_path, args.primary_camera, args.height, args.width)
        out_size_gb = out_path.stat().st_size / 1e9
        print(f"[{i + 1}/{len(in_paths)}] {in_path.name}: {in_size_gb:.1f}GB -> {out_size_gb:.1f}GB")


if __name__ == "__main__":
    main()
