"""Copy a LeRobot v2.1 dataset, keeping only episodes labeled 'success'.

Also generates meta/episodes_stats.jsonl (required by recent LeRobot for local loading).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source dataset directory")
    ap.add_argument("--dst", required=True, help="Destination dataset directory")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    episodes = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text().splitlines()]
    success_eps = [e for e in episodes if e["success"] == "success"]
    print(f"Keeping {len(success_eps)}/{len(episodes)} success episodes.")

    old_to_new = {e["episode_index"]: i for i, e in enumerate(success_eps)}

    # meta/
    (dst / "meta").mkdir(exist_ok=True)
    info = json.loads((src / "meta/info.json").read_text())
    info["total_episodes"] = len(success_eps)
    info["total_frames"] = sum(e["length"] for e in success_eps)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2))
    shutil.copy(src / "meta/tasks.jsonl", dst / "meta/tasks.jsonl")
    with open(dst / "meta/episodes.jsonl", "w") as f:
        for new_idx, ep in enumerate(success_eps):
            ep = dict(ep, episode_index=new_idx)
            f.write(json.dumps(ep) + "\n")

    # data/
    (dst / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    for ep in success_eps:
        old_i, new_i = ep["episode_index"], old_to_new[ep["episode_index"]]
        src_f = src / f"data/chunk-000/episode_{old_i:06d}.parquet"
        dst_f = dst / f"data/chunk-000/episode_{new_i:06d}.parquet"
        df = pd.read_parquet(src_f)
        df["episode_index"] = new_i
        df.to_parquet(dst_f, index=False)
        print(f"  {old_i:03d} -> {new_i:03d}  ({len(df)} frames)")

    # videos/
    for cam_dir in sorted((src / "videos/chunk-000").iterdir()):
        out_cam = dst / "videos/chunk-000" / cam_dir.name
        out_cam.mkdir(parents=True, exist_ok=True)
        for ep in success_eps:
            old_i, new_i = ep["episode_index"], old_to_new[ep["episode_index"]]
            src_v = cam_dir / f"episode_{old_i:06d}.mp4"
            dst_v = out_cam / f"episode_{new_i:06d}.mp4"
            if src_v.exists():
                shutil.copy(src_v, dst_v)
        print(f"  Camera {cam_dir.name}: {len(success_eps)} videos copied")

    # episodes_stats.jsonl — required by LeRobot >= v2.1 for local loading.
    array_features = [
        k for k, v in info["features"].items()
        if v["dtype"] not in ("video",) and k not in ("task_index", "timestamp", "frame_index", "episode_index")
    ]
    stats_path = dst / "meta/episodes_stats.jsonl"
    for new_idx, ep in enumerate(success_eps):
        df = pd.read_parquet(dst / f"data/chunk-000/episode_{new_idx:06d}.parquet")
        ep_stats = {}
        for feat in array_features:
            if feat not in df.columns:
                continue
            arr = np.stack(df[feat].values).astype(np.float32)
            ep_stats[feat] = {
                "mean": arr.mean(0).tolist(),
                "std":  arr.std(0).tolist(),
                "min":  arr.min(0).tolist(),
                "max":  arr.max(0).tolist(),
                "count": [len(arr)],  # scalar total-timestep count as required by LeRobot
            }
        with open(stats_path, "a") as f:
            f.write(json.dumps({"episode_index": new_idx, "stats": ep_stats}) + "\n")
    print(f"  episodes_stats.jsonl written ({len(success_eps)} entries)")

    print(f"\nDone. Filtered dataset written to {dst}")
    print(f"  {len(success_eps)} episodes, {info['total_frames']} frames")


if __name__ == "__main__":
    main()
