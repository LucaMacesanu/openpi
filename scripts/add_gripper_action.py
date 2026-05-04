"""Add action.right_gripper column to the filtered YOR dataset.

Derives gripper action from observation.state index 15 (right_gripper).
Also regenerates meta/episodes_stats.jsonl to include the new feature.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATASET_DIR = Path("/local_data/lim2045/vla/yor_data/place_the_orange_cube_on_the_plate_success")


def main():
    parquet_dir = DATASET_DIR / "data/chunk-000"

    # 1. Add action.right_gripper column to every parquet file.
    print("Augmenting parquet files...")
    for pf in sorted(parquet_dir.glob("*.parquet")):
        df = pd.read_parquet(pf)
        if "action.right_gripper" not in df.columns:
            df["action.right_gripper"] = df["observation.state"].apply(
                lambda s: np.array([np.asarray(s)[15]], dtype=np.float32)
            )
            df.to_parquet(pf, index=False)
        print(f"  {pf.name}: right_gripper range [{df['action.right_gripper'].apply(lambda x: x[0]).min():.3f}, "
              f"{df['action.right_gripper'].apply(lambda x: x[0]).max():.3f}]")

    # 2. Update meta/info.json.
    info_path = DATASET_DIR / "meta/info.json"
    info = json.loads(info_path.read_text())
    if "action.right_gripper" not in info["features"]:
        info["features"]["action.right_gripper"] = {
            "shape": [1],
            "dtype": "float32",
            "names": ["right_gripper"],
        }
        info_path.write_text(json.dumps(info, indent=2))
        print("\nUpdated meta/info.json")

    # 3. Regenerate meta/episodes_stats.jsonl.
    print("\nRegenerating episodes_stats.jsonl...")
    episodes = [json.loads(l) for l in (DATASET_DIR / "meta/episodes.jsonl").read_text().splitlines()]
    array_features = [
        k for k, v in info["features"].items()
        if v["dtype"] not in ("video",) and k not in ("task_index", "timestamp", "frame_index", "episode_index")
    ]

    stats_path = DATASET_DIR / "meta/episodes_stats.jsonl"
    stats_path.unlink(missing_ok=True)
    for ep in episodes:
        idx = ep["episode_index"]
        df = pd.read_parquet(parquet_dir / f"episode_{idx:06d}.parquet")
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
                "count": [len(arr)],
            }
        with open(stats_path, "a") as f:
            f.write(json.dumps({"episode_index": idx, "stats": ep_stats}) + "\n")
    print(f"  Written {len(episodes)} episode entries.")
    print("\nDone.")


if __name__ == "__main__":
    main()
