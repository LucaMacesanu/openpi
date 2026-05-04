"""Compute and store action.right_delta_joints in the filtered YOR success dataset.

Action layout (8D):
  [Δrj0, Δrj1, Δrj2, Δrj3, Δrj4, Δrj5, Δrj6, right_gripper_absolute]

  Δrj0-6 = state[t+1, 7:14] - state[t, 7:14]   (last frame gets zeros)
  right_gripper_absolute = state[t, 15]

This matches pi0/pi0.5 pretraining: delta joint angles + absolute gripper.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATASET_DIR = Path("/local_data/lim2045/vla/yor_data/place_the_orange_cube_on_the_plate_success")
RIGHT_JOINT_INDICES = slice(7, 14)   # rj0-rj6 in the 17-dim state
GRIPPER_INDEX = 15                    # right_gripper in the 17-dim state


def compute_delta_joints(df: pd.DataFrame) -> pd.Series:
    states = np.stack(df["observation.state"].values).astype(np.float32)  # (T, 17)
    right_joints = states[:, RIGHT_JOINT_INDICES]                          # (T, 7)

    deltas = np.zeros_like(right_joints)                                   # last frame = 0
    deltas[:-1] = right_joints[1:] - right_joints[:-1]

    gripper = states[:, GRIPPER_INDEX : GRIPPER_INDEX + 1]                 # (T, 1)

    actions = np.concatenate([deltas, gripper], axis=1).astype(np.float32)  # (T, 8)
    return pd.Series([actions[i] for i in range(len(actions))], index=df.index)


def main():
    parquet_dir = DATASET_DIR / "data/chunk-000"

    # 1. Add action.right_delta_joints to every episode parquet.
    print("Augmenting parquet files...")
    for pf in sorted(parquet_dir.glob("*.parquet")):
        df = pd.read_parquet(pf)
        if "action.right_delta_joints" not in df.columns:
            df["action.right_delta_joints"] = compute_delta_joints(df)
            df.to_parquet(pf, index=False)

        col = df["action.right_delta_joints"]
        arr = np.stack(col.values)
        delta_mag = np.abs(arr[:, :7]).mean()
        gripper_range = (arr[:, 7].min(), arr[:, 7].max())
        print(
            f"  {pf.name}: mean |Δjoint|={delta_mag:.4f}, "
            f"gripper=[{gripper_range[0]:.3f}, {gripper_range[1]:.3f}]"
        )

    # 2. Update meta/info.json.
    info_path = DATASET_DIR / "meta/info.json"
    info = json.loads(info_path.read_text())
    if "action.right_delta_joints" not in info["features"]:
        info["features"]["action.right_delta_joints"] = {
            "shape": [8],
            "dtype": "float32",
            "names": ["delta_rj0", "delta_rj1", "delta_rj2", "delta_rj3",
                      "delta_rj4", "delta_rj5", "delta_rj6", "right_gripper"],
        }
        info_path.write_text(json.dumps(info, indent=2))
        print("\nUpdated meta/info.json")

    # 3. Regenerate meta/episodes_stats.jsonl.
    print("\nRegenerating episodes_stats.jsonl...")
    episodes = [json.loads(l) for l in (DATASET_DIR / "meta/episodes.jsonl").read_text().splitlines()]
    array_features = [
        k for k, v in info["features"].items()
        if v["dtype"] not in ("video",)
        and k not in ("task_index", "timestamp", "frame_index", "episode_index")
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
                "mean":  arr.mean(0).tolist(),
                "std":   arr.std(0).tolist(),
                "min":   arr.min(0).tolist(),
                "max":   arr.max(0).tolist(),
                "count": [len(arr)],
            }
        with open(stats_path, "a") as f:
            f.write(json.dumps({"episode_index": idx, "stats": ep_stats}) + "\n")

    print(f"  Written {len(episodes)} episode entries.")
    print("\nDone.")


if __name__ == "__main__":
    main()
