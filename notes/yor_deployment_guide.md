# YOR Robot Deployment Guide

Fine-tuned policy for **"place the orange cube on the plate"** (right arm only).

## Checkpoint Location

```
checkpoints/pi0_yor_right_arm/yor_right_arm_run_0/
├── 5000/    # mid-training snapshot
└── 9999/    # final checkpoint (use this)
```

Model: π₀ LoRA (`pi0_yor_right_arm` config), trained 10k steps on 48 success episodes.

---

## Serving the Policy

Run from `/local_data/lim2045/openpi`:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_yor_right_arm \
    --policy.dir=checkpoints/pi0_yor_right_arm/yor_right_arm_run_0/9999
```

The server listens on **port 8000** (WebSocket). It stays alive and handles repeated inference calls; restart it only if you change the checkpoint.

---

## Connecting from the Robot Runtime

Install the client package on the robot-side machine:

```bash
pip install openpi-client
```

Then in your robot control loop:

```python
from openpi_client import websocket_client_policy as _client
import numpy as np

policy = _client.WebsocketClientPolicy(host="<server-ip>", port=8000)

# Build one observation per control step.
obs = {
    "observation/zed":        zed_image,        # (H, W, 3) uint8, any resolution
    "observation/fish_right":  fish1_image,      # (H, W, 3) uint8, any resolution
    "observation/state":       state_17d,        # (17,) float32 — full robot state
    "prompt":                  "place the orange cube on the plate",
}

result = policy.infer(obs)
delta_joints = result["action.right_delta_joints"]  # (action_horizon, 8)
```

The server applies all pre-processing (crop, resize to 224×224, norm) internally — send raw images.

---

## Action Space

Output: `action.right_delta_joints`, shape `(action_horizon, 8)`.

| Dims | Meaning |
|---|---|
| `[0:7]` | Δrj0–Δrj6 — delta right arm joint angles (radians) |
| `[7]`   | right gripper — absolute position, range [0, 1] |

**Applying actions on the robot:**

```python
# delta_joints shape: (action_horizon, 8)
# Typical usage: re-execute every 1–2 steps (receding horizon), not the full chunk.

delta = delta_joints[0]           # take first predicted step
new_joints = current_joints + delta[:7]
gripper_cmd = delta[7]            # direct absolute gripper command
```

Delta magnitudes are small (~0.007–0.023 rad/step at 10 Hz). Apply them directly to the current joint position without scaling.

---

## Observation Space

### State vector (17D)

| Indices | Field |
|---|---|
| 0–6 | Left arm joints lj0–lj6 (unused by model, but must be present in the 17D vector) |
| 7–13 | Right arm joints rj0–rj6 |
| 14 | Left gripper |
| 15 | Right gripper |
| 16 | Lift column |

The policy internally slices indices `[7, 8, 9, 10, 11, 12, 13, 15]` (right arm + right gripper). Fill unused dims with zeros if unavailable.

### Cameras

| Key | Camera | Notes |
|---|---|---|
| `observation/zed` | ZED stereo (left RGB) | Center-cropped 1280×720 → 720×720 before resize |
| `observation/fish_right` | fish1 fisheye | Right-side view; resized as-is |

Send images as `(H, W, 3)` uint8 HWC arrays. Any resolution works; the server resizes to 224×224.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Robot jerky / overshooting | Applying the full chunk open-loop | Re-execute every 1–2 steps instead |
| Arm drifts in wrong direction | State vector indices off | Verify joints 7–13 are right arm in your state layout |
| Gripper not closing/opening | Gripper dim encoding mismatch | Check that `state[:, 15]` is [0, 1] linear (not angular) |
| `ConnectionRefused` on client | Server not running | Re-launch `serve_policy.py` on the server machine |
| OOM on server launch | Not enough GPU memory | Set `CUDA_VISIBLE_DEVICES` to a single GPU with ≥16 GB |
