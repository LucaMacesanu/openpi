# YOR Inference Pipeline — Plan

End-to-end pipeline that runs on this Jetson Orin to drive the YOR robot
through the fine-tuned `pi0_yor_right_arm` policy. **Two machines only:**
the orin runs *everything* — cameras, preprocessing, model inference,
TCP command publishing — and the Raspberry Pi at `192.168.0.198`
executes joint commands on the right arm.

---

## ▶︎ Current state (paused here — 2026-05-04)

**Status: BLOCKED on bringing the YOR fork onto this machine.** No code
has been written yet; this file is the design doc to resume from.

### What's on the orin right now

- `/home/ai4ce-orin/manipulation_ws/openpi` — mainline openpi base.
  Contains `scripts/serve_policy.py`, `packages/openpi-client/…`, all
  the training/policy infra for ALOHA/DROID/Libero. No YOR-specific
  code.
- `/home/ai4ce-orin/manipulation_ws/openpi/checkpoints/` — exists but
  **empty**.
- `/home/ai4ce-orin/manipulation_ws/openpi/notes/yor_deployment_guide.md`
  — the spec we're implementing against (schema, prompt, action
  layout, preprocessing summary).
- `/home/ai4ce-orin/yor_ws/` — `collect.py`, `camera_viewer.py`,
  `visualize.py`, teleop subscriber guide. This is the source of
  truth for camera device indices (`FISH1_DEV=6` for the right hand,
  `cv2.ROTATE_180`) and the 17-D state vector layout.
- `/home/ai4ce-orin/dejaview_ws/DejaView/nomad_route.py` — the
  `TCPServer` class pattern we're reusing for the Pi command stream.
- Cameras (ZED + 3 fisheyes) and the YOR teleop publisher
  (`192.168.0.198:5558` ZMQ) are physically wired and known to work
  with `collect.py`.

### What's missing — must be fetched from the training host

These four assets must be copied onto this orin before any inference
can run. The training host paths come from
`notes/yor_deployment_guide.md` ("Run from `/local_data/lim2045/openpi`").

| # | Asset | Training-host path | Where it goes on the orin |
|---|-------|--------------------|---------------------------|
| 1 | Final checkpoint | `/local_data/lim2045/openpi/checkpoints/pi0_yor_right_arm/yor_right_arm_run_0/9999/` | `./checkpoints/pi0_yor_right_arm/yor_right_arm_run_0/9999/` |
| 2 | YOR policy module (input/output transforms — crop, resize, normalise, slice 17-D state, name `action.right_delta_joints`) | likely `src/openpi/policies/yor_policy.py` (or similar — `grep -rn "yor\|right_delta_joints\|fish_right" src/openpi/`) | `src/openpi/policies/yor_policy.py` |
| 3 | Training config registration for `pi0_yor_right_arm` | a patch in `src/openpi/training/config.py` registering the config name and pointing at #2's transforms | merge into `src/openpi/training/config.py` |
| 4 | Any model-specific assets the checkpoint expects (norm stats, tokenizer config, etc.) | usually inside the checkpoint dir — `assets/` subfolder | preserved by copying #1 verbatim |

**Easiest path:** rsync the entire training-host repo as a sibling
checkout, diff against this orin's openpi tree, and cherry-pick #2/#3
(or just rsync everything if there are no orin-specific changes
here yet).

```bash
# from orin, fetching from training host
rsync -avz <training-host>:/local_data/lim2045/openpi/checkpoints/pi0_yor_right_arm \
    /home/ai4ce-orin/manipulation_ws/openpi/checkpoints/

rsync -avz <training-host>:/local_data/lim2045/openpi/src/openpi/ \
    /tmp/openpi_yor_src/
diff -ruN src/openpi/ /tmp/openpi_yor_src/   # review before merging
```

### Other blockers / open coordination items

- **Pi-side state stream.** The teleop publisher only emits during an
  active episode (per `teleop_episode_subscriber_guide.txt`). Inference
  needs continuous state. Recommended: ask whoever owns the Pi to add
  a `state/live` topic on the same ZMQ port that publishes at ≥10 Hz
  regardless of episode state. Without it, we'll need a workaround
  (warmup pose / poll endpoint).
- **Pi-side action executor.** Needs to be written on the Pi. Schema
  in §4 below. Coordinate the schema before they start coding so we
  don't have to revise twice.
- **Channel order + normalisation.** Cannot be answered until #2 (YOR
  policy module) is on the orin. The plan defaults to running the
  websocket server locally precisely so we can avoid having to know.

### Resume checklist (when you come back to this)

1. Confirm checkpoint copied: `ls
   checkpoints/pi0_yor_right_arm/yor_right_arm_run_0/9999/` is
   non-empty.
2. Confirm YOR policy + config code merged: `python -c "from
   openpi.training import config as c;
   print(c.get_config('pi0_yor_right_arm'))"` runs without
   `KeyError`.
3. Smoke-test the server: `uv run scripts/serve_policy.py
   policy:checkpoint --policy.config=pi0_yor_right_arm
   --policy.dir=checkpoints/pi0_yor_right_arm/yor_right_arm_run_0/9999`
   should bind to `:8000` and log a metadata header.
4. Then proceed to "Order of implementation" §1 below.

---

## High-level data flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Jetson Orin (this machine, AGX, CUDA capable)                              │
│                                                                             │
│  ┌────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐    │
│  │ ZED HD720      │  │ fish1 (right)    │  │ commlink Sub             │    │
│  │ V4L2 BGRA      │  │ /dev/video6 BGR  │  │ teleop/episode :5558     │    │
│  │   (1280×720)   │  │   (640×480)      │  │ → 17D state              │    │
│  └───────┬────────┘  └────────┬─────────┘  └────────────┬─────────────┘    │
│          │ raw                │ raw, 180° rot           │ joints+grip+lift │
│          ▼                    ▼                         ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Preprocessor (this script — see §3)                                 │   │
│  │  ZED:    BGRA → BGR → center-crop 1280×720 → 720×720 → resize 224  │   │
│  │  fish1:  BGR  →                          → resize 224              │   │
│  │  both:   uint8 HWC                                                  │   │
│  │  state:  build 17-D float32 vector                                  │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    ▼                                        │
│                        ┌──────────────────────────┐                         │
│                        │ pi0_yor_right_arm policy │ ← checkpoint /9999      │
│                        │ (in-process, JAX/Flax,   │                         │
│                        │  GPU)                    │                         │
│                        └────────────┬─────────────┘                         │
│                                     │ action.right_delta_joints (H, 8)     │
│                                     ▼                                        │
│                        ┌──────────────────────────┐                         │
│                        │ TCP cmd server :5006     │ (newline JSON,         │
│                        │ (pattern: nomad_route)   │  one msg / tick)       │
│                        └────────────┬─────────────┘                         │
└─────────────────────────────────────┼───────────────────────────────────────┘
                                      │ TCP 5006
                                      ▼
                       ┌──────────────────────────────┐
                       │ Raspberry Pi @ 192.168.0.198 │
                       │  • teleop publisher (5558)   │
                       │  • NEW cmd executor:         │
                       │    new_joints = q + Δq[0:7]  │
                       │    gripper    = Δq[7]        │
                       └──────────────────────────────┘
```

The orin **does the cropping, resizing, normalisation, and channel
conversion itself** — there is no remote inference service. Whether we
use openpi's `WebsocketPolicyServer` on `localhost:8000` or call the
policy in-process is a structural choice (see §4); either way every
tensor passes through this machine.

## What's missing on the orin right now

`/home/ai4ce-orin/manipulation_ws/openpi` has the openpi base (mainline
fork) but **none of the YOR-specific pieces**:

| Asset                                            | Where it lives now                  | Why we need it                                                |
|--------------------------------------------------|-------------------------------------|---------------------------------------------------------------|
| `checkpoints/pi0_yor_right_arm/.../9999/`        | training host (`/local_data/lim2045/openpi`) | model params + assets                                |
| `pi0_yor_right_arm` entry in `openpi.training.config` | training-host fork only          | tells `serve_policy.py`/the loader the model + transforms shape |
| YOR policy module (input/output transforms — crop, resize, slice state, name actions) | training-host fork only | turns raw obs into model input and names output dims |
| `notes/yor_deployment_guide.md`                  | here                                | already in tree                                                |

**Action item before any code runs:** rsync the checkpoint + training-host
openpi fork onto the orin (or merge the YOR commits into this tree).
Without the YOR transform module the openpi server cannot interpret
`observation/zed`, `observation/fish_right`, or
`action.right_delta_joints` — those keys are defined by that module.

## Components to build

### 1. Camera + state collector (orin script)

New file: `scripts/run_yor_inference.py`. Threading model copied from
`collect.py`:

- **`zed_thread`** — ZED SDK, `HD720 @ 15 fps`, retrieve LEFT view as
  BGRA, drop alpha, push to `_latest["zed"]` as BGR `(720,1280,3)
  uint8`.
- **`fish1_thread`** — `cv2.VideoCapture(6, cv2.CAP_V4L2)`, MJPG
  `640×480 @ 30 fps`, apply `cv2.ROTATE_180` (matches collect.py — the
  training data is rotated), push to `_latest["fish1"]` as BGR
  `(480,640,3) uint8`.
- **`state_thread`** — `commlink.Subscriber(YOR_IP, port=5558,
  topics=["teleop/episode"])`. Build the 17-D state from each frame:
  ```
  state[0:7]   = left_joints
  state[7:14]  = right_joints
  state[14]    = left_gripper
  state[15]    = right_gripper
  state[16]    = lift
  ```
  Cache the most recent value with its receive timestamp. **The
  publisher only emits during an active episode** (per
  `teleop_episode_subscriber_guide.txt`), which is a problem for an
  inference loop that drives the robot when no episode is running. See
  "Open questions" — recommended fix is a `state/live` topic on the
  Pi.

Single shared `_latest` dict + `threading.Lock` — same idiom as
`collect.py` lines 44–67. Tick the main loop at **10 Hz**, the rate the
policy was trained at.

### 2. Preprocessing (do it ourselves — it lives in this script)

This is the part the deployment guide hand-waves as "the server applies
all pre-processing internally". Since the server is *also* this machine,
we have to reproduce the training-time transforms exactly. Best source
of truth is the YOR policy module on the training host; until that's
copied here, the guide tells us:

```python
# --- ZED ---
# input:  BGRA (720, 1280, 4), or BGR (720, 1280, 3) after stripping alpha
zed_bgr = cv2.cvtColor(zed_bgra, cv2.COLOR_BGRA2BGR)            # (720,1280,3)
# center crop to a square 720x720
h, w = zed_bgr.shape[:2]                                        # 720, 1280
x0   = (w - h) // 2                                             # 280
zed_sq = zed_bgr[:, x0:x0+h]                                    # (720, 720, 3)
zed_224 = cv2.resize(zed_sq, (224, 224), interpolation=cv2.INTER_AREA)

# --- fish_right ---
# input: BGR (480, 640, 3), already 180° rotated
fish_224 = cv2.resize(fish1_bgr, (224, 224), interpolation=cv2.INTER_AREA)

# --- channel order ---
# pi0 base was trained on RGB. Convert before passing to the model.
zed_rgb  = cv2.cvtColor(zed_224,  cv2.COLOR_BGR2RGB)
fish_rgb = cv2.cvtColor(fish_224, cv2.COLOR_BGR2RGB)
```

Open question: whether the training pipeline expected uint8 [0,255] or
float32 [0,1]/[-1,1] and whether ImageNet mean/std normalisation was
applied. **The YOR policy module is authoritative — do not guess.**
Copy that module over, read its `make_*_transforms` (or equivalent),
and replicate in this script. Until then, the safest fallback is to
*not* normalise here and instead use the websocket server path (§4
option A), which calls the same transforms the model was trained
with — that removes one whole class of bugs.

State preprocessing: cast to `float32`. The model internally slices
`[7,8,9,10,11,12,13,15]` (right arm + right gripper) per the
deployment guide, so left-arm/lift values can be zero if unavailable —
but the **17-D shape must be exact**.

Prompt: hard-code `"place the orange cube on the plate"` (CLI flag
with that default).

### 3. Inference call

Two viable paths. Pick one and stick with it.

**Option A (recommended) — run openpi's websocket server on this
machine, connect to `localhost:8000` from the same script.**

Pros: server applies the same transforms used at training time (zero
risk of preprocessing drift), msgpack handles arbitrary obs shapes,
matches the deployment guide verbatim.

Cons: two processes (server + client), inference + serialisation
round-trip overhead, server expects the YOR policy module to be
importable.

```bash
# terminal 1 (server)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_yor_right_arm \
    --policy.dir=checkpoints/pi0_yor_right_arm/yor_right_arm_run_0/9999

# terminal 2 (client + cameras + TCP publisher)
uv run scripts/run_yor_inference.py \
    --policy-host=localhost --policy-port=8000 \
    --pi-cmd-port=5006
```

In `run_yor_inference.py`:
```python
from openpi_client import websocket_client_policy as _wcp
policy = _wcp.WebsocketClientPolicy(host="localhost", port=8000)
result = policy.infer({
    "observation/zed":        zed_raw_bgr,        # server crops+resizes
    "observation/fish_right": fish_raw_bgr,
    "observation/state":      state_17d.astype(np.float32),
    "prompt":                 "place the orange cube on the plate",
})
delta = result["action.right_delta_joints"][0]  # (8,) — first chunk step
```

**Option B — load the policy in-process, no websocket layer.**

Pros: single process, lowest latency, no IPC.

Cons: we must reproduce the exact preprocessing the YOR policy module
does. Easy to get wrong silently. Only worth it if §A's RTT is the
bottleneck.

```python
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
policy = _policy_config.create_trained_policy(
    _config.get_config("pi0_yor_right_arm"),
    "checkpoints/pi0_yor_right_arm/yor_right_arm_run_0/9999",
    default_prompt="place the orange cube on the plate",
)
result = policy.infer(obs)   # obs already preprocessed
```

Default to A. Switch to B only after measuring §A latency on this orin.

### 4. TCP command publisher

Reuse the `TCPServer` class pattern from
`/home/ai4ce-orin/dejaview_ws/DejaView/nomad_route.py` (lines 240–291).
Listen on `0.0.0.0:5006`; the Pi connects as the client. One
newline-delimited JSON per inference cycle:

```json
{
  "seq": 42,
  "timestamp": 1718000000.123,
  "right_delta_joints": [d0, d1, d2, d3, d4, d5, d6],
  "right_gripper": 0.85,
  "stop": false
}
```

- `right_delta_joints` is `result["action.right_delta_joints"][0][0:7]`
  — additive radians. Magnitudes ~0.007–0.023 per the guide; if we see
  >0.1 something is wrong (preprocessing or state vector).
- `right_gripper` is `result["action.right_delta_joints"][0][7]`,
  absolute `[0,1]`.
- `stop:true` once on graceful shutdown so the Pi parks.

Port 5006 to avoid colliding with the NoMaD :5005 if both stacks ever
run together.

### 5. Pi-side executor (out of scope for this repo)

The Pi needs a small client that:
- connects to `<orin-ip>:5006`,
- reads newline JSON,
- on each frame: `target_joints = current_right_joints + delta_joints`,
  command the right arm at that target; write `right_gripper`
  directly,
- if no frame for >0.5s: hold pose,
- on `stop:true`: park and ignore further frames.

Coordinate the schema above with whoever owns the Pi-side stack.

## Order of implementation

1. **Get the model on the orin.** rsync the checkpoint and merge the
   YOR fork commits (training config + policy transforms) into this
   tree. Verify with `uv run scripts/serve_policy.py
   policy:checkpoint --policy.config=pi0_yor_right_arm
   --policy.dir=...` — server should start, log a metadata header,
   wait for a client. Connect with the openpi-client repl and send a
   dummy obs; confirm the response keys/shapes match the guide
   (`action.right_delta_joints`, shape `(H, 8)`).
2. **Stub the orin script with a fake policy.** Capture cameras and
   state, build the obs dict, but emit zero-deltas + last-known
   gripper at 10 Hz over TCP. Verify on a laptop with `nc <orin-ip>
   5006` that one clean newline-JSON arrives per tick at the expected
   rate.
3. **Wire in the real policy (Option A).** Same script, now calling
   the websocket server. Log returned action shapes + delta
   magnitudes. **Don't dispatch to the Pi yet** — sanity-check the
   numbers. Inspect at least 50 inferences across varied obs to
   confirm the gripper output is in `[0,1]` and joint deltas are in
   the expected band.
4. **Closed loop, low stakes.** Pi executor on, **clamp delta scale
   to 0.3×**, hover in free space without the cube. Watch for drift
   and direction sanity.
5. **Real task.** Full delta scale, cube on the table. Iterate on
   camera framing / lighting before iterating on the model.

## Open questions

- **State stream when idle.** The teleop publisher only emits inside
  an episode. We need either (a) a `state/live` topic on the Pi that
  publishes continuously, (b) the Pi to expose a request/reply
  current-state endpoint we poll at 10 Hz, or (c) a hand-coded
  warmup pose for the first frame and a reliance on the very first
  delta dispatch to wake the publisher. (a) is cleanest. **Confirm
  with whoever owns the Pi.**
- **Channel order + normalisation.** Must match the YOR policy
  module's transforms exactly. Read that module the moment we have
  it; do not guess. Option A in §4 sidesteps the normalisation
  question entirely.
- **Crop spec for ZED.** Guide says "Center-cropped 1280×720 →
  720×720". If the training transform crops differently (e.g. crops
  off the top, or uses a non-square crop), the model will see a
  shifted scene. Verify against the YOR policy module.
- **Fisheye device index stability.** V4L2 numbering is not stable
  across reboots/replugs (the YOR `CLAUDE.md` warns about this);
  `collect.py` and `camera_viewer.py` already disagree. At startup,
  run `v4l2-ctl --list-devices` and pick the entry whose USB path
  matches the right hand.
- **Inference latency.** Pi0 LoRA on Orin AGX is ~few hundred ms per
  call; on Orin Nano probably worse. If we can't hold 10 Hz, drop
  to 5 Hz **or** apply two action-chunk steps per inference (still
  receding-horizon, just coarser). Measure in step 3 before
  committing.
- **Kill switch.** Ctrl-C in the orin terminal must send `stop:true`
  before exiting (atexit handler around the TCP server).

## File layout

```
notes/
  yor_deployment_guide.md            (existing)
  inference_pipeline_plan.md         (this file)
scripts/
  run_yor_inference.py               (NEW — §1–§4)
checkpoints/
  pi0_yor_right_arm/
    yor_right_arm_run_0/
      9999/                          (NEW — copy from training host)
src/openpi/
  policies/yor_policy.py             (NEW — copy from training fork)
  training/config.py                 (PATCH — register pi0_yor_right_arm)
```

`run_yor_inference.py` should depend only on `openpi-client`, stdlib,
cv2, numpy, pyzed, and commlink. The heavy training stack
(jax/flax/torch) only needs to load inside the `serve_policy.py`
process.
