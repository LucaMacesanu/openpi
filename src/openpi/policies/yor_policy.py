import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.policies import yor_rotation


def _parse_image(image) -> np.ndarray:
    """LeRobotDataset yields images as (C, H, W) float32 in [0, 1]; the model wants
    (H, W, C) uint8."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YorInputs(transforms.DataTransformFn):
    """icl-dataset's 3 real cameras map 1:1 onto pi05's 3 fixed image slots -- no
    zero-padded slot needed (unlike droid_policy.DroidInputs' single-wrist setup).
    State (15: left/right arm joints + lift) and action (20: left/right ee pose +
    gripper + base vel + lift, see yor-icl-pi05-easy-pnp-v2.yaml) are passed through
    mostly raw -- action dims 16:20 (base vel + lift_cmd) are zeroed by default, see
    __call__ -- and ModelTransformFactory's PadStatesAndActions zero-pads both to the
    model's action_dim (32) downstream.

    drop_base_lift=True (pi05_extended_quantilesfixed) slices dims 16:20 off entirely
    instead of zeroing them. Zeroing alone doesn't fix quantile normalization: dims
    16:20's q01/q99 (computed from the raw, near-but-not-exactly-zero base/lift jitter)
    are themselves near-identical near-zero constants, so Normalize's
    (x-q01)/(q99-q01+1e-6) still blows up a deliberately-zeroed x=0 into a huge
    constant (e.g. ~63 for base_vel.vx) -- accounts for ~125 of the ~161 step-0 flow-
    matching loss measured on yor_icl_pi05_expanded (see notes/training_runs.md).
    Dropping the dims outright means Normalize never touches them (it slices
    norm_stats to `x.shape[-1]`), so no new norm_stats asset is needed -- the existing
    file's first 16 entries (the real dims) are used as-is."""

    drop_base_lift: bool = False

    def __call__(self, data: dict) -> dict:
        images = {
            "base_0_rgb": _parse_image(data["observation.images.zed"]),
            "left_wrist_0_rgb": _parse_image(data["observation.images.fish0"]),
            "right_wrist_0_rgb": _parse_image(data["observation.images.fish1"]),
        }
        image_masks = {name: np.True_ for name in images}

        inputs = {
            "state": np.asarray(data["observation.state"]),
            "image": images,
            "image_mask": image_masks,
        }

        if "action" in data:
            # Dims 16:20 (base_vel.vx/vy/omega, lift_cmd) are always ~0 -- this is
            # stationary tabletop bimanual data, base/lift are never actually
            # commanded. But not EXACTLY 0 (teleop/sensor jitter), so norm_stats'
            # std collapses to ~0 there (std=0.0 for omega/lift_cmd, ~8e-4 for
            # vx/vy) and Normalize's (x-mean)/(std+1e-6) amplifies that jitter by
            # ~1e3-1e6x on whatever frames have nonzero noise -- catastrophic
            # per-item loss/grad-norm spikes (up to ~6e5 loss) in every arm that
            # feeds this "actions" field to a continuous flow-matching loss
            # (yor_icl_pi05_expanded, yor_icl_ki_expanded_subtask's primary flow
            # branch, yor_icl_victr_vision_expanded, yor_icl_victr_value_expanded
            # all hit this on the expanded 1,784-episode set; the discrete
            # FAST-token consumers are unaffected since tokenizer.py/yor_ki.py
            # already clip before quantizing). Zero these dims here so they
            # normalize to a fixed constant instead of amplified per-frame noise
            # -- harmless: they carry no task signal, and the model already pads
            # unused action dims to 32 the same way (PadStatesAndActions).
            #
            # drop_base_lift=True slices them off instead (see class docstring) --
            # closes the residual quantile-norm blowup that zeroing alone doesn't fix.
            action = np.array(data["action"], dtype=np.float32, copy=True)
            if self.drop_base_lift:
                action = action[..., :16]
            else:
                action[..., 16:20] = 0.0
            inputs["actions"] = action

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        # Pass through VICTR's retrieval-context fields unchanged, if an earlier
        # transform (openpi.policies.yor_retrieval.RetrievalContextInputs) added them --
        # this dict is otherwise built from scratch, so anything not explicitly
        # forwarded here would silently vanish for the plain pi05 config too.
        for key in (
            "context_images",
            "context_image_masks",
            "context_tokens",
            "context_tokens_mask",
            "fast_action_tokens",
            "fast_action_tokens_mask",
            "subtask_tokens",
            "subtask_tokens_mask",
            "nearest_action_tokens",
            "nearest_action_tokens_mask",
            "exp_lamda_distance",
        ):
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class YorOutputs(transforms.DataTransformFn):
    drop_base_lift: bool = False

    def __call__(self, data: dict) -> dict:
        # icl-dataset's raw action is 20-dim (left_ee(7) | right_ee(7) | gripper(2) |
        # base_vel(3) | lift(1)); the model pads to 32, so slice back down. When
        # drop_base_lift=True the model itself only ever saw/predicted 16 dims (see
        # YorInputs) -- slice to that instead. Any caller needing a real 20-dim robot
        # command back (e.g. serving) must re-pad dims 16:20 with zeros itself.
        n = 16 if self.drop_base_lift else 20
        return {"actions": np.asarray(data["actions"][..., :n])}


# Raw icl-dataset action column layout (20-dim, see info.json's `action` feature
# names): [0:4]=left quat(wxyz) [4:7]=left xyz [7:11]=right quat(wxyz) [11:14]=right
# xyz [14]=left gripper [15]=right gripper [16:20]=base_vel(3)+lift_cmd (dropped, see
# YorInputs.drop_base_lift).
_LEFT_QUAT, _LEFT_POS = slice(0, 4), slice(4, 7)
_RIGHT_QUAT, _RIGHT_POS = slice(7, 11), slice(11, 14)
_LEFT_GRIP, _RIGHT_GRIP = slice(14, 15), slice(15, 16)


@dataclasses.dataclass(frozen=True)
class YorInputsAbsoluteJoint(transforms.DataTransformFn):
    """Absolute-joint-space variant: both state AND action are icl-dataset's 15-dim
    observation.state (left/right arm joints (7+7) + lift) -- unlike YorInputs' 20-dim
    EE-pose action, this trains the model to predict future absolute joint positions
    directly. icl-dataset never recorded a joint-space *action* (teleop commands EE
    poses; raw_action.* has no joint equivalent) -- the only joint-space signal on
    disk is observation.state, measured per frame. So the "action" here is literally
    the future window of observation.state itself: LeRobotYorAbsoluteJointDataConfig
    sets action_sequence_keys=("observation.state",), which makes data_loader.py's
    delta_timestamps fetch an (action_horizon, 15) window of observation.state instead
    of the usual single current frame.

    Because action_sequence_keys and the current-state input share the SAME dataset
    column name here (there is no separate column to key them apart), the fetched
    value's rank tells them apart: at inference (real robot serving, no
    action_sequence_keys windowing applied) data["observation.state"] is a plain
    (15,) vector; during training it comes back windowed as (action_horizon, 15). The
    window's own first row is exactly "now" (delta_timestamps starts at t=0), so it
    doubles as the current-state input -- no extra lookback/lookahead frame needed
    beyond what data_loader.py already fetches.

    No zeroing/dropping needed (unlike YorInputs' base_vel/lift_cmd fix): state has no
    near-constant-zero dims to begin with, every one of the 15 is a real joint/lift
    reading."""

    def __call__(self, data: dict) -> dict:
        images = {
            "base_0_rgb": _parse_image(data["observation.images.zed"]),
            "left_wrist_0_rgb": _parse_image(data["observation.images.fish0"]),
            "right_wrist_0_rgb": _parse_image(data["observation.images.fish1"]),
        }
        image_masks = {name: np.True_ for name in images}

        state_field = np.asarray(data["observation.state"])
        inputs = {
            "state": state_field[0] if state_field.ndim > 1 else state_field,
            "image": images,
            "image_mask": image_masks,
        }
        if state_field.ndim > 1:
            inputs["actions"] = state_field.astype(np.float32)

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        for key in (
            "context_images",
            "context_image_masks",
            "context_tokens",
            "context_tokens_mask",
            "fast_action_tokens",
            "fast_action_tokens_mask",
            "subtask_tokens",
            "subtask_tokens_mask",
            "nearest_action_tokens",
            "nearest_action_tokens_mask",
            "exp_lamda_distance",
        ):
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class YorOutputsAbsoluteJoint(transforms.DataTransformFn):
    """Inverse of YorInputsAbsoluteJoint: unpads the model's 32-dim output back to the
    15-dim [left arm(7), right arm(7), lift(1)] absolute joint-position layout."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :15])}


@dataclasses.dataclass(frozen=True)
class YorInputsDeltaRot6D(transforms.DataTransformFn):
    """pi05_extended_deltarot6d's action representation: per arm, delta cartesian
    position (relative to the first frame of the fetched action_horizon window, i.e.
    "now") + absolute orientation as the first two rows of the rotation matrix
    (Zhou et al. 2019 6D representation, see yor_rotation.py) instead of xyz+quaternion.
    Per arm: delta_xyz(3) + rot6d(6) + gripper(1) = 10; both arms = 20 (base_vel/lift
    are dropped entirely, same as YorInputs(drop_base_lift=True) -- this dataset never
    actually commands them).

    Delta is anchored to the window's own first frame (not a frame-to-frame delta
    against the previous timestep) -- needs no extra lookback frame beyond the
    (action_horizon, 20) window data_loader.py already fetches via
    action_sequence_keys, and delta[0] is identically (0,0,0) rather than depending on
    a frame outside the window.

    Same camera/state/prompt/retrieval-context handling as YorInputs -- only the
    action encoding differs, so this mirrors that class rather than subclassing it
    (the action logic has nothing in common)."""

    def __call__(self, data: dict) -> dict:
        images = {
            "base_0_rgb": _parse_image(data["observation.images.zed"]),
            "left_wrist_0_rgb": _parse_image(data["observation.images.fish0"]),
            "right_wrist_0_rgb": _parse_image(data["observation.images.fish1"]),
        }
        image_masks = {name: np.True_ for name in images}

        inputs = {
            "state": np.asarray(data["observation.state"]),
            "image": images,
            "image_mask": image_masks,
        }

        if "action" in data:
            action = np.asarray(data["action"], dtype=np.float64)  # (horizon, 20)

            left_delta_pos = action[..., _LEFT_POS] - action[..., 0:1, _LEFT_POS]
            right_delta_pos = action[..., _RIGHT_POS] - action[..., 0:1, _RIGHT_POS]
            left_rot6d = yor_rotation.quat_wxyz_to_rot6d(action[..., _LEFT_QUAT])
            right_rot6d = yor_rotation.quat_wxyz_to_rot6d(action[..., _RIGHT_QUAT])

            inputs["actions"] = np.concatenate(
                [
                    left_delta_pos,
                    left_rot6d,
                    action[..., _LEFT_GRIP],
                    right_delta_pos,
                    right_rot6d,
                    action[..., _RIGHT_GRIP],
                ],
                axis=-1,
            ).astype(np.float32)

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        for key in (
            "context_images",
            "context_image_masks",
            "context_tokens",
            "context_tokens_mask",
            "fast_action_tokens",
            "fast_action_tokens_mask",
            "subtask_tokens",
            "subtask_tokens_mask",
            "nearest_action_tokens",
            "nearest_action_tokens_mask",
            "exp_lamda_distance",
        ):
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class YorOutputsDeltaRot6D(transforms.DataTransformFn):
    """Inverse of YorInputsDeltaRot6D's action encoding. Training itself never calls
    this (the flow-matching loss operates in normalized space inside the model) --
    it's for eventual eval/serving, unmapping the model's 32-dim padded output back to
    the 20-dim [delta_xyz(3)+rot6d(6)+gripper(1)]x2 layout, with rot6d converted back
    to a quaternion.

    NOT a full inverse to a robot-executable command: the delta positions are still
    relative to whatever frame anchored the query window (see class docstring on
    YorInputsDeltaRot6D) -- unlike YorOutputs' absolute xyz, a caller must supply its
    own current end-effector position (not tracked anywhere in this policy's
    observation, which only carries joint-space state) to add the delta onto before
    sending to the robot."""

    def __call__(self, data: dict) -> dict:
        action = np.asarray(data["actions"])[..., :20]
        left_delta_pos, left_rot6d, left_grip = action[..., 0:3], action[..., 3:9], action[..., 9:10]
        right_delta_pos, right_rot6d, right_grip = action[..., 10:13], action[..., 13:19], action[..., 19:20]
        left_quat = yor_rotation.rot6d_to_quat_wxyz(left_rot6d)
        right_quat = yor_rotation.rot6d_to_quat_wxyz(right_rot6d)
        return {
            "actions": np.concatenate(
                [left_quat, left_delta_pos, right_quat, right_delta_pos, left_grip, right_grip], axis=-1
            )
        }


@dataclasses.dataclass(frozen=True)
class YorInputsAlignedQ(transforms.DataTransformFn):
    """Absolute-joint-space variant with a REAL recorded action, unlike
    YorInputsAbsoluteJoint's reused-future-state hack: icl-dataset-fixed-action's
    action.q_target (14-dim: left/right arm joints, IK-reconstructed from
    action.left_ee/action.right_ee and anchored per-episode to
    observation.state's own FK, see that dataset's README) is a genuine action
    column, aligned with observation.state's joint space. State is
    icl-dataset's 15-dim observation.state with the lift dim (index 14)
    dropped -- 14-dim [lj0..lj6, rj0..rj6] -- since neither q_target nor
    action.q_target's paired lift/base signal exists to keep proprioception
    and action genuinely in the same space (icl-dataset's action column has no
    base component at all; lift_cmd is dropped from the action side too, see
    below). Action is action.q_target (14) with gripper appended (2, from the
    raw 20-dim "action" column's _LEFT_GRIP/_RIGHT_GRIP dims 14:16 -- q_target
    itself has no gripper channel), giving a 16-dim
    [lj0..lj6, rj0..rj6, left_grip, right_grip] action -- no lift_cmd/base_vel
    (raw action dims 16:20), fetched via LeRobotYorAlignedQDataConfig's
    action_sequence_keys=("action.q_target", "action"). Unlike
    YorInputsAbsoluteJoint, no rank trick is needed to tell current state
    apart from the action window -- they're different dataset columns."""

    def __call__(self, data: dict) -> dict:
        images = {
            "base_0_rgb": _parse_image(data["observation.images.zed"]),
            "left_wrist_0_rgb": _parse_image(data["observation.images.fish0"]),
            "right_wrist_0_rgb": _parse_image(data["observation.images.fish1"]),
        }
        image_masks = {name: np.True_ for name in images}

        # Drop lift (index 14) -- state has no base component to begin with, and
        # action.q_target's action space has neither lift nor base, so keep state
        # aligned to arms-only (see class docstring).
        state = np.asarray(data["observation.state"])[..., :14]
        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if "action.q_target" in data:
            q_target = np.asarray(data["action.q_target"], dtype=np.float32)
            action = np.asarray(data["action"], dtype=np.float32)
            inputs["actions"] = np.concatenate(
                [q_target, action[..., _LEFT_GRIP], action[..., _RIGHT_GRIP]], axis=-1
            )

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        for key in (
            "context_images",
            "context_image_masks",
            "context_tokens",
            "context_tokens_mask",
            "fast_action_tokens",
            "fast_action_tokens_mask",
            "subtask_tokens",
            "subtask_tokens_mask",
            "nearest_action_tokens",
            "nearest_action_tokens_mask",
            "exp_lamda_distance",
        ):
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class YorOutputsAlignedQ(transforms.DataTransformFn):
    """Inverse of YorInputsAlignedQ: unpads the model's 32-dim output back to
    the 16-dim [left arm(7), right arm(7), left_grip(1), right_grip(1)] layout."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :16])}


@dataclasses.dataclass(frozen=True)
class YorInputsAlignedDeltaRot6D(transforms.DataTransformFn):
    """YorInputsDeltaRot6D, but state is icl-dataset-fixed-obs's
    observation.left_ee/observation.right_ee (absolute EE pose, quat+xyz per
    arm, FK'd from the real joint encoders, quaternion-hemisphere-corrected)
    instead of observation.state (joint space) -- aligns proprioception with
    the EE-space delta-position + absolute-rot6d-orientation action
    YorInputsDeltaRot6D already computes (see that class's docstring for the
    action math, unchanged here), instead of mixing joint-space state with
    EE-space action. State is 14-dim: [left qw,qx,qy,qz,x,y,z, right
    qw,qx,qy,qz,x,y,z]."""

    def __call__(self, data: dict) -> dict:
        images = {
            "base_0_rgb": _parse_image(data["observation.images.zed"]),
            "left_wrist_0_rgb": _parse_image(data["observation.images.fish0"]),
            "right_wrist_0_rgb": _parse_image(data["observation.images.fish1"]),
        }
        image_masks = {name: np.True_ for name in images}

        state = np.concatenate(
            [np.asarray(data["observation.left_ee"]), np.asarray(data["observation.right_ee"])],
            axis=-1,
        )
        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if "action" in data:
            action = np.asarray(data["action"], dtype=np.float64)  # (horizon, 20)

            left_delta_pos = action[..., _LEFT_POS] - action[..., 0:1, _LEFT_POS]
            right_delta_pos = action[..., _RIGHT_POS] - action[..., 0:1, _RIGHT_POS]
            left_rot6d = yor_rotation.quat_wxyz_to_rot6d(action[..., _LEFT_QUAT])
            right_rot6d = yor_rotation.quat_wxyz_to_rot6d(action[..., _RIGHT_QUAT])

            inputs["actions"] = np.concatenate(
                [
                    left_delta_pos,
                    left_rot6d,
                    action[..., _LEFT_GRIP],
                    right_delta_pos,
                    right_rot6d,
                    action[..., _RIGHT_GRIP],
                ],
                axis=-1,
            ).astype(np.float32)

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        for key in (
            "context_images",
            "context_image_masks",
            "context_tokens",
            "context_tokens_mask",
            "fast_action_tokens",
            "fast_action_tokens_mask",
            "subtask_tokens",
            "subtask_tokens_mask",
            "nearest_action_tokens",
            "nearest_action_tokens_mask",
            "exp_lamda_distance",
        ):
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class YorOutputsAlignedDeltaRot6D(transforms.DataTransformFn):
    """Identical to YorOutputsDeltaRot6D -- action encoding is unchanged, only
    YorInputsAlignedDeltaRot6D's state source differs."""

    def __call__(self, data: dict) -> dict:
        action = np.asarray(data["actions"])[..., :20]
        left_delta_pos, left_rot6d, left_grip = action[..., 0:3], action[..., 3:9], action[..., 9:10]
        right_delta_pos, right_rot6d, right_grip = action[..., 10:13], action[..., 13:19], action[..., 19:20]
        left_quat = yor_rotation.rot6d_to_quat_wxyz(left_rot6d)
        right_quat = yor_rotation.rot6d_to_quat_wxyz(right_rot6d)
        return {
            "actions": np.concatenate(
                [left_quat, left_delta_pos, right_quat, right_delta_pos, left_grip, right_grip], axis=-1
            )
        }


# icl-dataset-fixed-obs's observation.left_ee/observation.right_ee raw column layout
# (7-dim each, see that dataset's info.json): [0:4]=quat(wxyz) [4:7]=xyz.
_EE_QUAT, _EE_POS = slice(0, 4), slice(4, 7)


@dataclasses.dataclass(frozen=True)
class YorInputsCanonical(transforms.DataTransformFn):
    """ICRA canonical action/observation space (notes/ICRA_plan.md): proprioception is
    ONLY absolute EE position + absolute rot6d orientation, per arm -- no gripper, no
    lift, no base. Action is unchanged from YorInputsAlignedDeltaRot6D/
    YorInputsDeltaRot6D: per arm, delta EE position (relative to the query window's
    own first frame) + absolute rot6d orientation + gripper -- no lift, no base.

    State: from icl-dataset-fixed-obs's observation.left_ee/observation.right_ee
    (quat+xyz, FK'd from the real joint encoders), converted to
    concatenate([pos_L(3), rot6d_L(6), pos_R(3), rot6d_R(6)]) = 18-dim. Unlike
    YorInputsAlignedDeltaRot6D (which passes the raw quat+xyz straight through, 14-dim),
    orientation here is rot6d to match the action side's orientation representation --
    the only genuinely new logic in this class; the action math itself is copied
    verbatim from YorInputsDeltaRot6D/YorInputsAlignedDeltaRot6D (see those classes'
    docstrings for the delta-position/rot6d/gripper derivation)."""

    def __call__(self, data: dict) -> dict:
        images = {
            "base_0_rgb": _parse_image(data["observation.images.zed"]),
            "left_wrist_0_rgb": _parse_image(data["observation.images.fish0"]),
            "right_wrist_0_rgb": _parse_image(data["observation.images.fish1"]),
        }
        image_masks = {name: np.True_ for name in images}

        left_ee = np.asarray(data["observation.left_ee"])
        right_ee = np.asarray(data["observation.right_ee"])
        state = np.concatenate(
            [
                left_ee[..., _EE_POS],
                yor_rotation.quat_wxyz_to_rot6d(left_ee[..., _EE_QUAT]),
                right_ee[..., _EE_POS],
                yor_rotation.quat_wxyz_to_rot6d(right_ee[..., _EE_QUAT]),
            ],
            axis=-1,
        )
        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if "action" in data:
            action = np.asarray(data["action"], dtype=np.float64)  # (horizon, 20)

            left_delta_pos = action[..., _LEFT_POS] - action[..., 0:1, _LEFT_POS]
            right_delta_pos = action[..., _RIGHT_POS] - action[..., 0:1, _RIGHT_POS]
            left_rot6d = yor_rotation.quat_wxyz_to_rot6d(action[..., _LEFT_QUAT])
            right_rot6d = yor_rotation.quat_wxyz_to_rot6d(action[..., _RIGHT_QUAT])

            inputs["actions"] = np.concatenate(
                [
                    left_delta_pos,
                    left_rot6d,
                    action[..., _LEFT_GRIP],
                    right_delta_pos,
                    right_rot6d,
                    action[..., _RIGHT_GRIP],
                ],
                axis=-1,
            ).astype(np.float32)

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        for key in (
            "context_images",
            "context_image_masks",
            "context_tokens",
            "context_tokens_mask",
            "fast_action_tokens",
            "fast_action_tokens_mask",
            "subtask_tokens",
            "subtask_tokens_mask",
            "nearest_action_tokens",
            "nearest_action_tokens_mask",
            "exp_lamda_distance",
        ):
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class YorOutputsCanonical(transforms.DataTransformFn):
    """Identical to YorOutputsDeltaRot6D/YorOutputsAlignedDeltaRot6D -- action encoding
    is unchanged, only YorInputsCanonical's state source/representation differs (state
    is never unpacked here; nothing about it is model output)."""

    def __call__(self, data: dict) -> dict:
        action = np.asarray(data["actions"])[..., :20]
        left_delta_pos, left_rot6d, left_grip = action[..., 0:3], action[..., 3:9], action[..., 9:10]
        right_delta_pos, right_rot6d, right_grip = action[..., 10:13], action[..., 13:19], action[..., 19:20]
        left_quat = yor_rotation.rot6d_to_quat_wxyz(left_rot6d)
        right_quat = yor_rotation.rot6d_to_quat_wxyz(right_rot6d)
        return {
            "actions": np.concatenate(
                [left_quat, left_delta_pos, right_quat, right_delta_pos, left_grip, right_grip], axis=-1
            )
        }
