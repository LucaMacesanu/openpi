"""Policy transforms for the YOR bimanual robot, right-arm-only config.

Observation space:
  - ZED stereo camera (base_0_rgb slot)
  - fish1 fisheye, right-side view (left_wrist_0_rgb slot)
  - right arm joints rj0-rj6 + right_gripper state (8 dims)

Action space:
  - action.right_delta_joints: [Δrj0..6, right_gripper_absolute] (8 dims)
    Matches pi0/pi0.5 pretraining distribution (delta joints + absolute gripper).
"""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Indices into the 17-dim observation.state vector:
#   lj0-lj6 = 0-6, rj0-rj6 = 7-13, left_gripper = 14, right_gripper = 15, lift = 16
RIGHT_ARM_STATE_INDICES = [7, 8, 9, 10, 11, 12, 13, 15]  # rj0-rj6 + right_gripper (8 dims)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _center_crop_square(image: np.ndarray) -> np.ndarray:
    """Center-crop an HWC image to a square using the shorter dimension."""
    h, w = image.shape[:2]
    size = min(h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return image[top : top + size, left : left + size]


@dataclasses.dataclass(frozen=True)
class YorRightArmInputs(transforms.DataTransformFn):
    """Maps dataset/inference keys to model input format for right-arm-only config."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        zed_image = _center_crop_square(_parse_image(data["observation/zed"]))
        fish_image = _parse_image(data["observation/fish_right"])

        state = np.asarray(data["observation/state"])[RIGHT_ARM_STATE_INDICES]  # (8,)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": zed_image,
                "left_wrist_0_rgb": fish_image,
                "right_wrist_0_rgb": np.zeros_like(zed_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])  # (horizon, 8) delta joints
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class YorRightArmOutputs(transforms.DataTransformFn):
    """Maps model output back to robot action keys."""

    def __call__(self, data: dict) -> dict:
        return {"action.right_delta_joints": np.asarray(data["actions"])}
