"""Convert HDF5 LIBERO demos to LeRobot format compatible with OpenPI's lerobot v2.1."""

import h5py
import numpy as np
import shutil
import argparse
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


def convert_dataset(input_path: str, repo_id: str, push_to_hub: bool = False):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find HDF5 file at: {input_path}")

    task_description = input_path.stem.replace("_", " ")

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    with h5py.File(input_path, 'r') as f:
        demos = list(f['data'].keys())
        first_demo = f['data'][demos[0]]

        img_shape = first_demo['obs']['agentview_rgb'].shape[1:]
        wrist_shape = first_demo['obs']['eye_in_hand_rgb'].shape[1:]
        joint_dim = first_demo['obs']['joint_states'].shape[1]
        gripper_dim = first_demo['obs']['gripper_states'].shape[1]
        state_dim = joint_dim + gripper_dim
        action_dim = first_demo['actions'].shape[1]

        print(f"Detected Dimensions - State: {state_dim}, Action: {action_dim}")
        print(f"Task Description: '{task_description}'")

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="panda",
            fps=10,
            features={
                "image": {
                    "dtype": "image",
                    "shape": img_shape,
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": wrist_shape,
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (state_dim,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (action_dim,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

        print(f"Converting {len(demos)} episodes from {input_path}...")

        for demo_key in demos:
            demo_group = f['data'][demo_key]
            num_steps = demo_group['actions'].shape[0]

            for i in range(num_steps):
                state_vec = np.concatenate([
                    demo_group['obs']['joint_states'][i],
                    demo_group['obs']['gripper_states'][i]
                ]).astype(np.float32)

                frame_data = {
                    "image": demo_group['obs']['agentview_rgb'][i],
                    "wrist_image": demo_group['obs']['eye_in_hand_rgb'][i],
                    "state": state_vec,
                    "actions": demo_group['actions'][i].astype(np.float32),
                    "task": task_description,
                }

                dataset.add_frame(frame_data)

            dataset.save_episode()

    print(f"Success! Dataset saved to {output_path}")

    if push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub(tags=["libero", "openpi"], private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the .hdf5 file")
    parser.add_argument("--repo_id", type=str, required=True, help="Name for the LeRobot dataset")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload to HF Hub")

    args = parser.parse_args()
    convert_dataset(args.input, args.repo_id, args.push_to_hub)
