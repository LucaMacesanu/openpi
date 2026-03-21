"""Evaluate a trained policy on a single named LIBERO task.

Run after activating the libero venv and setting PYTHONPATH:
    source examples/libero/.venv/bin/activate
    export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
    python scripts/libero_eval_task.py --task "pick up the black bowl ..."

The policy server must already be running:
    uv run scripts/serve_policy.py --env LIBERO --checkpoint <path>
"""

import collections
import dataclasses
import datetime
import json
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

ALL_SUITE_NAMES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
]

# Maximum rollout steps per suite (from main.py).
MAX_STEPS_PER_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    # Task to evaluate — matched against task.language in all suites.
    # Use --list_tasks to see available descriptions.
    task: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # Eval
    num_trials: int = 50
    num_steps_wait: int = 10
    seed: int = 7

    # Output
    video_out_path: str = "data/libero/videos"
    # If set, write a JSON results file to this path after eval.
    results_file: str = ""

    # Utility: print all available task descriptions and exit.
    list_tasks: bool = False
    # Optionally restrict --list_tasks to one suite.
    suite: str = ""


# ---------------------------------------------------------------------------
# Helpers (mirrors examples/libero/main.py)
# ---------------------------------------------------------------------------

def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _load_all_tasks():
    """Return list of (suite_name, task_obj, task_id, description)."""
    benchmark_dict = benchmark.get_benchmark_dict()
    results = []
    for suite_name in ALL_SUITE_NAMES:
        suite = benchmark_dict[suite_name]()
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            results.append((suite_name, task, task_id, task.language))
    return results


def _find_task(task_description: str):
    """Search all suites for a task whose language matches task_description.

    Matching is case-insensitive substring; raises if ambiguous or not found.
    Returns (suite_name, task_suite, task_id, task_obj, task_description).
    """
    query = task_description.strip().lower()
    benchmark_dict = benchmark.get_benchmark_dict()
    matches = []
    for suite_name in ALL_SUITE_NAMES:
        suite = benchmark_dict[suite_name]()
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            if query == task.language.lower():
                # Exact match — return immediately.
                return suite_name, suite, task_id, task, task.language
            if query in task.language.lower():
                matches.append((suite_name, suite, task_id, task, task.language))

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise ValueError(
            f"No task found matching '{task_description}'.\n"
            "Run with --list_tasks to see all available task descriptions."
        )
    descriptions = "\n".join(f"  [{s}] {d}" for s, _, _, _, d in matches)
    raise ValueError(
        f"Ambiguous query '{task_description}' matched {len(matches)} tasks:\n"
        f"{descriptions}\n"
        "Please use a more specific description."
    )


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def eval_task(args: Args) -> None:
    np.random.seed(args.seed)

    if args.list_tasks:
        suite_filter = args.suite.strip().lower()
        all_tasks = _load_all_tasks()
        for suite_name, _, task_id, desc in all_tasks:
            if suite_filter and suite_filter not in suite_name.lower():
                continue
            print(f"[{suite_name}] task_id={task_id}: {desc}")
        return

    if not args.task:
        raise ValueError("--task is required (or use --list_tasks to browse tasks).")

    suite_name, task_suite, task_id, task, task_description = _find_task(args.task)
    max_steps = MAX_STEPS_PER_SUITE[suite_name]

    logging.info(f"Suite: {suite_name}  |  task_id: {task_id}")
    logging.info(f"Task:  {task_description}")
    logging.info(f"Trials: {args.num_trials}  |  max_steps: {max_steps}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    initial_states = task_suite.get_task_init_states(task_id)
    env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    num_trials = min(args.num_trials, len(initial_states))
    if num_trials < args.num_trials:
        logging.warning(
            f"Only {len(initial_states)} initial states available; "
            f"running {num_trials} trials."
        )

    successes = 0
    trial_outcomes = []
    for episode_idx in tqdm.tqdm(range(num_trials), desc="Trials"):
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])

        action_plan: collections.deque = collections.deque()
        replay_images = []
        done = False
        t = 0

        while t < max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        wrist_img, args.resize_size, args.resize_size
                    )
                )
                replay_images.append(img)

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": task_description,
                    }
                    action_chunk = client.infer(element)["actions"]
                    assert len(action_chunk) >= args.replan_steps, (
                        f"Policy predicts {len(action_chunk)} steps but "
                        f"replan_steps={args.replan_steps}."
                    )
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    successes += 1
                    break
                t += 1

            except Exception as e:
                logging.error(f"Episode {episode_idx} error: {e}")
                break

        trial_outcomes.append(bool(done))
        suffix = "success" if done else "failure"
        task_segment = task_description.replace(" ", "_")[:60]
        video_path = (
            pathlib.Path(args.video_out_path)
            / f"trial_{episode_idx:03d}_{task_segment}_{suffix}.mp4"
        )
        if replay_images:
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

        logging.info(
            f"Trial {episode_idx + 1}/{num_trials} — "
            f"{'SUCCESS' if done else 'failure'} — "
            f"running SR: {successes}/{episode_idx + 1} "
            f"({100 * successes / (episode_idx + 1):.1f}%)"
        )

    print(
        f"\n=== Results: {task_description} ===\n"
        f"Success rate: {successes}/{num_trials} "
        f"({100 * successes / num_trials:.1f}%)"
    )

    if args.results_file:
        results_path = pathlib.Path(args.results_file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "task": task_description,
            "suite": suite_name,
            "num_trials": num_trials,
            "successes": successes,
            "success_rate": successes / num_trials,
            "trials": trial_outcomes,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        results_path.write_text(json.dumps(results, indent=2))
        logging.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_task)
