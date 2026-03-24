"""Sequential fine-tuning (SFT) script for continual learning research.

Trains a pi0.5 policy on a sequence of Libero tasks one at a time, using the
pi05_libero_low_mem_finetune config (LoRA on both paligemma and action expert).

After completing each task, saves a checkpoint under its own subdirectory and
writes a `tasks_trained.json` file recording the ordered list of all tasks the
model has been trained on up to that point.

Usage:
    CUDA_VISIBLE_DEVICES=4,7 uv run scripts/sft_train.py \
        --tasks "pick up the black bowl between the plate and the ramekin and place it on the plate" \
                "open the top drawer and put the bowl inside" \
        --steps_per_task 5000 \
        --exp_name sft_run_0 \
        --checkpoint_dir /scratch/lim2045/openpi/checkpoints/sft_checkpoints

Checkpoint layout:
    {checkpoint_dir}/{exp_name}/
        task_00_{task_name_slug}/
            {step}/                  <- orbax checkpoint
            tasks_trained.json       <- cumulative task list up to this point
        task_01_{task_name_slug}/
            ...
"""

import dataclasses
import functools
import json
import logging
import platform
from pathlib import Path
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from flax.traverse_util import flatten_dict, unflatten_dict

def _load_weights_and_validate(loader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return unflatten_dict(
        {k: v for k, v in flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info

@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding

def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SFTArgs:
    # Ordered list of task name strings exactly as they appear in the dataset.
    # Mutually exclusive with --num_tasks. Run with --list_tasks to browse.
    tasks: list[str] = dataclasses.field(default_factory=list)

    # If > 0, randomly sample this many tasks from the full dataset instead of
    # specifying them explicitly with --tasks.
    num_tasks: int = 0

    # Seed used when randomly sampling tasks with --num_tasks.
    task_seed: int = 42

    # Number of gradient steps to train on each task.
    steps_per_task: int = 5_000

    # Named config to use as the base training config.
    config_name: str = "pi05_libero_sft"

    # Config name whose precomputed norm stats to use. Defaults to pi05_libero_low_mem_finetune
    # since all SFT configs share the same dataset and embodiment. Only change this if you have
    # computed norm stats for a different config.
    norm_stats_from: str = "pi0_libero_low_mem_finetune"

    # Experiment name — used for W&B run name and the parent checkpoint subdirectory.
    exp_name: str = "sft_run"

    # Root directory where per-task checkpoint subdirectories will be created.
    checkpoint_dir: str = "/scratch/lim2045/openpi/checkpoints/sft_checkpoints"

    # Global batch size (must be divisible by the number of JAX devices).
    batch_size: int = 32

    # DataLoader worker processes.
    num_workers: int = 2

    # Log training metrics every N steps.
    log_interval: int = 100

    # Enable Weights & Biases logging.
    wandb_enabled: bool = True

    # Random seed.
    seed: int = 42

    # If True, print all available task names and exit without training.
    list_tasks: bool = False


# ---------------------------------------------------------------------------
# Task / episode helpers
# ---------------------------------------------------------------------------

def list_available_tasks(repo_id: str) -> list[str]:
    """Return all task names present in the dataset."""
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    return list(meta.tasks.values())


# ---------------------------------------------------------------------------
# Per-task data loader
# ---------------------------------------------------------------------------

def create_task_data_loader(
    config: _config.TrainConfig,
    task_name: str,
    data_sharding: jax.sharding.Sharding,
) -> _data_loader.DataLoader:
    """Build a DataLoader containing only frames that belong to *task_name*.

    Loads the FULL LeRobotDataset so that lerobot's episode_data_index covers
    all episodes (required for correct delta-timestamp lookups), then filters
    down to task frames using torch.utils.data.Subset before applying transforms.
    Filtering by passing `episodes=` to LeRobotDataset fails for v2.0 datasets
    because the sparse global episode indices exceed the dense index array size.
    """
    import torch.utils.data as torch_data

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)

    if task_name not in dataset_meta.task_to_task_index:
        available = "\n".join(f"  - {t}" for t in sorted(dataset_meta.tasks.values()))
        raise ValueError(
            f"Task '{task_name}' not found in '{data_config.repo_id}'.\n"
            f"Available tasks:\n{available}"
        )
    task_idx = dataset_meta.task_to_task_index[task_name]

    # Load full dataset — no episode filter so episode_data_index spans all episodes.
    raw_dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    # Inject task prompt before subsetting (transform reads task_index from each frame).
    if data_config.prompt_from_task:
        prompted = _data_loader.TransformedDataset(
            raw_dataset,
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )
    else:
        prompted = raw_dataset

    # Find frame-level indices for this task.
    task_frame_indices = [
        i for i, t in enumerate(raw_dataset.hf_dataset["task_index"])
        if int(t) == task_idx
    ]
    logging.info(
        f"Task '{task_name}': {len(task_frame_indices)} frames "
        f"(task_index={task_idx})"
    )

    # Subset to task frames; Subset.__getitem__ maps local -> global index,
    # so raw_dataset's full episode_data_index handles lookups correctly.
    task_dataset = torch_data.Subset(prompted, task_frame_indices)
    task_dataset = _data_loader.transform_dataset(task_dataset, data_config)

    local_batch_size = config.batch_size // jax.process_count()
    loader = _data_loader.TorchDataLoader(
        task_dataset,
        local_batch_size=local_batch_size,
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    return _data_loader.DataLoaderImpl(data_config, loader)


# ---------------------------------------------------------------------------
# Checkpoint metadata
# ---------------------------------------------------------------------------

def save_metadata(task_ckpt_dir: Path, tasks_trained: list[str], args: "SFTArgs", base_config) -> None:
    """Write metadata.json alongside the orbax checkpoint directory."""
    metadata = {
        "tasks_trained": tasks_trained,
        "num_tasks_trained": len(tasks_trained),
        "config": {
            "name": args.config_name,
            "paligemma_variant": base_config.model.paligemma_variant,
            "action_expert_variant": base_config.model.action_expert_variant,
            "steps_per_task": args.steps_per_task,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    }
    (task_ckpt_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logging.info(f"Saved metadata to {task_ckpt_dir / 'metadata.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: SFTArgs) -> None:
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    base_config = _config.get_config(args.config_name)
    # Override the config name to point assets_dirs at an existing norm stats folder.
    # All SFT configs share the same dataset/embodiment so norms are interchangeable.
    base_config = dataclasses.replace(base_config, name=args.norm_stats_from)
    repo_id = base_config.data.repo_id

    # --list_tasks: just print and exit.
    if args.list_tasks:
        tasks = list_available_tasks(repo_id)
        print(f"\nAvailable tasks in '{repo_id}':")
        for t in sorted(tasks):
            print(f"  - {t}")
        return

    if args.num_tasks > 0 and args.tasks:
        raise ValueError("Specify either --tasks or --num_tasks, not both.")

    if args.num_tasks > 0:
        import random
        all_tasks = list_available_tasks(repo_id)
        rng_sample = random.Random(args.task_seed)
        args = dataclasses.replace(
            args, tasks=rng_sample.sample(all_tasks, args.num_tasks)
        )
        logging.info(
            f"Sampled {args.num_tasks} tasks (task_seed={args.task_seed}):\n"
            + "\n".join(f"  {i}: {t}" for i, t in enumerate(args.tasks))
        )

    if not args.tasks:
        raise ValueError("Specify at least one task via --tasks or use --num_tasks.")

    logging.info(
        f"Sequential fine-tuning on {len(args.tasks)} task(s):\n"
        + "\n".join(f"  {i}: {t}" for i, t in enumerate(args.tasks))
    )

    # Build a config with CLI overrides applied.
    # exp_name is set to a placeholder here; checkpoint dirs are managed manually.
    config = dataclasses.replace(
        base_config,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.steps_per_task,
        log_interval=args.log_interval,
        wandb_enabled=args.wandb_enabled,
        seed=args.seed,
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"batch_size {config.batch_size} must be divisible by "
            f"device count {jax.device_count()}."
        )

    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path("~/.cache/jax").expanduser()),
    )

    # Top-level checkpoint dir: {checkpoint_dir}/{exp_name}/
    run_ckpt_root = Path(args.checkpoint_dir) / args.exp_name
    run_ckpt_root.mkdir(parents=True, exist_ok=True)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    # W&B — single run covering all tasks.
    if args.wandb_enabled:
        wandb.init(
            name=args.exp_name,
            project=config.project_name,
            config={
                "tasks": args.tasks,
                "steps_per_task": args.steps_per_task,
                **dataclasses.asdict(args),
            },
        )
    else:
        wandb.init(mode="disabled")

    # ------------------------------------------------------------------
    # Initialize train state once from the base pi0.5 checkpoint.
    # Model weights carry over across tasks; the step counter and optimizer
    # state continue uninterrupted so the LR schedule runs smoothly.
    # ------------------------------------------------------------------
    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=False
    )
    jax.block_until_ready(train_state)
    logging.info(
        f"Initialized train state:\n"
        f"{training_utils.array_tree_to_info(train_state.params)}"
    )

    # JIT-compile the train step once; reused across all tasks.
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # ------------------------------------------------------------------
    # Sequential task loop
    # ------------------------------------------------------------------
    tasks_trained_so_far: list[str] = []
    global_step_offset = 0  # for W&B x-axis continuity

    for task_idx, task_name in enumerate(args.tasks):
        logging.info(
            f"\n{'=' * 64}\n"
            f"Task {task_idx + 1}/{len(args.tasks)}: {task_name}\n"
            f"{'=' * 64}"
        )

        # Per-task checkpoint directory.
        safe_slug = task_name.replace(" ", "_")[:50]
        task_ckpt_dir = run_ckpt_root / f"task_{task_idx:02d}_{safe_slug}"

        checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
            task_ckpt_dir,
            keep_period=None,
            overwrite=True,
            resume=False,
        )

        # Build data loader for this task (filters full dataset by task_index at frame level).
        data_loader = create_task_data_loader(config, task_name, data_sharding)
        data_iter = iter(data_loader)
        batch = next(data_iter)

        # Train for steps_per_task steps.
        pbar = tqdm.tqdm(
            range(args.steps_per_task),
            total=args.steps_per_task,
            dynamic_ncols=True,
            desc=f"[{task_idx:02d}] {task_name[:45]}",
        )

        infos = []
        task_step = 0
        for task_step in pbar:
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
            infos.append(info)

            if task_step % config.log_interval == 0:
                stacked = common_utils.stack_forest(infos)
                reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced.items())
                pbar.write(f"[task {task_idx:02d} step {task_step:05d}] {info_str}")
                wandb.log(
                    {
                        **{f"task_{task_idx:02d}/{k}": v for k, v in reduced.items()},
                        "task_index": task_idx,
                    },
                    step=global_step_offset + task_step,
                )
                infos = []

            batch = next(data_iter)

        # Save checkpoint at the final step of this task.
        final_step = task_step
        _checkpoints.save_state(checkpoint_manager, train_state, data_loader, final_step)
        checkpoint_manager.wait_until_finished()

        # Record this task and write cumulative metadata.
        tasks_trained_so_far.append(task_name)
        save_metadata(task_ckpt_dir, list(tasks_trained_so_far), args, base_config)

        logging.info(
            f"Checkpoint saved: {task_ckpt_dir}\n"
            f"Tasks trained so far ({len(tasks_trained_so_far)}): "
            + ", ".join(f"'{t}'" for t in tasks_trained_so_far)
        )

        global_step_offset += args.steps_per_task

    logging.info("Sequential fine-tuning complete.")
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(SFTArgs))
