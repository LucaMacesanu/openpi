"""Sequential fine-tuning script with MoE routing metrics.

This keeps the task-by-task SFT behavior from ``scripts/sft_train.py`` while
using the MoE-aware train step from ``scripts/train_moe.py`` so router and
expert-usage metrics are logged during each task.
"""

import dataclasses
import functools
import logging
import platform
from pathlib import Path

import etils.epath as epath
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as weight_loaders

try:
    from scripts import sft_train as base_sft
    from scripts import train_moe
except ModuleNotFoundError:
    import sft_train as base_sft
    import train_moe


@dataclasses.dataclass
class MoESFTArgs(base_sft.SFTArgs):
    # Optional override for model.moe_config.load_balance_loss_weight.
    # Set to 0.0 to disable the MoE / residual-MoE load-balancing loss.
    load_balance_loss_weight: float | None = None

    # Optional checkpoint params path used to initialize the first sequential SFT task.
    # Example: /path/to/openpi_ckpts/config/exp/29999/params
    initial_checkpoint_params: str | None = None


def _format_scalar_metrics(metrics: dict[str, object]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if np.ndim(v) == 0)


def _validate_moe_config(config: _config.TrainConfig) -> None:
    if not hasattr(config.model, "moe_config"):
        raise ValueError(
            "scripts/sft_train_moe.py requires a MoE model config with a "
            "moe_config field. Use a config such as "
            "'pi0_libero_residual_moe_spatial_only'."
        )


def _override_load_balance_loss_weight(config: _config.TrainConfig, weight: float | None) -> _config.TrainConfig:
    if weight is None:
        return config
    if weight < 0:
        raise ValueError(f"load_balance_loss_weight must be non-negative, got {weight}.")

    moe_config = dataclasses.replace(config.model.moe_config, load_balance_loss_weight=weight)
    model = dataclasses.replace(config.model, moe_config=moe_config)
    return dataclasses.replace(config, model=model)


def main(args: MoESFTArgs) -> None:
    base_sft.init_logging()
    logging.info(f"Running on: {platform.node()}")

    base_config = _config.get_config(args.config_name)
    _validate_moe_config(base_config)

    if args.norm_stats_from is not None:
        # Match sft_train.py: use another config name only to locate norm stats.
        base_config = dataclasses.replace(base_config, name=args.norm_stats_from)
    base_config = _override_load_balance_loss_weight(base_config, args.load_balance_loss_weight)
    if args.initial_checkpoint_params is not None:
        base_config = dataclasses.replace(
            base_config,
            weight_loader=weight_loaders.CheckpointWeightLoader(args.initial_checkpoint_params),
        )
        logging.info(f"Initializing SFT from checkpoint params: {args.initial_checkpoint_params}")
    if args.load_balance_loss_weight is not None:
        logging.info(f"Overriding MoE load_balance_loss_weight={args.load_balance_loss_weight}")
    repo_id = base_config.data.repo_id

    if args.list_tasks:
        tasks = base_sft.list_available_tasks(repo_id)
        print(f"\nAvailable tasks in '{repo_id}':")
        for task in sorted(tasks):
            print(f"  - {task}")
        return

    if args.num_tasks > 0 and args.tasks:
        raise ValueError("Specify either --tasks or --num_tasks, not both.")

    if args.num_tasks > 0:
        import random

        all_tasks = base_sft.list_available_tasks(repo_id)
        rng_sample = random.Random(args.task_seed)
        args = dataclasses.replace(args, tasks=rng_sample.sample(all_tasks, args.num_tasks))
        logging.info(
            f"Sampled {args.num_tasks} tasks (task_seed={args.task_seed}):\n"
            + "\n".join(f"  {i}: {task}" for i, task in enumerate(args.tasks))
        )

    if not args.tasks:
        raise ValueError("Specify at least one task via --tasks or use --num_tasks.")

    logging.info(
        f"Sequential MoE fine-tuning on {len(args.tasks)} task(s):\n"
        + "\n".join(f"  {i}: {task}" for i, task in enumerate(args.tasks))
    )

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

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    run_ckpt_root = Path(args.checkpoint_dir) / args.exp_name
    run_ckpt_root.mkdir(parents=True, exist_ok=True)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

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

    start_task_idx = 0
    tasks_trained_so_far: list[str] = []
    global_step_offset = 0
    partial_task_resume_step: int | None = None

    if args.resume:
        for i, task_name in enumerate(args.tasks):
            task_slug = task_name.replace(" ", "_")[:50]
            task_dir = run_ckpt_root / f"task_{i:02d}_{task_slug}"
            if (task_dir / "metadata.json").exists():
                tasks_trained_so_far.append(task_name)
                global_step_offset += args.steps_per_task
                start_task_idx = i + 1
            elif task_dir.exists():
                tmp_mgr, has_ckpt = _checkpoints.initialize_checkpoint_dir(
                    task_dir, keep_period=None, overwrite=False, resume=True
                )
                if has_ckpt:
                    partial_task_resume_step = tmp_mgr.latest_step()
                    logging.info(
                        f"Partial task at task_{i:02d}: resuming from local step {partial_task_resume_step}"
                    )
                start_task_idx = i
                break
            else:
                break
        logging.info(
            f"Resume: {start_task_idx} completed task(s), partial_step={partial_task_resume_step}"
        )

    need_restore = args.resume and (start_task_idx > 0 or partial_task_resume_step is not None)
    train_state, train_state_sharding = base_sft.init_train_state(config, init_rng, mesh, resume=need_restore)
    if not need_restore:
        jax.block_until_ready(train_state)
        logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if need_restore:
        restore_idx = start_task_idx if partial_task_resume_step is not None else start_task_idx - 1
        restore_slug = args.tasks[restore_idx].replace(" ", "_")[:50]
        restore_dir = run_ckpt_root / f"task_{restore_idx:02d}_{restore_slug}"
        restore_mgr, _ = _checkpoints.initialize_checkpoint_dir(
            restore_dir, keep_period=None, overwrite=False, resume=True
        )
        train_state = _checkpoints.restore_state(restore_mgr, train_state, None)
        jax.block_until_ready(train_state)
        logging.info(f"Restored train state from {restore_dir} (train_state.step={int(train_state.step)})")

    ptrain_step = jax.jit(
        functools.partial(train_moe.train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    for task_idx, task_name in enumerate(args.tasks):
        if task_idx < start_task_idx:
            continue

        logging.info(
            f"\n{'=' * 64}\n"
            f"Task {task_idx + 1}/{len(args.tasks)}: {task_name}\n"
            f"{'=' * 64}"
        )

        safe_slug = task_name.replace(" ", "_")[:50]
        task_ckpt_dir = run_ckpt_root / f"task_{task_idx:02d}_{safe_slug}"
        is_partial = task_idx == start_task_idx and partial_task_resume_step is not None

        if is_partial:
            checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
                task_ckpt_dir,
                keep_period=None,
                overwrite=False,
                resume=True,
            )
            step_range = range(partial_task_resume_step + 1, args.steps_per_task)
            logging.info(
                f"Resuming partial task from local step {partial_task_resume_step}; "
                f"{len(step_range)} steps remaining."
            )
        else:
            checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
                task_ckpt_dir,
                keep_period=args.checkpoint_interval,
                overwrite=True,
                resume=False,
            )
            step_range = range(args.steps_per_task)

        data_loader = base_sft.create_task_data_loader(config, task_name, data_sharding)
        data_iter = iter(data_loader)
        batch = next(data_iter)

        pbar = tqdm.tqdm(
            step_range,
            total=len(step_range),
            dynamic_ncols=True,
            desc=f"[{task_idx:02d}] {task_name[:45]}",
        )

        infos = []
        task_step = partial_task_resume_step if is_partial else 0
        for task_step in pbar:
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
            infos.append(info)

            if task_step % config.log_interval == 0:
                stacked = common_utils.stack_forest(infos)
                reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
                info_str = _format_scalar_metrics(reduced)
                pbar.write(f"[task {task_idx:02d} step {task_step:05d}] {info_str}")
                wandb.log(
                    {
                        **{f"task_{task_idx:02d}/{k}": v for k, v in reduced.items()},
                        "task_index": task_idx,
                    },
                    step=global_step_offset + task_step,
                )
                infos = []

            if (
                args.checkpoint_interval is not None
                and task_step > 0
                and task_step % args.checkpoint_interval == 0
            ):
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, task_step)

            batch = next(data_iter)

        final_step = task_step
        _checkpoints.save_state(checkpoint_manager, train_state, data_loader, final_step)
        checkpoint_manager.wait_until_finished()

        tasks_trained_so_far.append(task_name)
        base_sft.save_metadata(task_ckpt_dir, list(tasks_trained_so_far), args, base_config)

        logging.info(
            f"Checkpoint saved: {task_ckpt_dir}\n"
            f"Tasks trained so far ({len(tasks_trained_so_far)}): "
            + ", ".join(f"'{task}'" for task in tasks_trained_so_far)
        )

        global_step_offset += args.steps_per_task

    logging.info("Sequential MoE fine-tuning complete.")
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(MoESFTArgs))
