# SFT Training Pipeline

## Invocation chain

```
shells/submit_sft.sh  →  sbatches/train_sft_ordered.sh  →  shells/run_sft_ordered.sh  →  scripts/sft_train.py
```

### 1. `shells/submit_sft.sh`
Thin wrapper. Extracts `--exp_name` to set the SLURM job name as
`sft_train_<exp_name>`, then calls `sbatch sbatches/train_sft_ordered.sh "$@"`.
All other args are passed through unchanged.

```bash
bash shells/submit_sft.sh \
    --tasks_files notes/tasks_libero_spatial.txt \
    --exp_name spatial_run_0 \
    [--steps_per_task 5000] \
    [--config_name pi05_libero_sft] \
    [--resume true]
```

### 2. `sbatches/train_sft_ordered.sh`
SLURM job script. Key resource settings:
- Partition: `h200_tandon` or `h100_tandon`, constraint `h100|h200`
- 2 GPUs, 16 CPUs, 200 GB RAM, 24-hour wall time
- Account: `torch_pr_50_tandon_advanced`
- Logs: `logs/<jobid>_train_sft_ordered.out`

Sets up the environment: enters singularity container
(`cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif`), activates `.venv`, sets
`WANDB_API_KEY`, then delegates to `shells/run_sft_ordered.sh "$@"`.

### 3. `shells/run_sft_ordered.sh`
Parses CLI args. Reads one or more task files (comma-separated via
`--tasks_files`; lines starting with `#` are skipped) and accumulates task
strings into a bash array. Constructs and runs the `uv run scripts/sft_train.py`
command with all tasks passed as positional `--tasks` arguments.

Key defaults:
| flag | default |
|---|---|
| `--config_name` | `pi05_libero_sft` |
| `--norm_stats_from` | selected config |
| `--steps_per_task` | 5000 |
| `--batch_size` | 32 |
| `--checkpoint_dir` | `/scratch/lim2045/openpi/checkpoints/sft_checkpoints` |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.8 |

Multiple task files are concatenated in order, so cross-suite sequential runs
are possible with a single sbatch:
```bash
--tasks_files notes/tasks_libero_spatial.txt,notes/tasks_libero_object.txt
```

### 4. `scripts/sft_train.py`
The actual training logic. Key behaviors:

**Config loading**: calls `_config.get_config(args.config_name)` from
`src/openpi/training/config.py`. If `--norm_stats_from` is set, it overrides
`name` so asset/norm-stat paths resolve against an existing pre-computed
directory; otherwise the selected config's own assets are used.

**Data loading**: loads the *full* `physical-intelligence/libero` LeRobot dataset,
then filters to per-task frames by matching `task_index` at the frame level
(using `torch.utils.data.Subset`). This avoids the sparse episode-index bug
in lerobot v2.0. The libero "suite" (spatial / object / 10) is not encoded in
the config — it's implicit in whichever task strings you pass.

**Checkpoint layout**:
```
{checkpoint_dir}/{exp_name}/
    task_00_{task_slug}/
        {step}/              # orbax checkpoint
        metadata.json        # cumulative task list + config snapshot
    task_01_{task_slug}/
        ...
```

**State continuity**: model weights and optimizer state carry over across tasks;
only the data loader changes. The LR schedule therefore runs continuously across
the full sequential training (step counter never resets).

**Resume**: `--resume` scans task dirs, finds the last completed task (has
`metadata.json`) and the last partial task (has an orbax checkpoint but no
`metadata.json`), restores state, and continues from where training stopped.

**W&B**: single run per experiment covering all tasks. Metrics logged as
`task_XX/<metric>` with `task_index` as a scalar.

---

## Config system (`src/openpi/training/config.py`)

Configs are `TrainConfig` dataclass instances registered in a list. Retrieve
one with `_config.get_config(name)`. Key fields relevant to SFT:

| field | purpose |
|---|---|
| `model` | `Pi0Config(pi05=True, paligemma_variant=..., action_expert_variant=...)` |
| `data` | `LeRobotLiberoDataConfig(repo_id=..., extra_delta_transform=...)` |
| `weight_loader` | loads from GCS checkpoint path |
| `freeze_filter` | which params to freeze; SFT configs freeze all except action-expert LoRA |
| `lr_schedule` | `CosineDecaySchedule` or `RsqrtDecaySchedule` from `optimizer.py` |
| `optimizer` | `AdamW(weight_decay=...)` |
| `ema_decay` | set to `None` for all SFT configs |

The canonical SFT baseline config is **`pi05_libero_sft`**: pi05 model,
`gemma_2b` backbone (full weights, frozen), `gemma_300m_lora` action expert
(only LoRA adapters trainable), default cosine schedule
(warmup=1000, peak=2.5e-5, decay over 30k steps to 2.5e-6).

## LR schedules (`src/openpi/training/optimizer.py`)

| class | description |
|---|---|
| `CosineDecaySchedule` | linear warmup → cosine decay; fields: `warmup_steps`, `peak_lr`, `decay_steps`, `decay_lr` |
| `RsqrtDecaySchedule` | linear warmup → inverse-sqrt decay; fields: `warmup_steps`, `peak_lr`, `timescale` |

To get a flat (no-decay) schedule, either add a `ConstantSchedule` class or
set `decay_lr = peak_lr` on `CosineDecaySchedule`.

## Task files

| file | suite |
|---|---|
| `notes/tasks_libero_spatial.txt` | LIBERO-Spatial (10 tasks) |
| `notes/tasks_libero_object.txt` | LIBERO-Object |
| `notes/tasks_libero_10.txt` | LIBERO-10 |
| `notes/canonical_task_order.txt` | canonical ordering reference |

Task strings must match exactly what is stored in the LeRobot dataset
(`dataset_meta.tasks`). Use `--list_tasks` flag on `sft_train.py` to
enumerate available strings.
