# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & Package Management

This repo uses [uv](https://docs.astral.sh/uv/) for dependency management. Always prefix Python commands with `uv run`.

```bash
# Install / sync dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

The repo targets Python 3.11 (see `.python-version`). The virtual environment is at `.venv/`.

## Common Commands

```bash
# Lint and format (ruff, line length 120)
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest

# Run a single test file
uv run pytest src/openpi/models/model_test.py

# Compute norm stats before training (required first time per config)
uv run scripts/compute_norm_stats.py --config-name pi05_libero_sft

# Standard JAX training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite

# PyTorch training (single GPU)
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Serve a trained policy checkpoint
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero_sft --policy.dir=checkpoints/.../5000
```

### This Fork's SFT / Continual-Learning Scripts

```bash
# Sequential SFT on N randomly sampled LIBERO tasks (JAX, continual learning)
bash shells/run_sft.sh \
    --num_tasks 10 \
    --exp_name sft_run_0 \
    --config_name pi05_libero_sft \
    [--steps_per_task 5000] \
    [--cuda_devices 4,7] \
    [--batch_size 32] \
    [--wandb_enabled true]

# Dynamic MoE variant (with adaptive expert add/remove between tasks)
bash shells/run_dynamic_sft.sh \
    --num_tasks 10 \
    --exp_name dyn_moe_run_0 \
    --config_name pi05_libero_dyn_moe \
    [--num_record_batches 50]

# Evaluate all task checkpoints in a run folder
bash shells/eval_run.sh \
    --run_dir checkpoints/sft_checkpoints/sft_run_0 \
    --num_trials 10 \
    --cuda_devices 4

# Evaluate multiple run folders in parallel (one GPU per run)
bash shells/eval_multi_run.sh \
    --run_dir checkpoints/sft_checkpoints/sft_run_0 \
    --run_dir checkpoints/sft_checkpoints/moe_run_0 \
    --cuda_devices 0,1,2,3

# Compile evaluation results into a summary
uv run scripts/compile_eval_results.py checkpoints/sft_checkpoints/sft_run_0/evals
```

### Slurm (NYU HPC)

The `sbatches/` directory has Slurm scripts for the cluster (h100/h200 partitions). They wrap the same `shells/` scripts.

## Architecture Overview

### Model Family

Three VLA model variants, all built on a **PaliGemma backbone** (SigLIP vision encoder + Gemma LLM):

| Model | Key | File |
|---|---|---|
| π₀ / π₀.₅ | Flow-matching action expert | `src/openpi/models/pi0.py` |
| π₀-FAST | Autoregressive with FAST tokenizer | `src/openpi/models/pi0_fast.py` |
| π₀-MoE | Sparse MoE action expert | `src/openpi/models/pi0_moe.py` |
| π₀-DynMoE | Dynamic MoE (this fork) | `src/openpi/models/pi0_dyn_moe.py` |

All models inherit from `BaseModel` in `src/openpi/models/model.py` and implement `compute_loss` and `sample_actions`.

### Dual-Expert Design

Each transformer layer has **two separate FFW/projection sets**:
- **Expert 0** (index 0): PaliGemma backbone, processes image+language tokens, frozen during fine-tuning
- **Expert 1** (index 1): Action expert, processes action+timestep tokens, trained with LoRA

The two streams share the same attention mechanism (cross-attention is allowed via concatenated positions/masks) but have independent weights. This is the key architectural pattern — "dual expert" is not MoE, it's two separate networks for two token types.

### Config & Training System

`src/openpi/training/config.py` is the central registry. `_CONFIGS` maps string names to `TrainConfig` instances. Key configs in this fork:
- `pi05_libero_sft` — baseline LoRA SFT on LIBERO
- `pi05_libero_moe` — static 4-expert MoE, top-2 routing
- `pi05_libero_dyn_moe` — DynMoE with adaptive expert add/remove

`TrainConfig` holds: model config, data config, optimizer/LR schedule, weight loader, freeze filter, and ema settings.

### Data Pipeline

Data flows: LeRobot dataset → `repack_transforms` → norm stats → `data_transforms` → `model_transforms` → batched `(Observation, Actions)`.

`Observation` and `Actions` are typed dataclasses in `src/openpi/models/model.py`. Robot-specific input/output mapping lives in `src/openpi/policies/` (e.g., `libero_policy.py`).

Norm stats must be computed once with `compute_norm_stats.py` and are cached in the `assets/` directory under each config name.

### Weight Loading

`WeightLoader` subclasses in `src/openpi/training/weight_loaders.py` handle loading pre-trained checkpoints. For MoE models, `MoEWeightLoader` fans out a single action expert FFW to N expert slots. `DynMoEWeightLoader` (`src/openpi/training/dyn_moe_weight_loader.py`) similarly fans out to `max_experts` pre-allocated slots.

### DynMoE (This Fork's Research Addition)

`src/openpi/models/dyn_moe.py` implements the DynMoE paper:
- **Top-Any gating**: sigmoid thresholds (not softmax+top-k); k varies per token
- **Diverse-and-Simple auxiliary loss**: orthogonality + magnitude on expert representations
- **`adaptive_update_experts()`**: called between tasks to prune dead experts and seed new ones from unrouted token embeddings

The `experts_mask` param (`(depth, max_experts)` binary) controls which expert slots are active. It is frozen from the optimizer but mutated directly via NNX state between tasks.

`scripts/dynamic_sft_train.py` mirrors `scripts/sft_train.py` and calls `adaptive_update_experts()` after each task completes.

### Inference / Serving

`scripts/serve_policy.py` launches a WebSocket server (`src/openpi/serving/websocket_policy_server.py`). The `openpi-client` package (in `packages/openpi-client/`) provides the client-side interface for robot runtimes to stream actions from the server.

## Active Fine-Tuning Project: YOR Robot

Fine-tuning π₀.₅ on real-robot data for `"place the orange cube on the plate"` (right arm only).

**Full plan and progress**: `/local_data/lim2045/vla/yor_data/place_the_orange_cube_on_the_plate/CLAUDE.md`

Key files added for this project:
- `src/openpi/policies/yor_policy.py` — `YorRightArmInputs` / `YorRightArmOutputs`
- `src/openpi/training/config.py` — `LeRobotYorDataConfig` class + `pi05_yor_right_arm` config entry
- `scripts/filter_success_episodes.py` — one-time script to create success-only dataset copy

Filtered dataset (48 success episodes): `/local_data/lim2045/vla/yor_data/place_the_orange_cube_on_the_plate_success`

**Status**: **Ready to train** — delta joint action space implemented. GPUs 3+7 free.

Active config: `pi0_yor_right_arm` (π₀ LoRA, not π₀.₅). Also `pi05_yor_right_arm` available.

Training command:
```bash
HF_LEROBOT_HOME=/local_data/lim2045/vla/yor_data \
CUDA_VISIBLE_DEVICES=3,7 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi0_yor_right_arm --exp-name yor_right_arm_run_0 --overwrite
```

## Key File Locations

| Purpose | Path |
|---|---|
| All named training configs | `src/openpi/training/config.py` |
| Base model interface | `src/openpi/models/model.py` |
| LoRA layers | `src/openpi/models/lora.py` |
| Data loading / transforms | `src/openpi/training/data_loader.py`, `src/openpi/transforms.py` |
| Checkpoint utilities | `src/openpi/training/checkpoints.py` |
| MoE layers | `src/openpi/models/moe.py` |
| DynMoE layers | `src/openpi/models/dyn_moe.py` |
| Architecture notes | `notes/architecture.md`, `notes/dynmoe_plan.md` |

## JAX-Specific Notes

- Training uses Flax NNX throughout. Models are created via `config.model.create(rng)`.
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` (or higher) to avoid OOM during training.
- `jax.jit` compilation cache is written to `~/.cache/jax`.
- FSDP sharding via `src/openpi/training/sharding.py`; configure with `fsdp_devices` in `TrainConfig`.
- Avoid Python `if deterministic:` branches inside `nn.scan` bodies — use `jnp.where` instead (traced bool constraint).

## PyTorch Support

A PyTorch implementation of π₀/π₀.₅ lives in `src/openpi/models_pytorch/`. Before use, apply patches to the installed transformers library:

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

**Warning**: with hardlink mode (uv default) this permanently modifies the transformers cache. To undo: `uv cache clean transformers`.
