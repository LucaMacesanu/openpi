# LIBERO 28-Task Joint MoE + 12-Task Sequential SFT Workflow

This workflow uses the first 7 tasks from each of `libero_10`, `libero_goal`,
`libero_object`, and `libero_spatial` for joint MoE training, then sequentially
fine-tunes on the remaining 3 tasks from each suite.

## Task Files

- Joint training list: `notes/tasks_libero_28_joint_train.txt`
- Sequential SFT list: `notes/tasks_libero_12_sft_remaining.txt`
- Per-suite split files:
  - `notes/tasks_libero_10_train7.txt`
  - `notes/tasks_libero_10_sft3.txt`
  - `notes/tasks_libero_goal_train7.txt`
  - `notes/tasks_libero_goal_sft3.txt`
  - `notes/tasks_libero_object_train7.txt`
  - `notes/tasks_libero_object_sft3.txt`
  - `notes/tasks_libero_spatial_train7.txt`
  - `notes/tasks_libero_spatial_sft3.txt`

## 1. Compute Norm Stats

Run this once if `assets/pi0_libero_residual_moe_28_train/libero_28_joint_train`
does not already contain `norm_stats.json`.

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_libero_residual_moe_28_train
```

## 2. Joint Train the 28 Selected Tasks

```bash
bash shells/train_moe_libero_28_joint.sh \
  --checkpoint_base_dir "$SCRATCH/openpi_ckpts" \
  --batch_size 64 \
  --resume true
```

Equivalent direct command:

```bash
uv run scripts/train_moe.py pi0_libero_residual_moe_28_train \
  --exp-name pi0_libero_residual_moe_libero_28_joint_30k \
  --checkpoint-base-dir "$SCRATCH/openpi_ckpts" \
  --num-train-steps 30000 \
  --batch-size 64 \
  --resume
```

The final checkpoint params path for 30,000 steps is:

```bash
$SCRATCH/openpi_ckpts/pi0_libero_residual_moe_28_train/pi0_libero_residual_moe_libero_28_joint_30k/29999/params
```

The directory is named `29999` because the training loop saves at zero-based
loop step `num_train_steps - 1`; the saved train state has advanced through
30,000 optimizer steps.

## 3. Sequential SFT on the 12 Held-Out Tasks

```bash
bash shells/train_sft_moe_libero_12_remaining.sh \
  --checkpoint_dir /scratch/tz2668/openpi/checkpoints/sft_checkpoints \
  --batch_size 64 \
  --resume true \
  --load_balance_loss_weight 0
```

Equivalent command through the general MoE SFT wrapper:

```bash
bash shells/train_sft_moe_local.sh \
  --tasks_files notes/tasks_libero_12_sft_remaining.txt \
  --exp_name pi0_libero_residual_moe_libero_12_remaining_sft \
  --config_name pi0_libero_residual_moe_28_train \
  --norm_stats_from pi0_libero_residual_moe_28_train \
  --checkpoint_dir /scratch/tz2668/openpi/checkpoints/sft_checkpoints \
  --batch_size 64 \
  --resume true \
  --load_balance_loss_weight 0 \
  --initial_checkpoint_params "$SCRATCH/openpi_ckpts/pi0_libero_residual_moe_28_train/pi0_libero_residual_moe_libero_28_joint_30k/29999/params"
```

## 4. Evaluate

```bash
UV_BIN="$(which uv)" bash shells/eval_sft_local.sh \
  --run_dir /scratch/tz2668/openpi/checkpoints/sft_checkpoints/pi0_libero_residual_moe_libero_12_remaining_sft \
  --num_trials 50 \
  --cuda_devices 0 \
  --max_parallel 1 \
  --force
```
