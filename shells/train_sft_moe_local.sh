#!/usr/bin/env bash
# train_sft_moe_local.sh — Local wrapper for ordered SFT runs with MoE metrics.
#
# Same arguments as shells/train_sft_local.sh / shells/run_sft_ordered.sh, but
# launches scripts/sft_train_moe.py instead of scripts/sft_train.py.
#
# Usage:
#   bash shells/train_sft_moe_local.sh \
#       --tasks_files notes/tasks_libero_spatial.txt \
#       --exp_name pi0_libero_residual_moe_spatial_only_sft \
#       --config_name pi0_libero_residual_moe_spatial_only \
#       --norm_stats_from pi0_libero_residual_moe_spatial_only \
#       --checkpoint_dir /scratch/tz2668/openpi/checkpoints/sft_checkpoints \
#       --batch_size 64 \
#       --resume true \
#       --initial_checkpoint_params /path/to/joint_ckpt/29999/params

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TASKS_FILES=""
EXP_NAME="sft_ordered_run"
STEPS_PER_TASK=5000
CHECKPOINT_DIR="/scratch/lim2045/openpi/checkpoints/sft_checkpoints"
CUDA_DEVICES=""
BATCH_SIZE=32
WANDB_ENABLED=true
CONFIG_NAME="pi05_libero_sft"
NORM_STATS_FROM=""
CHECKPOINT_INTERVAL=""
RESUME=false
LOAD_BALANCE_LOSS_WEIGHT=""
INITIAL_CHECKPOINT_PARAMS=""
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks_files)         TASKS_FILES="$2";          shift 2 ;;
        --exp_name)            EXP_NAME="$2";             shift 2 ;;
        --steps_per_task)      STEPS_PER_TASK="$2";       shift 2 ;;
        --checkpoint_dir)      CHECKPOINT_DIR="$2";       shift 2 ;;
        --cuda_devices)        CUDA_DEVICES="$2";         shift 2 ;;
        --batch_size)          BATCH_SIZE="$2";           shift 2 ;;
        --wandb_enabled)       WANDB_ENABLED="$2";        shift 2 ;;
        --config_name)         CONFIG_NAME="$2";          shift 2 ;;
        --norm_stats_from)     NORM_STATS_FROM="$2";      shift 2 ;;
        --checkpoint_interval) CHECKPOINT_INTERVAL="$2";  shift 2 ;;
        --resume)              RESUME="$2";               shift 2 ;;
        --load_balance_loss_weight) LOAD_BALANCE_LOSS_WEIGHT="$2"; shift 2 ;;
        --initial_checkpoint_params) INITIAL_CHECKPOINT_PARAMS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$TASKS_FILES" ]]; then
    echo "Usage: $0 --tasks_files FILE[,FILE,...] --exp_name NAME [options]"
    echo ""
    echo "Options:"
    echo "  --tasks_files     FILE[,FILE,...]  Comma-separated task list files (required)"
    echo "  --exp_name        NAME             Experiment name (default: sft_ordered_run)"
    echo "  --steps_per_task  N                Gradient steps per task (default: 5000)"
    echo "  --checkpoint_dir  PATH             Root checkpoint directory"
    echo "  --cuda_devices    0,1,...           CUDA_VISIBLE_DEVICES value (default: all)"
    echo "  --batch_size      N                Global batch size (default: 32)"
    echo "  --config_name     NAME             Training config (default: pi05_libero_sft)"
    echo "  --norm_stats_from NAME             Config for norm stats (default: selected config)"
    echo "  --wandb_enabled   true|false       W&B logging (default: true)"
    echo "  --checkpoint_interval N            Save intermediate checkpoint every N steps"
    echo "  --resume          true|false       Resume from last completed/partial checkpoint (default: false)"
    echo "  --load_balance_loss_weight W       Override MoE LBL weight; set 0 to disable"
    echo "  --initial_checkpoint_params PATH   Initialize first SFT task from checkpoint params"
    exit 1
fi

# ---------------------------------------------------------------------------
# Read tasks from file(s)
# ---------------------------------------------------------------------------
TASKS=()
IFS=',' read -ra FILE_LIST <<< "$TASKS_FILES"
for tasks_file in "${FILE_LIST[@]}"; do
    if [[ ! -f "$tasks_file" ]]; then
        echo "Error: tasks file not found: $tasks_file"
        exit 1
    fi
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip blank lines and comments.
        [[ -z "$line" || "$line" == "#"* ]] && continue
        TASKS+=("$line")
    done < "$tasks_file"
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "Error: no tasks found in $TASKS_FILES"
    exit 1
fi

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
CMD=(
    uv run scripts/sft_train_moe.py
    --exp_name        "$EXP_NAME"
    --steps_per_task  "$STEPS_PER_TASK"
    --checkpoint_dir  "$CHECKPOINT_DIR"
    --batch_size      "$BATCH_SIZE"
    --config_name     "$CONFIG_NAME"
    --tasks           "${TASKS[@]}"
)

if [[ -n "$NORM_STATS_FROM" ]]; then
    CMD+=(--norm_stats_from "$NORM_STATS_FROM")
fi

if [[ "$WANDB_ENABLED" == "true" ]]; then
    CMD+=(--wandb_enabled)
else
    CMD+=(--no-wandb_enabled)
fi

if [[ -n "$CHECKPOINT_INTERVAL" ]]; then
    CMD+=(--checkpoint_interval "$CHECKPOINT_INTERVAL")
fi

if [[ -n "$LOAD_BALANCE_LOSS_WEIGHT" ]]; then
    CMD+=(--load_balance_loss_weight "$LOAD_BALANCE_LOSS_WEIGHT")
fi

if [[ -n "$INITIAL_CHECKPOINT_PARAMS" ]]; then
    CMD+=(--initial_checkpoint_params "$INITIAL_CHECKPOINT_PARAMS")
fi

if [[ "$RESUME" == "true" ]]; then
    CMD+=(--resume)
else
    CMD+=(--no-resume)
fi

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "SFT MoE Ordered Run"
echo "  Experiment  : $EXP_NAME"
echo "  Tasks       : ${#TASKS[@]} (from: $TASKS_FILES)"
for i in "${!TASKS[@]}"; do
    printf "    %02d: %s\n" "$i" "${TASKS[$i]}"
done
echo "  Config      : $CONFIG_NAME"
echo "  Norm stats  : ${NORM_STATS_FROM:-$CONFIG_NAME}"
echo "  Steps/task  : $STEPS_PER_TASK"
echo "  Checkpoint  : $CHECKPOINT_DIR/$EXP_NAME"
echo "  Script      : scripts/sft_train_moe.py"
if [[ -n "$CHECKPOINT_INTERVAL" ]]; then
    echo "  Ckpt interval: every $CHECKPOINT_INTERVAL steps"
fi
if [[ -n "$LOAD_BALANCE_LOSS_WEIGHT" ]]; then
    echo "  LBL weight   : $LOAD_BALANCE_LOSS_WEIGHT"
fi
if [[ -n "$INITIAL_CHECKPOINT_PARAMS" ]]; then
    echo "  Init params  : $INITIAL_CHECKPOINT_PARAMS"
fi
echo "  Resume      : $RESUME"
if [[ -n "$CUDA_DEVICES" ]]; then
    echo "  CUDA devices: $CUDA_DEVICES"
fi
echo "============================================================"
echo ""

if [[ -n "$CUDA_DEVICES" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "${CMD[@]}"
else
    "${CMD[@]}"
fi
