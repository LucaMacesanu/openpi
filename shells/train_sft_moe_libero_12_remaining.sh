#!/usr/bin/env bash
# Sequential MoE SFT for the 12 held-out LIBERO tasks.
#
# Initializes from the checkpoint produced by shells/train_moe_libero_28_joint.sh
# unless --initial_checkpoint_params is provided explicitly.

set -euo pipefail

TASKS_FILES="notes/tasks_libero_12_sft_remaining.txt"
EXP_NAME="pi0_libero_residual_moe_libero_12_remaining_sft"
CONFIG_NAME="pi0_libero_residual_moe_28_train"
NORM_STATS_FROM="pi0_libero_residual_moe_28_train"
CHECKPOINT_DIR="/scratch/tz2668/openpi/checkpoints/sft_checkpoints"
STEPS_PER_TASK=5000
BATCH_SIZE=64
CUDA_DEVICES=""
WANDB_ENABLED=true
RESUME=true
LOAD_BALANCE_LOSS_WEIGHT=0
CHECKPOINT_INTERVAL=""

JOINT_CONFIG_NAME="pi0_libero_residual_moe_28_train"
JOINT_EXP_NAME="pi0_libero_residual_moe_libero_28_joint_30k"
JOINT_CHECKPOINT_BASE_DIR="${SCRATCH:-/scratch/tz2668}/openpi_ckpts"
JOINT_FINAL_STEP=29999
INITIAL_CHECKPOINT_PARAMS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks_files)               TASKS_FILES="$2";               shift 2 ;;
        --exp_name)                  EXP_NAME="$2";                  shift 2 ;;
        --config_name)               CONFIG_NAME="$2";               shift 2 ;;
        --norm_stats_from)           NORM_STATS_FROM="$2";           shift 2 ;;
        --checkpoint_dir)            CHECKPOINT_DIR="$2";            shift 2 ;;
        --steps_per_task)            STEPS_PER_TASK="$2";            shift 2 ;;
        --batch_size)                BATCH_SIZE="$2";                shift 2 ;;
        --cuda_devices)              CUDA_DEVICES="$2";              shift 2 ;;
        --wandb_enabled)             WANDB_ENABLED="$2";             shift 2 ;;
        --resume)                    RESUME="$2";                    shift 2 ;;
        --load_balance_loss_weight)  LOAD_BALANCE_LOSS_WEIGHT="$2";  shift 2 ;;
        --checkpoint_interval)       CHECKPOINT_INTERVAL="$2";       shift 2 ;;
        --joint_config_name)         JOINT_CONFIG_NAME="$2";         shift 2 ;;
        --joint_exp_name)            JOINT_EXP_NAME="$2";            shift 2 ;;
        --joint_checkpoint_base_dir) JOINT_CHECKPOINT_BASE_DIR="$2"; shift 2 ;;
        --joint_final_step)          JOINT_FINAL_STEP="$2";          shift 2 ;;
        --initial_checkpoint_params) INITIAL_CHECKPOINT_PARAMS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$INITIAL_CHECKPOINT_PARAMS" ]]; then
    INITIAL_CHECKPOINT_PARAMS="$JOINT_CHECKPOINT_BASE_DIR/$JOINT_CONFIG_NAME/$JOINT_EXP_NAME/$JOINT_FINAL_STEP/params"
fi

CMD=(
    bash shells/train_sft_moe_local.sh
    --tasks_files "$TASKS_FILES"
    --exp_name "$EXP_NAME"
    --config_name "$CONFIG_NAME"
    --norm_stats_from "$NORM_STATS_FROM"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --steps_per_task "$STEPS_PER_TASK"
    --batch_size "$BATCH_SIZE"
    --wandb_enabled "$WANDB_ENABLED"
    --resume "$RESUME"
    --load_balance_loss_weight "$LOAD_BALANCE_LOSS_WEIGHT"
    --initial_checkpoint_params "$INITIAL_CHECKPOINT_PARAMS"
)

if [[ -n "$CUDA_DEVICES" ]]; then
    CMD+=(--cuda_devices "$CUDA_DEVICES")
fi

if [[ -n "$CHECKPOINT_INTERVAL" ]]; then
    CMD+=(--checkpoint_interval "$CHECKPOINT_INTERVAL")
fi

echo "============================================================"
echo "Sequential MoE SFT on held-out LIBERO tasks"
echo "  Tasks       : $TASKS_FILES"
echo "  Experiment  : $EXP_NAME"
echo "  Config      : $CONFIG_NAME"
echo "  Norm stats  : $NORM_STATS_FROM"
echo "  Init params : $INITIAL_CHECKPOINT_PARAMS"
echo "  Checkpoint  : $CHECKPOINT_DIR/$EXP_NAME"
echo "============================================================"
echo ""

"${CMD[@]}"
