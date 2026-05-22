#!/usr/bin/env bash
# Joint MoE training for the 28-task LIBERO split.
#
# Trains config pi0_libero_residual_moe_28_train for 30,000 steps by default.

set -euo pipefail

CONFIG_NAME="pi0_libero_residual_moe_28_train"
EXP_NAME="pi0_libero_residual_moe_libero_28_joint_30k"
CHECKPOINT_BASE_DIR="${SCRATCH:-/scratch/tz2668}/openpi_ckpts"
NUM_TRAIN_STEPS=30000
BATCH_SIZE=64
CUDA_DEVICES=""
SAVE_INTERVAL=1000
KEEP_PERIOD=5000
WANDB_ENABLED=true
RESUME=true
OVERWRITE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp_name)             EXP_NAME="$2";             shift 2 ;;
        --checkpoint_base_dir)  CHECKPOINT_BASE_DIR="$2";  shift 2 ;;
        --num_train_steps)      NUM_TRAIN_STEPS="$2";      shift 2 ;;
        --batch_size)           BATCH_SIZE="$2";           shift 2 ;;
        --cuda_devices)         CUDA_DEVICES="$2";         shift 2 ;;
        --save_interval)        SAVE_INTERVAL="$2";        shift 2 ;;
        --keep_period)          KEEP_PERIOD="$2";          shift 2 ;;
        --wandb_enabled)        WANDB_ENABLED="$2";        shift 2 ;;
        --resume)               RESUME="$2";               shift 2 ;;
        --overwrite)            OVERWRITE="$2";            shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$RESUME" == "true" && "$OVERWRITE" == "true" ]]; then
    echo "Error: --resume true and --overwrite true are mutually exclusive."
    exit 1
fi

FINAL_STEP=$((NUM_TRAIN_STEPS - 1))
FINAL_PARAMS="$CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME/$FINAL_STEP/params"

CMD=(
    uv run scripts/train_moe.py "$CONFIG_NAME"
    --exp-name "$EXP_NAME"
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
    --num-train-steps "$NUM_TRAIN_STEPS"
    --batch-size "$BATCH_SIZE"
    --save-interval "$SAVE_INTERVAL"
    --keep-period "$KEEP_PERIOD"
)

if [[ "$WANDB_ENABLED" == "true" ]]; then
    CMD+=(--wandb-enabled)
else
    CMD+=(--no-wandb-enabled)
fi

if [[ "$RESUME" == "true" ]]; then
    CMD+=(--resume)
fi

if [[ "$OVERWRITE" == "true" ]]; then
    CMD+=(--overwrite)
fi

echo "============================================================"
echo "Joint MoE LIBERO 28-task training"
echo "  Config      : $CONFIG_NAME"
echo "  Experiment  : $EXP_NAME"
echo "  Steps       : $NUM_TRAIN_STEPS"
echo "  Tasks       : 28 (notes/tasks_libero_28_joint_train.txt)"
echo "  Batch size  : $BATCH_SIZE"
echo "  Checkpoints : $CHECKPOINT_BASE_DIR/$CONFIG_NAME/$EXP_NAME"
echo "  Final params: $FINAL_PARAMS"
echo "  Resume      : $RESUME"
echo "  Overwrite   : $OVERWRITE"
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
