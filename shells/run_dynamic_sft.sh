#!/usr/bin/env bash
# run_dynamic_sft.sh — Launch a DynMoE sequential fine-tuning run.
#
# Runs dynamic_sft_train.py, which mirrors sft_train.py with one addition:
# after each task, adaptive_update_experts() prunes dead experts and seeds
# new ones from unrouted token embeddings.
#
# Usage:
'''
   bash shells/run_dynamic_sft.sh \
       --num_tasks 5 \
       --exp_name dyn_moe_run_0 \
       [--steps_per_task 5000] \
       [--checkpoint_dir /local_data/lim2045/openpi/checkpoints/sft_checkpoints] \
       [--task_seed 42] \
       [--cuda_devices 4,7] \
       [--batch_size 32] \
       [--num_record_batches 50] \
       [--wandb_enabled true]
'''
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
NUM_TASKS=0
EXP_NAME="dyn_moe_run"
STEPS_PER_TASK=5000
CHECKPOINT_DIR="/local_data/lim2045/openpi/checkpoints/sft_checkpoints"
TASK_SEED=42
CUDA_DEVICES=""
BATCH_SIZE=32
WANDB_ENABLED=true
CONFIG_NAME="pi05_libero_dyn_moe"
NORM_STATS_FROM="pi0_libero_low_mem_finetune"
CHECKPOINT_INTERVAL=""
NUM_RECORD_BATCHES=50
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_tasks)           NUM_TASKS="$2";           shift 2 ;;
        --exp_name)            EXP_NAME="$2";            shift 2 ;;
        --steps_per_task)      STEPS_PER_TASK="$2";      shift 2 ;;
        --checkpoint_dir)      CHECKPOINT_DIR="$2";      shift 2 ;;
        --task_seed)           TASK_SEED="$2";           shift 2 ;;
        --cuda_devices)        CUDA_DEVICES="$2";        shift 2 ;;
        --batch_size)          BATCH_SIZE="$2";          shift 2 ;;
        --wandb_enabled)       WANDB_ENABLED="$2";       shift 2 ;;
        --config_name)         CONFIG_NAME="$2";         shift 2 ;;
        --norm_stats_from)     NORM_STATS_FROM="$2";     shift 2 ;;
        --checkpoint_interval) CHECKPOINT_INTERVAL="$2"; shift 2 ;;
        --num_record_batches)  NUM_RECORD_BATCHES="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$NUM_TASKS" -eq 0 ]]; then
    echo "Usage: $0 --num_tasks N --exp_name NAME [options]"
    echo ""
    echo "Options:"
    echo "  --num_tasks           N        Number of tasks to sample (required)"
    echo "  --exp_name            NAME     Experiment name (default: dyn_moe_run)"
    echo "  --steps_per_task      N        Gradient steps per task (default: 5000)"
    echo "  --checkpoint_dir      PATH     Root checkpoint directory"
    echo "  --task_seed           N        Seed for task sampling (default: 42)"
    echo "  --cuda_devices        0,1,...  CUDA_VISIBLE_DEVICES value (default: all)"
    echo "  --batch_size          N        Global batch size (default: 32)"
    echo "  --num_record_batches  N        Batches for adaptive expert update (default: 50)"
    echo "  --wandb_enabled       true|false (default: true)"
    echo "  --checkpoint_interval N        Save intermediate checkpoint every N steps (default: off)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
CMD=(
    uv run scripts/dynamic_sft_train.py
    --num_tasks           "$NUM_TASKS"
    --task_seed           "$TASK_SEED"
    --exp_name            "$EXP_NAME"
    --steps_per_task      "$STEPS_PER_TASK"
    --checkpoint_dir      "$CHECKPOINT_DIR"
    --batch_size          "$BATCH_SIZE"
    --config_name         "$CONFIG_NAME"
    --norm_stats_from     "$NORM_STATS_FROM"
    --num_record_batches  "$NUM_RECORD_BATCHES"
)

if [[ "$WANDB_ENABLED" == "true" ]]; then
    CMD+=(--wandb_enabled)
else
    CMD+=(--no-wandb_enabled)
fi

if [[ -n "$CHECKPOINT_INTERVAL" ]]; then
    CMD+=(--checkpoint_interval "$CHECKPOINT_INTERVAL")
fi

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR="/local_data/lim2045/openpi/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/dynamic_sft_${EXP_NAME}_${TIMESTAMP}.log"

echo "============================================================"
echo "DynMoE SFT Run"
echo "  Experiment       : $EXP_NAME"
echo "  Tasks            : $NUM_TASKS (task_seed=$TASK_SEED)"
echo "  Config           : $CONFIG_NAME"
echo "  Steps/task       : $STEPS_PER_TASK"
echo "  Record batches   : $NUM_RECORD_BATCHES (for adaptive expert update)"
echo "  Checkpoint       : $CHECKPOINT_DIR/$EXP_NAME"
if [[ -n "$CHECKPOINT_INTERVAL" ]]; then
    echo "  Ckpt interval    : every $CHECKPOINT_INTERVAL steps"
fi
if [[ -n "$CUDA_DEVICES" ]]; then
    echo "  CUDA devices     : $CUDA_DEVICES"
fi
echo "  Log file         : $LOG_FILE"
echo "============================================================"
echo ""

if [[ -n "$CUDA_DEVICES" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
fi
