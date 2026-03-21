#!/usr/bin/env bash
# eval_run.sh — Evaluate all task checkpoints in an SFT run folder.
#
# Iterates over every task_NN_* subdirectory in the run folder (in order)
# and calls eval_checkpoint.sh on each one.
#
# Usage:
#   bash shells/eval_run.sh \
#       --run_dir checkpoints/sft_checkpoints/sft_run_0 \
#       --num_trials 10

set -euo pipefail

RUN_DIR=""
NUM_TRIALS=10
HOST="0.0.0.0"
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_dir)    RUN_DIR="$2";    shift 2 ;;
        --num_trials) NUM_TRIALS="$2"; shift 2 ;;
        --host)       HOST="$2";       shift 2 ;;
        --port)       PORT="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    echo "Usage: $0 --run_dir <path> [--num_trials N] [--host H] [--port P]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Collect task checkpoint dirs in sorted order (task_00, task_01, ...).
mapfile -t TASK_DIRS < <(find "$RUN_DIR" -maxdepth 1 -type d -name "task_*" | sort)

if [[ ${#TASK_DIRS[@]} -eq 0 ]]; then
    echo "No task_* directories found in $RUN_DIR"
    exit 1
fi

echo "Found ${#TASK_DIRS[@]} task checkpoint(s) in $RUN_DIR"

for TASK_DIR in "${TASK_DIRS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Evaluating: $(basename "$TASK_DIR")"
    echo "============================================================"
    bash "$SCRIPT_DIR/eval_checkpoint.sh" \
        --checkpoint_dir "$TASK_DIR" \
        --exp_dir "$RUN_DIR" \
        --num_trials "$NUM_TRIALS" \
        --host "$HOST" \
        --port "$PORT"
done

echo ""
echo "All checkpoints evaluated. Results in: $RUN_DIR/evals/"
