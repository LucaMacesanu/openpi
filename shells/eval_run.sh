#!/usr/bin/env bash
# eval_run.sh — Evaluate all task checkpoints in an SFT run folder.
#
# For each task_NN_* directory (in order):
#   1. Finds the latest orbax step dir inside it.
#   2. Starts the policy server on that checkpoint.
#   3. Runs eval_checkpoint.sh against the live server.
#   4. Kills the server before moving on.
#
# Usage:
#   bash shells/eval_run.sh \
#       --run_dir   checkpoints/sft_checkpoints/sft_run_0 \
#       --num_trials 10 \
#       --cuda_devices 4,6,7

set -euo pipefail

RUN_DIR=""
NUM_TRIALS=10
HOST="0.0.0.0"
PORT=8000
CUDA_DEVICES="0"
SERVER_TIMEOUT=300  # seconds to wait for server to come up

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_dir)        RUN_DIR="$2";        shift 2 ;;
        --num_trials)     NUM_TRIALS="$2";     shift 2 ;;
        --host)           HOST="$2";           shift 2 ;;
        --port)           PORT="$2";           shift 2 ;;
        --cuda_devices)   CUDA_DEVICES="$2";   shift 2 ;;
        --server_timeout) SERVER_TIMEOUT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    echo "Usage: $0 --run_dir <path> [--num_trials N] [--cuda_devices X,Y,Z] [--host H] [--port P]"
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

SERVER_PID=""

# Ensure the server is killed if this script exits for any reason.
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[eval_run] Killing policy server (PID=$SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for TASK_DIR in "${TASK_DIRS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Checkpoint: $(basename "$TASK_DIR")"
    echo "============================================================"

    # Find the latest orbax step dir (numeric subdirectory with the highest number).
    STEP_DIR=$(find "$TASK_DIR" -maxdepth 1 -mindepth 1 -type d \
               | grep -E '/[0-9]+$' | sort -t/ -k1 -V | tail -1)

    if [[ -z "$STEP_DIR" ]]; then
        echo "WARNING: No numeric step dir found in $TASK_DIR — skipping."
        continue
    fi
    echo "Step dir  : $STEP_DIR"

    # Kill previous server if still running.
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[eval_run] Stopping previous server (PID=$SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        # Brief pause to let the port free up.
        sleep 3
    fi

    # Start the policy server for this checkpoint.
    SERVER_PID=$(bash "$SCRIPT_DIR/serve_policy.sh" \
        --checkpoint_dir "$STEP_DIR" \
        --cuda_devices   "$CUDA_DEVICES" \
        --host           "$HOST" \
        --port           "$PORT" \
        --timeout        "$SERVER_TIMEOUT")

    echo "[eval_run] Server up (PID=$SERVER_PID)"

    # Evaluate all tasks for this checkpoint.
    bash "$SCRIPT_DIR/eval_checkpoint.sh" \
        --checkpoint_dir "$TASK_DIR" \
        --exp_dir        "$RUN_DIR" \
        --num_trials     "$NUM_TRIALS" \
        --host           "$HOST" \
        --port           "$PORT"
done

echo ""
echo "All checkpoints evaluated. Results in: $RUN_DIR/evals/"
