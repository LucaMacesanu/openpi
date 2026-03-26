#!/usr/bin/env bash
# eval_multi_run.sh — Evaluate multiple SFT run folders in parallel.
#
# Spawns one eval_run.sh worker per available GPU, so all GPUs stay busy.
# Each worker gets its own CUDA device and port offset (PORT+slot).
# Logs for each run go to logs/eval_<run_name>_gpu<N>.out.
#
# Usage:
#   bash shells/eval_multi_run.sh \
#       --run_dir checkpoints/sft_checkpoints/moe_run_0 \
#       --run_dir checkpoints/sft_checkpoints/sft_run_1 \
#       [--num_trials 10] \
#       [--cuda_devices 0,1,2,3] \
#       [--server_timeout 300]

set -uo pipefail

RUN_DIRS=()
NUM_TRIALS=10
CUDA_DEVICES="0,1,2,3"
SERVER_TIMEOUT=300
HOST="0.0.0.0"
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_dir)        RUN_DIRS+=("$2");      shift 2 ;;
        --num_trials)     NUM_TRIALS="$2";       shift 2 ;;
        --cuda_devices)   CUDA_DEVICES="$2";     shift 2 ;;
        --server_timeout) SERVER_TIMEOUT="$2";   shift 2 ;;
        --host)           HOST="$2";             shift 2 ;;
        --port)           PORT="$2";             shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo "Usage: $0 --run_dir <path> [--run_dir <path> ...] [options]"
    echo ""
    echo "Options:"
    echo "  --run_dir         PATH   Run directory to evaluate (repeat for multiple)"
    echo "  --num_trials      N      Trials per task (default: 10)"
    echo "  --cuda_devices    X,Y,Z  Comma-separated GPU IDs (default: 0,1,2,3)"
    echo "  --server_timeout  N      Seconds to wait for policy server (default: 300)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Split GPU list into an array; number of workers = number of GPUs.
IFS=',' read -ra GPU_LIST <<< "$CUDA_DEVICES"
NUM_WORKERS=${#GPU_LIST[@]}

TOTAL=${#RUN_DIRS[@]}
echo "============================================================"
echo "Multi-run eval: $TOTAL run(s), $NUM_WORKERS parallel worker(s)"
echo "GPUs: ${GPU_LIST[*]}"
for d in "${RUN_DIRS[@]}"; do echo "  $d"; done
echo "============================================================"
echo ""

mkdir -p logs

# Temp dir for per-run exit status files.
STATUS_DIR=$(mktemp -d)
trap 'rm -rf "$STATUS_DIR"' EXIT

# One PID slot per GPU; -1 means the slot is free.
declare -a SLOT_PIDS
for ((s=0; s<NUM_WORKERS; s++)); do SLOT_PIDS[$s]=-1; done

# Block until a GPU slot is free, then echo its index.
find_free_slot() {
    while true; do
        for s in $(seq 0 $((NUM_WORKERS - 1))); do
            local pid="${SLOT_PIDS[$s]}"
            if [[ $pid -lt 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
                echo "$s"
                return
            fi
        done
        sleep 5
    done
}

for i in "${!RUN_DIRS[@]}"; do
    RUN_DIR="${RUN_DIRS[$i]}"
    RUN_NUM=$((i + 1))
    STATUS_FILE="$STATUS_DIR/run_${i}.status"

    SLOT=$(find_free_slot)
    GPU="${GPU_LIST[$SLOT]}"
    PORT_I=$((PORT + SLOT))
    LOG="logs/eval_$(basename "$RUN_DIR")_gpu${GPU}.out"

    echo ""
    echo "############################################################"
    echo "Run $RUN_NUM/$TOTAL  GPU=$GPU  port=$PORT_I"
    echo "  dir: $RUN_DIR"
    echo "  log: $LOG"
    echo "############################################################"

    (
        if bash "$SCRIPT_DIR/eval_run.sh" \
                --run_dir        "$RUN_DIR" \
                --num_trials     "$NUM_TRIALS" \
                --cuda_devices   "$GPU" \
                --server_timeout "$SERVER_TIMEOUT" \
                --host           "$HOST" \
                --port           "$PORT_I" \
                >> "$LOG" 2>&1; then
            echo "ok" > "$STATUS_FILE"
        else
            echo "fail" > "$STATUS_FILE"
            echo "[eval_multi] FAILED: $RUN_DIR  (see $LOG)" >&2
        fi
    ) &

    SLOT_PIDS[$SLOT]=$!
done

echo ""
echo "All $TOTAL job(s) launched — waiting for completion..."
wait || true   # collect all background jobs; failures tracked via status files

echo ""
echo "############################################################"
echo "All runs complete."
FAILED_RUNS=()
for i in "${!RUN_DIRS[@]}"; do
    STATUS_FILE="$STATUS_DIR/run_${i}.status"
    STATUS=$(cat "$STATUS_FILE" 2>/dev/null || echo "missing")
    if [[ "$STATUS" != "ok" ]]; then
        FAILED_RUNS+=("${RUN_DIRS[$i]}")
    fi
done

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo "WARNING: ${#FAILED_RUNS[@]} run(s) failed:"
    for r in "${FAILED_RUNS[@]}"; do echo "  - $r"; done
    exit 1
else
    echo "All runs succeeded."
fi
echo "############################################################"
