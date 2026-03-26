#!/usr/bin/env bash
# eval_multi_run.sh — Evaluate multiple SFT run folders sequentially.
#
# Runs eval_run.sh for each --run_dir in order, so you can hold a node
# and batch through several experiments without re-queuing.
#
# Usage:
#   bash shells/eval_multi_run.sh \
#       --run_dir checkpoints/sft_checkpoints/moe_run_0 \
#       --run_dir checkpoints/sft_checkpoints/sft_run_1 \
#       [--num_trials 10] \
#       [--cuda_devices 0] \
#       [--server_timeout 300]

set -euo pipefail

RUN_DIRS=()
NUM_TRIALS=10
CUDA_DEVICES="0"
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
    echo "  --cuda_devices    X      CUDA_VISIBLE_DEVICES (default: 0)"
    echo "  --server_timeout  N      Seconds to wait for policy server (default: 300)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOTAL=${#RUN_DIRS[@]}
echo "============================================================"
echo "Multi-run eval: $TOTAL run(s) to evaluate"
for d in "${RUN_DIRS[@]}"; do echo "  $d"; done
echo "============================================================"
echo ""

FAILED_RUNS=()

for i in "${!RUN_DIRS[@]}"; do
    RUN_DIR="${RUN_DIRS[$i]}"
    RUN_NUM=$((i + 1))
    echo ""
    echo "############################################################"
    echo "Run $RUN_NUM / $TOTAL: $RUN_DIR"
    echo "############################################################"

    bash "$SCRIPT_DIR/eval_run.sh" \
        --run_dir        "$RUN_DIR" \
        --num_trials     "$NUM_TRIALS" \
        --cuda_devices   "$CUDA_DEVICES" \
        --server_timeout "$SERVER_TIMEOUT" \
        --host           "$HOST" \
        --port           "$PORT" \
        || { echo "ERROR: eval_run.sh failed for $RUN_DIR"; FAILED_RUNS+=("$RUN_DIR"); }
done

echo ""
echo "############################################################"
echo "All runs complete."
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo "WARNING: ${#FAILED_RUNS[@]} run(s) failed:"
    for r in "${FAILED_RUNS[@]}"; do echo "  - $r"; done
fi
echo "############################################################"
