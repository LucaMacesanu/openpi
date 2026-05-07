#!/usr/bin/env bash
# eval_sft_local.sh - Local wrapper for SFT eval runs.
#
# Mirrors shells/submit_eval.sh argument handling, but launches the eval flow
# directly on the current machine instead of submitting a Slurm job.
#
# Usage (same args as eval_run_parallel.sh / submit_eval.sh):
#   bash shells/eval_sft_local.sh \
#       --run_dir checkpoints/sft_checkpoints/spatial_10_run_0 \
#       [--num_trials 10] \
#       [--cuda_devices 0] \
#       [--max_parallel 1]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_DIR=""

# Extract --run_dir from the argument list without consuming it.
args=("$@")
for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "--run_dir" ]]; then
        RUN_DIR="${args[$((i + 1))]}"
        break
    fi
done

if [[ -z "$RUN_DIR" ]]; then
    echo "Error: --run_dir is required."
    echo "Usage: $0 --run_dir <path> [eval_run_parallel.sh options...]"
    exit 1
fi

RUN_NAME="$(basename "$RUN_DIR")"
RUN_LABEL="eval_${RUN_NAME}"
echo "Launching local eval: $RUN_LABEL"

bash "$SCRIPT_DIR/eval_run_parallel.sh" "$@"
