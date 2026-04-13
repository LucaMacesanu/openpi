#!/usr/bin/env bash
# submit_eval.sh — Wrapper that submits eval_run_parallel.sh with the job name
# set to eval_<run_name> (derived from --run_dir) for easy identification.
#
# Usage (same args as eval_run_parallel.sh):
#   bash shells/submit_eval.sh \
#       --run_dir checkpoints/sft_checkpoints/spatial_10_run_0 \
#       [--num_trials 10] \
#       [--cuda_devices 0]

set -euo pipefail

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
JOB_NAME="eval_${RUN_NAME}"
echo "Submitting job: $JOB_NAME"
sbatch --job-name="$JOB_NAME" sbatches/eval_run_parallel.sh "$@"
