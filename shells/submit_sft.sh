#!/usr/bin/env bash
# submit_sft.sh — Wrapper that submits train_sft_ordered.sh with the job name
# set to sft_train_<exp_name> so runs are easy to identify in squeue/sacct.
#
# Usage (same args as run_sft_ordered.sh):
#   bash shells/submit_sft.sh \
#       --tasks_files notes/tasks_libero_spatial.txt \
#       --exp_name spatial_10_run_0 \
#       [--steps_per_task 10000] \
#       [--resume true]

set -euo pipefail

EXP_NAME="sft_ordered_run"

# Extract --exp_name from the argument list without consuming it.
args=("$@")
for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "--exp_name" ]]; then
        EXP_NAME="${args[$((i + 1))]}"
        break
    fi
done

JOB_NAME="sft_train_${EXP_NAME}"
echo "Submitting job: $JOB_NAME"
sbatch --job-name="$JOB_NAME" sbatches/train_sft_ordered.sh "$@"
