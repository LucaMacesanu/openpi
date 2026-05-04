#!/usr/bin/env bash
# run_sft_local.sh — Local wrapper for ordered SFT runs.
#
# Mirrors shells/submit_sft.sh argument handling, but launches the training
# flow directly on the current machine instead of submitting a Slurm job.
#
# Usage (same args as run_sft_ordered.sh / submit_sft.sh):
#   bash shells/run_sft_local.sh \
#       --tasks_files notes/tasks_libero_spatial.txt \
#       --exp_name spatial_run_0 \
#       [--steps_per_task 5000] \
#       [--config_name pi05_libero_sft] \
#       [--checkpoint_interval 1000] \
#       [--resume true]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_NAME="sft_ordered_run"

# Extract --exp_name from the argument list without consuming it.
args=("$@")
for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "--exp_name" ]]; then
        EXP_NAME="${args[$((i + 1))]}"
        break
    fi
done

RUN_NAME="sft_train_${EXP_NAME}"
echo "Launching local run: $RUN_NAME"

bash "$SCRIPT_DIR/run_sft_ordered.sh" "$@"
