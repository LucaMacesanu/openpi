#!/bin/bash
#SBATCH --job-name=compute_norm_stats
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j_norm_stats.out
#SBATCH --account=torch_pr_50_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lim2045@nyu.edu

# ---------------------------------------------------------------------------
# Computes normalization statistics for a given config and saves them to
# the config's assets directory. Only needs to be run once per config.
#
# SFT training uses norm stats from --norm_stats_from (default:
# pi0_libero_low_mem_finetune), so run this with that config name first.
#
# Usage:
#   sbatch sbatches/compute_norm_stats.sh
#   sbatch sbatches/compute_norm_stats.sh --config_name pi0_libero_low_mem_finetune
# ---------------------------------------------------------------------------

CONFIG_NAME="pi0_libero_low_mem_finetune"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config_name) CONFIG_NAME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Computing norm stats for config: $CONFIG_NAME"

cd /scratch/lim2045/singularity
singularity exec /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash

cd /scratch/lim2045/openpi
source .venv/bin/activate

export WANDB_API_KEY=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

uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
