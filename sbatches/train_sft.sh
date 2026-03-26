#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --partition=h200_tandon,h100_tandon
#SBATCH --constraint="h100|h200"
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_train_sft.out
#SBATCH --account=torch_pr_50_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lim2045@nyu.edu

# ---------------------------------------------------------------------------
# Usage:
#   sbatch sbatches/train_sft.sh \
#       --num_tasks 10 \
#       --exp_name sft_run_0 \
#       [--steps_per_task 5000] \
#       [--checkpoint_dir /scratch/lim2045/openpi/checkpoints/sft_checkpoints] \
#       [--task_seed 42] \
#       [--batch_size 32] \
#       [--wandb_enabled true] \
#       [--config_name pi05_libero_sft] \
#       [--norm_stats_from pi0_libero_low_mem_finetune]
# ---------------------------------------------------------------------------

cd /scratch/lim2045/singularity
singularity exec --nv /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash

cd /scratch/lim2045/openpi
source .venv/bin/activate

export WANDB_API_KEY=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

bash shells/run_sft.sh "$@"
