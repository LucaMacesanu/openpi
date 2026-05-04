#!/bin/bash
#SBATCH --job-name=eval_multi
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_eval_multi_run.out
#SBATCH --account=torch_pr_50_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lim2045@nyu.edu

# ---------------------------------------------------------------------------
# Usage:
#   sbatch sbatches/eval_multi_run.sh \
#       --run_dir checkpoints/sft_checkpoints/moe_run_0 \
#       --run_dir checkpoints/sft_checkpoints/sft_run_1 \
#       [--num_trials 10] \
#       [--cuda_devices 0,1,2,3]
#
# Runs up to 4 eval_run.sh workers in parallel (one per GPU).
# ---------------------------------------------------------------------------

cd /local_data/lim2045/openpi
source .venv/bin/activate

export WANDB_API_KEY=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

bash shells/eval_multi_run.sh "$@"
