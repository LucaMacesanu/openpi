#!/bin/bash
#SBATCH --job-name=eval_run_parallel
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%j_eval_run_parallel.out
#SBATCH --account=torch_pr_50_tandon_advanced
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lim2045@nyu.edu

# ---------------------------------------------------------------------------
# Usage:
#   sbatch sbatches/eval_run_parallel.sh \
#       --run_dir checkpoints/sft_checkpoints/huihan_run_0 \
#       [--num_trials 10] \
#       [--cuda_devices 0]
#
# Each checkpoint gets its own policy server on a unique port:
#   PORT = BASE_PORT + checkpoint_index
# where BASE_PORT = 8000 + (SLURM_JOB_ID % 50000) % 10000
# ---------------------------------------------------------------------------

cd /scratch/lim2045/singularity
singularity exec --nv /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash

cd /scratch/lim2045/openpi
source .venv/bin/activate

export WANDB_API_KEY=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

# Unique base port for this job; individual checkpoints add their index on top.
BASE_PORT=$(( 8000 + (SLURM_JOB_ID % 50000) % 10000 ))
echo "[sbatch] Base port $BASE_PORT (SLURM_JOB_ID=$SLURM_JOB_ID)"

bash shells/eval_run_parallel.sh --base_port "$BASE_PORT" --server_timeout 900 "$@"
