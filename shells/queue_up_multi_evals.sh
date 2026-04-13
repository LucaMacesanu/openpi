sbatch sbatches/eval_run_parallel.sh --run_dir checkpoints/sft_checkpoints/huihan_aefft_only --force
sbatch sbatches/eval_run_parallel.sh --run_dir checkpoints/sft_checkpoints/huihan_fast_0 --force
sbatch sbatches/eval_run_parallel.sh --run_dir checkpoints/sft_checkpoints/ae_fft_only_0/ --force
sbatch sbatches/eval_run_parallel.sh --run_dir checkpoints/sft_checkpoints/moe_full_learning_hyperparms --force
sbatch sbatches/eval_run_parallel.sh --run_dir checkpoints/sft_checkpoints/convergence_moe_0 --force
sbatch sbatches/eval_run_parallel.sh --run_dir checkpoints/sft_checkpoints/convergence_huihan_0 --force
sbatch sbatches/eval_run_parallel.sh --run_dir checkpoints/sft_checkpoints/a100_test --force
