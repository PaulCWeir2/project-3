#!/bin/bash
#SBATCH --job-name=PA3_Final
#SBATCH --output=slurm_results/exp_%A_%a.out
#SBATCH --partition=regular
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-6

# Navigate to project root
cd $HOME/project-3

# Ensure uv is in the path for the compute node
export PATH="$HOME/.local/bin:$PATH"

case $SLURM_ARRAY_TASK_ID in
    0)
        uv run python src/main.py --mode baseline && \
        uv run python src/eval_task2.py "baseline" ;;
    1)
        uv run python src/main.py --train_pct 0.3 --lora_r 16 && \
        uv run python src/eval_task2.py "./results_r16_pct30/final_adapter" ;;
    2)
        uv run python src/main.py --train_pct 0.5 --lora_r 16 && \
        uv run python src/eval_task2.py "./results_r16_pct50/final_adapter" ;;
    3)
        uv run python src/main.py --train_pct 1.0 --lora_r 16 && \
        uv run python src/eval_task2.py "./results_r16_pct100/final_adapter" ;;
    4)
        uv run python src/main.py --train_pct 0.5 --lora_r 8 && \
        uv run python src/eval_task2.py "./results_r8_pct50/final_adapter" ;;
    5)
        uv run python src/main.py --train_pct 0.5 --lora_r 32 && \
        uv run python src/eval_task2.py "./results_r32_pct50/final_adapter" ;;
    6)
        uv run python src/main.py --train_pct 0.5 --lora_r 64 && \
        uv run python src/eval_task2.py "./results_r64_pct50/final_adapter" ;;
esac