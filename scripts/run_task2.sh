#!/bin/bash
#SBATCH --job-name=task2_codegen
#SBATCH --output=logs/task2_%j.out
#SBATCH --error=logs/task2_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Load necessary modules (adjust based on your cluster)
# module load python/3.10
# module load cuda/11.8

# Activate virtual environment
source .venv/bin/activate

# Run training
python -m src.task2_codegen.train2

# Run evaluation
python -m src.task2_codegen.evaluate2

echo "Task 2 complete!"