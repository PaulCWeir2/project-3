#!/bin/bash
#SBATCH --job-name=task1_qa
#SBATCH --output=logs/task1_%j.out
#SBATCH --error=logs/task1_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load necessary modules (adjust based on your cluster)
# module load python/3.10
# module load cuda/11.8

# Activate virtual environment
source .venv/bin/activate

# Run training
python -m src.task1_qa.train1

# Run evaluation
python -m src.task1_qa.evaluate1

echo "Task 1 complete!"