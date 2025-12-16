#!/bin/bash
#SBATCH --job-name=cs231_pa3_all
#SBATCH --output=logs/exp_%A_%a.log
#SBATCH --error=logs/exp_%A_%a.err
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

# Create logs directory
mkdir -p logs

# Setup environment variables for disk space issues
export TMPDIR=$PWD/.uv_tmp
export UV_CACHE_DIR=$PWD/.uv_cache
export HF_HOME=$PWD/.hf_cache
mkdir -p $TMPDIR $UV_CACHE_DIR $HF_HOME

# Activate virtual environment
source .venv/bin/activate

echo "=============================================="
echo "Running experiment array task: $SLURM_ARRAY_TASK_ID"
echo "=============================================="

# Map array index to experiment
case $SLURM_ARRAY_TASK_ID in
    # TASK 1 Experiments (0-5)
    0)
        echo "Task 1: Pretrained Baseline (no training)"
        python run_any_task1_experiment.py pretrained 0 0
        ;;
    1)
        echo "Task 1: 30% data, LoRA r=8"
        python run_any_task1_experiment.py size_30 0.3 8
        ;;
    2)
        echo "Task 1: 50% data, LoRA r=8"
        python run_any_task1_experiment.py size_50 0.5 8
        ;;
    3)
        echo "Task 1: 100% data, LoRA r=8 (BASELINE)"
        python run_any_task1_experiment.py baseline 1.0 8
        ;;
    4)
        echo "Task 1: 100% data, LoRA r=4"
        python run_any_task1_experiment.py lora_r4 1.0 4
        ;;
    5)
        echo "Task 1: 100% data, LoRA r=16"
        python run_any_task1_experiment.py lora_r16 1.0 16
        ;;

    # TASK 2 Experiments (6-11)
    6)
        echo "Task 2: Pretrained Baseline (no training)"
        python run_any_task2_experiment.py pretrained 0 0
        ;;
    7)
        echo "Task 2: 30% data, LoRA r=8"
        python run_any_task2_experiment.py size_30 0.3 8
        ;;
    8)
        echo "Task 2: 50% data, LoRA r=8"
        python run_any_task2_experiment.py size_50 0.5 8
        ;;
    9)
        echo "Task 2: 100% data, LoRA r=8 (BASELINE)"
        python run_any_task2_experiment.py baseline 1.0 8
        ;;
    10)
        echo "Task 2: 100% data, LoRA r=4"
        python run_any_task2_experiment.py lora_r4 1.0 4
        ;;
    11)
        echo "Task 2: 100% data, LoRA r=16"
        python run_any_task2_experiment.py lora_r16 1.0 16
        ;;
    *)
        echo "Invalid array task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "=============================================="
echo "Experiment $SLURM_ARRAY_TASK_ID complete!"
echo "=============================================="