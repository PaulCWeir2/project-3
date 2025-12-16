#!/bin/bash
#SBATCH --job-name=PA3_Final
#SBATCH --output=logs/exp_%A_%a.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-13

# 1. Setup Environment
export TMPDIR=$PWD/.uv_tmp
export UV_CACHE_DIR=$PWD/.uv_cache
export HF_HOME=$PWD/.hf_cache

# 2. Determine Task Type
if [ $SLURM_ARRAY_TASK_ID -lt 7 ]; then
    TASK="task1"
    T_IDX=$SLURM_ARRAY_TASK_ID
    TRAIN_SCRIPT="src.task1_qa.train1"
    EVAL_SCRIPT="src.task1_qa.evaluate1"
else
    TASK="task2"
    T_IDX=$((SLURM_ARRAY_TASK_ID - 7))
    TRAIN_SCRIPT="src.task2_codegen.train2"
    EVAL_SCRIPT="src.task2_codegen.evaluate2"
fi

# 3. Map T_IDX (0-6) to specific Experiment Params
# Default values
FRAC=1.0
RANK=8
IS_BASELINE=false

case $T_IDX in
    0) IS_BASELINE=true ;;
    1) FRAC=0.3 ;;
    2) FRAC=0.5 ;;
    3) FRAC=1.0 ;; # (Note: This is redundant with Rank=8, but completes the set)
    4) RANK=4 ;;
    5) RANK=8 ;; # (Note: This is redundant with Frac=1.0, but completes the set)
    6) RANK=16 ;;
esac

OUT_DIR="outputs/${TASK}_exp${T_IDX}_f${FRAC}_r${RANK}"

# 4. Execution
if [ "$IS_BASELINE" = true ]; then
    echo "Running $TASK Baseline (No Fine-tuning)"
    # For baseline, we skip training and evaluate the base model name directly
    uv run python -m $EVAL_SCRIPT --model_path "roberta-base" # Adjust if Task 2 needs Mellum
else
    echo "Running $TASK: Frac=$FRAC, Rank=$RANK"
    uv run python -m $TRAIN_SCRIPT --frac $FRAC --rank $RANK --output_dir $OUT_DIR
    uv run python -m $EVAL_SCRIPT --model_path $OUT_DIR
fi