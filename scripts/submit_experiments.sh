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
# FIX: Add the src folders to the Python Path so imports work naturally
export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/src/task1_qa:$PWD/src/task2_codegen

# 2. Determine Task Type
if [ $SLURM_ARRAY_TASK_ID -lt 7 ]; then
    TASK="task1"
    T_IDX=$SLURM_ARRAY_TASK_ID
    TRAIN_SCRIPT="src.task1_qa.train1"
    EVAL_SCRIPT="src.task1_qa.evaluate1"
    BASE_MODEL="roberta-base"
else
    TASK="task2"
    T_IDX=$((SLURM_ARRAY_TASK_ID - 7))
    TRAIN_SCRIPT="src.task2_codegen.train2"
    EVAL_SCRIPT="src.task2_codegen.evaluate2"
    # Update this path if your Task 2 base model is stored elsewhere
    BASE_MODEL="mellum"
fi

# 3. Map T_IDX (0-6) to specific Experiment Params
FRAC=1.0
RANK=8
IS_BASELINE=false

case $T_IDX in
    0) IS_BASELINE=true; RANK=0 ;; # Setting rank 0 for baseline folder naming
    1) FRAC=0.3 ;;
    2) FRAC=0.5 ;;
    3) FRAC=1.0 ;;
    4) RANK=4 ;;
    5) RANK=8 ;;
    6) RANK=16 ;;
esac

# Create a unique output directory for every experiment
OUT_DIR="outputs/${TASK}_exp${T_IDX}_f${FRAC}_r${RANK}"
mkdir -p "$OUT_DIR"

# 4. Execution
if [ "$IS_BASELINE" = true ]; then
    echo "Running $TASK Baseline (No Fine-tuning) using $BASE_MODEL"
    # Skip training, evaluate the raw pre-trained model directly
    # The output of this will now be saved in its own exp0 folder
    uv run python -m $EVAL_SCRIPT --model_path "$BASE_MODEL"
else
    echo "Running $TASK: Frac=$FRAC, Rank=$RANK"
    echo "Saving results to: $OUT_DIR"

    # Run training and save to the unique directory
    uv run python -m $TRAIN_SCRIPT --frac $FRAC --rank $RANK --output_dir "$OUT_DIR"

    # Run evaluation on the weights we just saved in OUT_DIR
    uv run python -m $EVAL_SCRIPT --model_path "$OUT_DIR"
fi