#!/bin/bash
#SBATCH --job-name=task2_experiments
#SBATCH --output=logs/task2_exp_%A_%a.out
#SBATCH --error=logs/task2_exp_%A_%a.err
#SBATCH --array=1-7
#SBATCH --time=12:00:00
#SBATCH --partition=regular
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Single SBATCH script that runs all 7 experiments using array jobs
# Each experiment runs as a separate job with SLURM_ARRAY_TASK_ID

# Common parameters
MODEL_NAME="JetBrains/Mellum-4b-base"
DATASET_NAME="flytech/python-codes-25k"
CACHE_DIR="./cache"
OUTPUT_DIR="./task2_outputs"
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=16
LR=2e-4
MAX_SEQ_LEN=1024

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "SLURM Array Job: Experiment ${SLURM_ARRAY_TASK_ID}/7"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Configure experiment based on array task ID
case ${SLURM_ARRAY_TASK_ID} in
    1)
        # Experiment 1: Pre-trained baseline (no training, just evaluation)
        EXP_NAME="exp1_pretrained"
        TRAIN_SIZE=0.01
        EPOCHS=1
        USE_LORA=""
        LORA_RANK=8
        LORA_ALPHA=16
        IS_BASELINE=true
        echo "Experiment 1: Pre-trained Baseline (no fine-tuning)"
        ;;
    2)
        # Experiment 2: Fine-tuned 100% data, LoRA rank 8
        EXP_NAME="exp2_finetuned_100pct_rank8"
        TRAIN_SIZE=1.0
        EPOCHS=${NUM_EPOCHS}
        USE_LORA="--use_lora"
        LORA_RANK=8
        LORA_ALPHA=16
        IS_BASELINE=false
        echo "Experiment 2: Fine-tuned (100% data, LoRA rank 8)"
        ;;
    3)
        # Experiment 3: Fine-tuned 30% data, LoRA rank 8
        EXP_NAME="exp3_finetuned_30pct_rank8"
        TRAIN_SIZE=0.3
        EPOCHS=${NUM_EPOCHS}
        USE_LORA="--use_lora"
        LORA_RANK=8
        LORA_ALPHA=16
        IS_BASELINE=false
        echo "Experiment 3: Fine-tuned (30% data, LoRA rank 8)"
        ;;
    4)
        # Experiment 4: Fine-tuned 50% data, LoRA rank 8
        EXP_NAME="exp4_finetuned_50pct_rank8"
        TRAIN_SIZE=0.5
        EPOCHS=${NUM_EPOCHS}
        USE_LORA="--use_lora"
        LORA_RANK=8
        LORA_ALPHA=16
        IS_BASELINE=false
        echo "Experiment 4: Fine-tuned (50% data, LoRA rank 8)"
        ;;
    5)
        # Experiment 5: Fine-tuned 100% data, LoRA rank 4
        EXP_NAME="exp5_finetuned_100pct_rank4"
        TRAIN_SIZE=1.0
        EPOCHS=${NUM_EPOCHS}
        USE_LORA="--use_lora"
        LORA_RANK=4
        LORA_ALPHA=8
        IS_BASELINE=false
        echo "Experiment 5: Fine-tuned (100% data, LoRA rank 4)"
        ;;
    6)
        # Experiment 6: Fine-tuned 100% data, LoRA rank 16
        EXP_NAME="exp6_finetuned_100pct_rank16"
        TRAIN_SIZE=1.0
        EPOCHS=${NUM_EPOCHS}
        USE_LORA="--use_lora"
        LORA_RANK=16
        LORA_ALPHA=32
        IS_BASELINE=false
        echo "Experiment 6: Fine-tuned (100% data, LoRA rank 16)"
        ;;
    7)
        # Experiment 7: Fine-tuned 100% data, LoRA rank 32
        EXP_NAME="exp7_finetuned_100pct_rank32"
        TRAIN_SIZE=1.0
        EPOCHS=${NUM_EPOCHS}
        USE_LORA="--use_lora"
        LORA_RANK=32
        LORA_ALPHA=64
        IS_BASELINE=false
        echo "Experiment 7: Fine-tuned (100% data, LoRA rank 32)"
        ;;
esac

echo ""
echo "Configuration:"
echo "  Experiment Name: ${EXP_NAME}"
echo "  Training Size: ${TRAIN_SIZE}"
echo "  LoRA Rank: ${LORA_RANK}"
echo "  Output Directory: ${OUTPUT_DIR}/${EXP_NAME}"
echo ""

# Training
if [ "$IS_BASELINE" = true ]; then
    echo "Running minimal training to create test dataset..."
    uv run python task2_train.py \
        --model_name ${MODEL_NAME} \
        --dataset_name ${DATASET_NAME} \
        --cache_dir ${CACHE_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --experiment_name ${EXP_NAME} \
        --train_size ${TRAIN_SIZE} \
        --num_train_epochs ${EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LR} \
        --max_seq_length ${MAX_SEQ_LEN} \
        --logging_steps 10 \
        --save_steps 1000000 \
        --eval_steps 1000000
else
    echo "Starting training..."
    uv run python task2_train.py \
        --model_name ${MODEL_NAME} \
        --dataset_name ${DATASET_NAME} \
        --cache_dir ${CACHE_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --experiment_name ${EXP_NAME} \
        --train_size ${TRAIN_SIZE} \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --num_train_epochs ${EPOCHS} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LR} \
        --max_seq_length ${MAX_SEQ_LEN} \
        ${USE_LORA}
fi

# Evaluation
echo ""
echo "Starting evaluation..."

if [ "$IS_BASELINE" = true ]; then
    # Evaluate pretrained model
    uv run python task2_eval.py \
        --model_path ${MODEL_NAME} \
        --base_model ${MODEL_NAME} \
        --test_dataset_path "${OUTPUT_DIR}/${EXP_NAME}/test_dataset.json" \
        --output_dir "${OUTPUT_DIR}/${EXP_NAME}/evaluation" \
        --cache_dir ${CACHE_DIR} \
        --is_pretrained \
        --num_samples 100
else
    # Evaluate fine-tuned model
    uv run python task2_eval.py \
        --model_path "${OUTPUT_DIR}/${EXP_NAME}/final_model" \
        --base_model ${MODEL_NAME} \
        --test_dataset_path "${OUTPUT_DIR}/${EXP_NAME}/test_dataset.json" \
        --output_dir "${OUTPUT_DIR}/${EXP_NAME}/evaluation" \
        --cache_dir ${CACHE_DIR}
fi

echo ""
echo "=========================================="
echo "Experiment ${SLURM_ARRAY_TASK_ID} completed!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  ${OUTPUT_DIR}/${EXP_NAME}/"
echo ""