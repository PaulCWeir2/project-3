#!/bin/bash
#SBATCH --job-name=task2_all_experiments
#SBATCH --output=logs/task2_all_%j.out
#SBATCH --error=logs/task2_all_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Master script to run all 7 experiments for Task 2

# Create logs directory
mkdir -p logs

# Set common parameters
MODEL_NAME="JetBrains/Mellum-4b-base"
DATASET_NAME="flytech/python-codes-25k"
CACHE_DIR="./cache"
OUTPUT_DIR="./task2_outputs"
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
LR=2e-4
MAX_SEQ_LEN=1024

echo "=========================================="
echo "Starting Task 2: All 7 Experiments"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# ==========================================
# Experiment 1: Pre-trained Model (Baseline)
# ==========================================
echo "=========================================="
echo "Experiment 1: Pre-trained Model (Baseline)"
echo "=========================================="

EXP_NAME="exp1_pretrained"
EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"

# Create a dummy training to get test dataset (we won't actually train)
python task2_train.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --cache_dir ${CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name ${EXP_NAME} \
    --train_size 0.01 \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --logging_steps 10 \
    --save_steps 1000000 \
    --eval_steps 1000000

# Evaluate pretrained model
python task2_eval.py \
    --model_path ${MODEL_NAME} \
    --base_model ${MODEL_NAME} \
    --test_dataset_path "${EXP_DIR}/test_dataset.json" \
    --output_dir "${EXP_DIR}/evaluation" \
    --cache_dir ${CACHE_DIR} \
    --is_pretrained \
    --num_samples 100

echo "Experiment 1 completed!"
echo ""

# ==========================================
# Experiment 2: Fine-tuned (100% data, LoRA rank=8)
# ==========================================
echo "=========================================="
echo "Experiment 2: Fine-tuned (100% data, LoRA rank=8)"
echo "=========================================="

EXP_NAME="exp2_finetuned_100pct_rank8"
EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"

python task2_train.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --cache_dir ${CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name ${EXP_NAME} \
    --train_size 1.0 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --use_lora

python task2_eval.py \
    --model_path "${EXP_DIR}/final_model" \
    --base_model ${MODEL_NAME} \
    --test_dataset_path "${EXP_DIR}/test_dataset.json" \
    --output_dir "${EXP_DIR}/evaluation" \
    --cache_dir ${CACHE_DIR}

echo "Experiment 2 completed!"
echo ""

# ==========================================
# Experiment 3: Fine-tuned (30% data, LoRA rank=8)
# ==========================================
echo "=========================================="
echo "Experiment 3: Fine-tuned (30% data, LoRA rank=8)"
echo "=========================================="

EXP_NAME="exp3_finetuned_30pct_rank8"
EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"

python task2_train.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --cache_dir ${CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name ${EXP_NAME} \
    --train_size 0.3 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --use_lora

python task2_eval.py \
    --model_path "${EXP_DIR}/final_model" \
    --base_model ${MODEL_NAME} \
    --test_dataset_path "${EXP_DIR}/test_dataset.json" \
    --output_dir "${EXP_DIR}/evaluation" \
    --cache_dir ${CACHE_DIR}

echo "Experiment 3 completed!"
echo ""

# ==========================================
# Experiment 4: Fine-tuned (50% data, LoRA rank=8)
# ==========================================
echo "=========================================="
echo "Experiment 4: Fine-tuned (50% data, LoRA rank=8)"
echo "=========================================="

EXP_NAME="exp4_finetuned_50pct_rank8"
EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"

python task2_train.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --cache_dir ${CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name ${EXP_NAME} \
    --train_size 0.5 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --use_lora

python task2_eval.py \
    --model_path "${EXP_DIR}/final_model" \
    --base_model ${MODEL_NAME} \
    --test_dataset_path "${EXP_DIR}/test_dataset.json" \
    --output_dir "${EXP_DIR}/evaluation" \
    --cache_dir ${CACHE_DIR}

echo "Experiment 4 completed!"
echo ""

# ==========================================
# Experiment 5: Fine-tuned (100% data, LoRA rank=4)
# ==========================================
echo "=========================================="
echo "Experiment 5: Fine-tuned (100% data, LoRA rank=4)"
echo "=========================================="

EXP_NAME="exp5_finetuned_100pct_rank4"
EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"

python task2_train.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --cache_dir ${CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name ${EXP_NAME} \
    --train_size 1.0 \
    --lora_rank 4 \
    --lora_alpha 8 \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --use_lora

python task2_eval.py \
    --model_path "${EXP_DIR}/final_model" \
    --base_model ${MODEL_NAME} \
    --test_dataset_path "${EXP_DIR}/test_dataset.json" \
    --output_dir "${EXP_DIR}/evaluation" \
    --cache_dir ${CACHE_DIR}

echo "Experiment 5 completed!"
echo ""

# ==========================================
# Experiment 6: Fine-tuned (100% data, LoRA rank=16)
# ==========================================
echo "=========================================="
echo "Experiment 6: Fine-tuned (100% data, LoRA rank=16)"
echo "=========================================="

EXP_NAME="exp6_finetuned_100pct_rank16"
EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"

python task2_train.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --cache_dir ${CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name ${EXP_NAME} \
    --train_size 1.0 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --use_lora

python task2_eval.py \
    --model_path "${EXP_DIR}/final_model" \
    --base_model ${MODEL_NAME} \
    --test_dataset_path "${EXP_DIR}/test_dataset.json" \
    --output_dir "${EXP_DIR}/evaluation" \
    --cache_dir ${CACHE_DIR}

echo "Experiment 6 completed!"
echo ""

# ==========================================
# Experiment 7: Fine-tuned (100% data, LoRA rank=32)
# ==========================================
echo "=========================================="
echo "Experiment 7: Fine-tuned (100% data, LoRA rank=32)"
echo "=========================================="

EXP_NAME="exp7_finetuned_100pct_rank32"
EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"

python task2_train.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --cache_dir ${CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name ${EXP_NAME} \
    --train_size 1.0 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --use_lora

python task2_eval.py \
    --model_path "${EXP_DIR}/final_model" \
    --base_model ${MODEL_NAME} \
    --test_dataset_path "${EXP_DIR}/test_dataset.json" \
    --output_dir "${EXP_DIR}/evaluation" \
    --cache_dir ${CACHE_DIR}

echo "Experiment 7 completed!"
echo ""

# ==========================================
# Generate Summary Report
# ==========================================
echo "=========================================="
echo "Generating Summary Report"
echo "=========================================="

python task2_generate_report.py \
    --output_dir ${OUTPUT_DIR} \
    --report_path "${OUTPUT_DIR}/task2_summary_report.md"

echo ""
echo "=========================================="
echo "All 7 Experiments Completed!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results are saved in: ${OUTPUT_DIR}"
echo "Summary report: ${OUTPUT_DIR}/task2_summary_report.md"