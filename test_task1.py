"""
Quick test of Task 1 pipeline with tiny dataset
"""
from src.task1_qa.config1 import QATrainingConfig
from src.task1_qa.preprocess1 import prepare_qa_dataset
from src.task1_qa.train1 import setup_lora_model
from transformers import AutoTokenizer

# Create a minimal config for testing
config = QATrainingConfig(
    model_name="roberta-base",
    output_dir="./outputs/task1_test",
    logging_dir="./logs/task1_test",
    results_dir="./results/task1_test",

    # Make it tiny for testing
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=10,
    eval_steps=50,
    save_steps=50,

    # Use small fraction for testing
    train_size_fraction=0.01,  # Only 1% of data

    # LoRA config
    lora_r=8,
    lora_alpha=16,

    seed=42,
    fp16=False,  # Set to False for CPU testing
)

print("Step 1: Testing model loading...")
model, tokenizer = setup_lora_model(config)
print("✓ Model loaded successfully!")

print("\nStep 2: Testing data preprocessing...")
dataset = prepare_qa_dataset(config, tokenizer)
print(f"✓ Dataset prepared!")
print(f"  Train: {len(dataset['train'])} examples")
print(f"  Val: {len(dataset['validation'])} examples")
print(f"  Test: {len(dataset['test'])} examples")

print("\nStep 3: Inspecting a sample...")
sample = dataset['train'][0]
print(f"  Keys: {sample.keys()}")
print(f"  Input shape: {len(sample['input_ids'])}")
print(f"  Start pos: {sample['start_positions']}")
print(f"  End pos: {sample['end_positions']}")

print("\n✓ All components working! Ready for full training.")