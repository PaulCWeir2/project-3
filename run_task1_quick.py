"""
Quick training test - 1 epoch on 5% of data
Use this to verify training works before full run
"""
from src.task1_qa.config1 import QATrainingConfig
from src.task1_qa.train1 import train_qa_model

config = QATrainingConfig(
    model_name="roberta-base",
    output_dir="./outputs/task1_quick",
    logging_dir="./logs/task1_quick",
    results_dir="./results/task1_quick",

    # Quick test settings
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=20,
    eval_steps=100,
    save_steps=100,

    # Only 5% of data for quick test
    train_size_fraction=0.05,

    # LoRA
    lora_r=8,
    lora_alpha=16,

    seed=42,
    fp16=False,  # CPU compatible
)

print("Starting quick training test...")
print(f"This will train on ~{int(130000 * 0.7 * 0.05)} examples")

trainer, datasets = train_qa_model(config)

print("\n✓ Quick training complete!")
print(f"Model saved to: {config.output_dir}")