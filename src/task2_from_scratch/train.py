"""
Training script for Task 2
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer
from config import Task2Config
from data_loader import prepare_dataset
import json


def setup_model_and_tokenizer(config: Task2Config):
    """
    Load model, tokenizer, and apply LoRA.

    Returns:
        (model, tokenizer)
    """
    print(f"\n{'=' * 60}")
    print(f"Loading model: {config.model_name}")
    print(f"{'=' * 60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        cache_dir=config.cache_dir
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Set pad_token = eos_token")

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.use_fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=config.cache_dir
    )

    # Disable cache for training
    model.config.use_cache = False

    # Configure LoRA
    print(f"\nApplying LoRA (r={config.lora_r}, alpha={config.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj"],  # Standard attention modules
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train_model(config: Task2Config, is_pretrained_baseline: bool = False):
    """
    Main training function.

    Args:
        config: Task2Config object
        is_pretrained_baseline: If True, skip training and just save the base model

    Returns:
        (trainer, datasets)
    """
    experiment_name = config.get_experiment_name()
    output_path = os.path.join(config.output_dir, experiment_name)

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")

    # Create directories
    os.makedirs(output_path, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load dataset
    print(f"\n{'=' * 60}")
    print("LOADING DATASET")
    print(f"{'=' * 60}")
    datasets = prepare_dataset(config, tokenizer)

    # CRITICAL CHECK
    print(f"\n{'=' * 60}")
    print("PRE-TRAINING VERIFICATION")
    print(f"{'=' * 60}")
    print(f"✓ Train examples: {len(datasets['train'])}")
    print(f"✓ Val examples: {len(datasets['validation'])}")
    print(f"✓ Test examples: {len(datasets['test'])}")

    if len(datasets['train']) == 0:
        raise ValueError("❌ ERROR: Training dataset is EMPTY!")

    # Calculate expected training steps
    steps_per_epoch = len(datasets['train']) // (config.batch_size * config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_epochs
    print(f"✓ Expected training steps: {total_steps}")
    print(f"  ({steps_per_epoch} steps/epoch × {config.num_epochs} epochs)")
    print(f"{'=' * 60}")

    if is_pretrained_baseline:
        print("\n🔵 PRETRAINED BASELINE: Skipping training, saving base model")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        return None, datasets

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_path,

        # Training
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,

        # Logging & Evaluation
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Performance
        fp16=config.use_fp16 and torch.cuda.is_available(),

        # SFT specific
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,

        # Other
        report_to=["tensorboard"],
        seed=config.seed,
    )

    # Initialize trainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        tokenizer=tokenizer,
    )

    # Train
    print(f"\n{'=' * 60}")
    print("🚀 STARTING TRAINING")
    print(f"{'=' * 60}\n")

    train_result = trainer.train()

    print(f"\n{'=' * 60}")
    print("✓ TRAINING COMPLETE")
    print(f"{'=' * 60}")

    # Save model
    print("\nSaving model...")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save config
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2, default=str)

    print(f"✓ Model saved to: {output_path}")

    return trainer, datasets


if __name__ == "__main__":
    # Test training
    config = Task2Config(
        num_epochs=1,  # Quick test
        train_size_fraction=0.1,  # Small subset for testing
    )

    trainer, datasets = train_model(config)
    print("\n✓ Training test successful!")