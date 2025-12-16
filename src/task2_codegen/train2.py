"""
Training script for Task 2: Code Generation with Mellum + LoRA
"""
import os
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from src.task2_codegen.config2 import CodeGenTrainingConfig  # ← Add src.task2_codegen.
from src.task2_codegen.preprocess2 import prepare_code_dataset  # ← Add src.task2_codegen.
import json


def setup_lora_model(config: CodeGenTrainingConfig):
    """
    Load base Mellum model and apply LoRA.

    Args:
        config: CodeGenTrainingConfig object

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        trust_remote_code=True
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    model = AutoModelForSeq2SeqLM.from_pretrained(  # ← Change from AutoModelForCausalLM
        config.model_name,
        cache_dir=config.cache_dir,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare for training
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


def train_code_generation_model(config: CodeGenTrainingConfig):
    """
    Main training function for code generation model.

    Args:
        config: CodeGenTrainingConfig object
    """
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # Setup model and tokenizer
    model, tokenizer = setup_lora_model(config)

    # Prepare dataset
    formatted_datasets = prepare_code_dataset(config, tokenizer)

    # SFT Configuration
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=config.fp16 and torch.cuda.is_available(),
        report_to=["tensorboard"],
        seed=config.seed,
        #max_seq_length=config.max_seq_length,
        dataset_text_field="text",  # The field containing formatted text
        packing=False,  # Don't pack multiple examples together
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=formatted_datasets["train"],
        eval_dataset=formatted_datasets["validation"],
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()

    # Save model
    print("Saving model...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save config
    with open(os.path.join(config.output_dir, "training_config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    print("Training complete!")
    return trainer, formatted_datasets


if __name__ == "__main__":
    # Example usage
    config = CodeGenTrainingConfig(
        output_dir="./outputs/task2_baseline",
        logging_dir="./logs/task2_baseline",
        num_train_epochs=3,
        lora_r=8,
    )

    trainer, datasets = train_code_generation_model(config)