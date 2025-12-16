"""
Training script for Task 1: Question Answering with RoBERTa + LoRA
"""
import os
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from peft import LoraConfig, get_peft_model, TaskType
from src.task1_qa.config1 import QATrainingConfig
from src.task1_qa.preprocess1 import prepare_qa_dataset
import json


def setup_lora_model(config: QATrainingConfig):
    """
    Load base RoBERTa model and apply LoRA.

    Args:
        config: QATrainingConfig object

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir
    )

    # Load base model for QA
    model = AutoModelForQuestionAnswering.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.QUESTION_ANS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


def train_qa_model(config: QATrainingConfig):
    """
    Main training function for QA model.

    Args:
        config: QATrainingConfig object
    """
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # Setup model and tokenizer
    model, tokenizer = setup_lora_model(config)

    # Prepare dataset
    tokenized_datasets = prepare_qa_dataset(config, tokenizer)

    # Data collator
    data_collator = DefaultDataCollator()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=config.fp16 and torch.cuda.is_available(),
        report_to=["tensorboard"],
        seed=config.seed,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
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
    return trainer, tokenized_datasets


if __name__ == "__main__":
    # Example usage
    config = QATrainingConfig(
        output_dir="./outputs/task1_baseline",
        logging_dir="./logs/task1_baseline",
        num_train_epochs=3,
        lora_r=8,
    )

    trainer, datasets = train_qa_model(config)