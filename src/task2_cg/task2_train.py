"""
Task 2: Python Code Generation Training Script
Fine-tune JetBrains/Mellum-4b-base on python-codes-25k dataset using LoRA
"""

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2: Python Code Generation")
    parser.add_argument("--model_name", type=str, default="JetBrains/Mellum-4b-base")
    parser.add_argument("--dataset_name", type=str, default="flytech/python-codes-25k")
    parser.add_argument("--output_dir", type=str, default="./task2_outputs")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--train_size", type=float, default=1.0, help="Fraction of training data to use")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    parser.add_argument("--experiment_name", type=str, default="baseline", help="Name of the experiment")

    return parser.parse_args()


def load_and_prepare_dataset(dataset_name, cache_dir, train_size=1.0, seed=42):
    """Load and split the dataset into train/val/test"""
    print(f"Loading dataset: {dataset_name}")

    # Load the full dataset
    dataset = load_dataset(dataset_name, cache_dir=cache_dir, split="train")

    print(f"Total dataset size: {len(dataset)}")

    # Split into train (70%), val (15%), test (15%)
    train_val_test = dataset.train_test_split(test_size=0.3, seed=seed)
    train_dataset = train_val_test["train"]
    temp_dataset = train_val_test["test"]

    val_test = temp_dataset.train_test_split(test_size=0.5, seed=seed)
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]

    # If train_size < 1.0, reduce training data
    if train_size < 1.0:
        original_size = len(train_dataset)
        train_dataset = train_dataset.train_test_split(
            train_size=train_size,
            seed=seed
        )["train"]
        print(f"Reduced training data from {original_size} to {len(train_dataset)} ({train_size * 100}%)")

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def formatting_prompts_func(example):
    """
    Format the dataset examples for code completion
    The dataset has 'instruction', 'input', and 'output' fields
    Format: instruction + input (if present) followed by the Python code output
    """
    output_texts = []
    for i in range(len(example['instruction'])):
        # Combine instruction and input (input is often empty or very short)
        prompt = example['instruction'][i]
        if example['input'][i].strip():  # Only add input if it's not empty
            prompt = f"{prompt} {example['input'][i]}"

        # Format as: prompt followed by code block
        text = f"{prompt}\n```python\n{example['output'][i]}\n```"
        output_texts.append(text)
    return output_texts


def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directory
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(experiment_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    train_dataset, val_dataset, test_dataset = load_and_prepare_dataset(
        args.dataset_name,
        args.cache_dir,
        train_size=args.train_size,
        seed=args.seed
    )

    # Save test dataset for later evaluation
    test_dataset.to_json(experiment_dir / "test_dataset.json")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Apply LoRA if requested
    if args.use_lora:
        print(f"Applying LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}")

        # Prepare model for training
        model = prepare_model_for_kbit_training(model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        # Get PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(experiment_dir / "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        logging_dir=str(experiment_dir / "logs"),
        report_to="none",
        seed=args.seed,
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        max_seq_length=args.max_seq_length,
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()

    # Save the final model
    print("Saving model...")
    trainer.save_model(str(experiment_dir / "final_model"))
    tokenizer.save_pretrained(str(experiment_dir / "final_model"))

    # Save training metrics
    metrics = train_result.metrics
    with open(experiment_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot training curves
    plot_training_curves(trainer, experiment_dir)

    print(f"Training completed! Results saved to {experiment_dir}")


def plot_training_curves(trainer, output_dir):
    """Plot and save training/validation loss curves"""
    log_history = trainer.state.log_history

    train_losses = []
    eval_losses = []
    train_steps = []
    eval_steps = []

    for entry in log_history:
        if "loss" in entry:
            train_losses.append(entry["loss"])
            train_steps.append(entry["step"])
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])
            eval_steps.append(entry["step"])

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label="Training Loss", marker='o')
    plt.plot(eval_steps, eval_losses, label="Validation Loss", marker='s')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Loss curves saved to {output_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()