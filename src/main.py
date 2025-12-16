import argparse
import math
import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pct", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--mode", type=str, choices=["train", "baseline"], default="train")
    return parser.parse_args()


def run():
    args = get_args()
    model_id = "JetBrains/Mellum-4b-base"

    # 1. Dataset Loading & Manual Split (80/10/10)
    dataset = load_dataset("flytech/python-codes-25k", split='train')
    ds_split = dataset.train_test_split(test_size=0.2, seed=42)
    test_val = ds_split['test'].train_test_split(test_size=0.5, seed=42)

    train_ds = ds_split['train']
    if args.train_pct < 1.0:
        train_ds = train_ds.select(range(int(len(train_ds) * args.train_pct)))

    val_ds = test_val['train']  # For validation loss curves
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Baseline Mode (No Training)
    if args.mode == "baseline":
        print("--- RUNNING BASELINE EVALUATION ---")
        # Just create the directory so the eval script knows where to look
        os.makedirs("./results_baseline", exist_ok=True)
        return

    # 3. Fine-tuning Mode
    print(f"--- TRAINING: Size {args.train_pct * 100}%, Rank {args.lora_r} ---")
    output_dir = f"./results_r{args.lora_r}_pct{int(args.train_pct * 100)}"

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_seq_length=512,
        eval_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        logging_steps=10,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        tokenizer=tokenizer,
        formatting_func=lambda x: [f"### Instruction: {i}\n### Response: {o}{tokenizer.eos_token}" for i, o in
                                   zip(x['instruction'], x['output'])]
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_adapter"))


if __name__ == "__main__":
    run()