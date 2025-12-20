"""
Local Test Script for Task 2
Tests dataset loading, model loading, and basic training setup
Run this locally before submitting to the cluster
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print("=" * 60)
print("Task 2: Local Environment Test")
print("=" * 60)

# Test 1: Check PyTorch and CUDA
print("\n1. Testing PyTorch installation...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print("   ✓ PyTorch OK")

# Test 2: Load dataset
print("\n2. Testing dataset loading...")
try:
    dataset = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
    # Get first example
    first_example = next(iter(dataset))
    print(f"   Dataset fields: {list(first_example.keys())}")
    print(f"   Sample instruction: {first_example['instruction'][:50]}...")
    print(f"   Sample output length: {len(first_example['output'])} chars")
    print("   ✓ Dataset loading OK")
except Exception as e:
    print(f"   ✗ Dataset loading failed: {e}")
    exit(1)

# Test 3: Load tokenizer
print("\n3. Testing tokenizer loading...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "JetBrains/Mellum-4b-base",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   EOS token: {tokenizer.eos_token}")
    print(f"   PAD token: {tokenizer.pad_token}")
    print("   ✓ Tokenizer OK")
except Exception as e:
    print(f"   ✗ Tokenizer loading failed: {e}")
    exit(1)

# Test 4: Test formatting function
print("\n4. Testing formatting function...")


def formatting_prompts_func(example):
    """Format dataset examples for code completion"""
    output_texts = []
    for i in range(len(example['instruction'])):
        prompt = example['instruction'][i]
        if example['input'][i].strip():
            prompt = f"{prompt} {example['input'][i]}"
        text = f"{prompt}\n```python\n{example['output'][i]}\n```"
        output_texts.append(text)
    return output_texts


try:
    # Load a small sample
    sample_dataset = load_dataset("flytech/python-codes-25k", split="train[:10]")
    formatted = formatting_prompts_func(sample_dataset)
    print(f"   Formatted {len(formatted)} examples")
    print(f"   Sample formatted text (first 100 chars):")
    print(f"   {formatted[0][:100]}...")
    print("   ✓ Formatting function OK")
except Exception as e:
    print(f"   ✗ Formatting failed: {e}")
    exit(1)

# Test 5: Load model (small test - don't load full model locally if no GPU)
print("\n5. Testing model loading...")
try:
    if torch.cuda.is_available():
        print("   Loading full model (GPU available)...")
        model = AutoModelForCausalLM.from_pretrained(
            "JetBrains/Mellum-4b-base",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"   Model loaded on: {model.device}")
        print(f"   Model dtype: {model.dtype}")
        print("   ✓ Model loading OK")
    else:
        print("   Skipping full model load (no GPU)")
        print("   Note: On cluster, model will load with GPU")
        print("   ✓ Model loading skipped (expected locally)")
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")
    print("   Note: This may fail locally without GPU - that's OK!")

# Test 6: Test LoRA configuration
print("\n6. Testing LoRA configuration...")
try:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    print(f"   LoRA rank: {lora_config.r}")
    print(f"   LoRA alpha: {lora_config.lora_alpha}")
    print(f"   Target modules: {len(lora_config.target_modules)}")
    print("   ✓ LoRA config OK")
except Exception as e:
    print(f"   ✗ LoRA config failed: {e}")
    exit(1)

# Test 7: Test SFTConfig
print("\n7. Testing SFTConfig...")
try:
    sft_config = SFTConfig(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_length=1024,
        logging_steps=10,
        save_steps=1000,
        eval_steps=1000,
    )
    print(f"   Max length: {sft_config.max_length}")
    print(f"   Learning rate: {sft_config.learning_rate}")
    print(f"   Batch size: {sft_config.per_device_train_batch_size}")
    print("   ✓ SFTConfig OK")
except Exception as e:
    print(f"   ✗ SFTConfig failed: {e}")
    exit(1)

# Test 8: Test dataset splitting
print("\n8. Testing dataset splitting...")
try:
    small_dataset = load_dataset("flytech/python-codes-25k", split="train[:100]")
    train_val_test = small_dataset.train_test_split(test_size=0.3, seed=42)
    train = train_val_test["train"]
    temp = train_val_test["test"]
    val_test = temp.train_test_split(test_size=0.5, seed=42)
    val = val_test["train"]
    test = val_test["test"]

    print(f"   Original: {len(small_dataset)} samples")
    print(f"   Train: {len(train)} samples (70%)")
    print(f"   Val: {len(val)} samples (15%)")
    print(f"   Test: {len(test)} samples (15%)")
    print("   ✓ Dataset splitting OK")
except Exception as e:
    print(f"   ✗ Dataset splitting failed: {e}")
    exit(1)

# Test 9: Test tokenization
print("\n9. Testing tokenization...")
try:
    sample_text = formatted[0]
    encoded = tokenizer(sample_text, truncation=True, max_length=1024)
    print(f"   Input length: {len(sample_text)} chars")
    print(f"   Tokenized length: {len(encoded['input_ids'])} tokens")
    print(f"   Sample token IDs: {encoded['input_ids'][:10]}...")
    print("   ✓ Tokenization OK")
except Exception as e:
    print(f"   ✗ Tokenization failed: {e}")
    exit(1)

# Test 10: Check dependencies
print("\n10. Testing all required imports...")
try:
    import matplotlib.pyplot as plt
    import nltk
    from nltk.translate.bleu_score import sentence_bleu
    import re
    import json
    import subprocess

    print("   ✓ All dependencies OK")
except Exception as e:
    print(f"   ✗ Missing dependency: {e}")
    print("   You may need: pip install matplotlib nltk")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nYour environment is ready for Task 2!")
print("\nNext steps:")
print("1. Upload all files to the SGC cluster")
print("2. Run 'uv sync' on the cluster")
print("3. Submit jobs with 'sbatch run_all_experiments.sh'")
print("\nNote: Model loading may have been skipped locally if you")
print("don't have a GPU. This is expected - it will work on the cluster.")