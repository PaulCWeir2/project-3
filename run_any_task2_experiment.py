#!/usr/bin/env python3
"""
Run any Task 2 experiment with parameters.

Usage:
    python run_any_task2_experiment.py <exp_name> <train_size> <lora_r>

Examples:
    python run_any_task2_experiment.py pretrained 0 0        # No training, just eval
    python run_any_task2_experiment.py size_30 0.3 8         # 30% data
    python run_any_task2_experiment.py size_50 0.5 8         # 50% data
    python run_any_task2_experiment.py baseline 1.0 8        # 100% data, r=8
    python run_any_task2_experiment.py lora_r4 1.0 4         # 100% data, r=4
    python run_any_task2_experiment.py lora_r16 1.0 16       # 100% data, r=16
"""

import sys
import os

if len(sys.argv) != 4:
    print("Usage: python run_any_task2_experiment.py <exp_name> <train_size> <lora_r>")
    print("\nExamples:")
    print("  python run_any_task2_experiment.py pretrained 0 0")
    print("  python run_any_task2_experiment.py size_30 0.3 8")
    print("  python run_any_task2_experiment.py baseline 1.0 8")
    sys.exit(1)

exp_name = sys.argv[1]
train_size = float(sys.argv[2])
lora_r = int(sys.argv[3])

print("=" * 70)
print(f"TASK 2 EXPERIMENT: {exp_name}")
print("=" * 70)
print(f"Train size: {train_size * 100:.0f}%")
print(f"LoRA rank: {lora_r}")
print("=" * 70)

# Special case: pretrained baseline (no training)
if train_size == 0 and lora_r == 0:
    print("\nRunning PRETRAINED BASELINE evaluation (no fine-tuning)...")

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from datasets import load_dataset
    import torch
    from tqdm import tqdm
    import json

    model_name = "Salesforce/codet5-small"

    print(f"Loading pretrained model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    # Load test dataset (MBPP)
    print("Loading test dataset...")
    dataset = load_dataset("mbpp", "sanitized", split="test")


    # Helper functions
    def compute_exact_match(predictions, references):
        matches = sum(1 for pred, ref in zip(predictions, references)
                      if pred.strip() == ref.strip())
        return matches / len(predictions) if predictions else 0.0


    def compute_bleu(predictions, references):
        try:
            from sacrebleu import corpus_bleu
            bleu = corpus_bleu(predictions, [references])
            return bleu.score
        except ImportError:
            print("Warning: sacrebleu not installed, skipping BLEU score")
            return None


    # Generate predictions
    print(f"Generating predictions on {len(dataset)} test examples...")
    predictions = []
    references = []
    batch_size = 16

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]

        inputs = tokenizer(
            batch['text'],
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=5,
                early_stopping=True
            )

        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)
        references.extend(batch['code'])

    # Compute metrics
    print("\nComputing metrics...")
    exact_match = compute_exact_match(predictions, references)
    bleu = compute_bleu(predictions, references)

    results = {
        'model': model_name,
        'fine_tuned': False,
        'exact_match': exact_match * 100,
        'bleu': bleu,
        'num_examples': len(predictions)
    }

    print("\n" + "=" * 70)
    print("BASELINE RESULTS (No Fine-tuning)")
    print("=" * 70)
    print(f"Model: {results['model']}")
    print(f"Exact Match Accuracy: {results['exact_match']:.2f}%")
    if results['bleu'] is not None:
        print(f"BLEU Score: {results['bleu']:.2f}")
    print(f"Number of test examples: {results['num_examples']}")
    print("=" * 70)

    # Save results
    output_dir = f"outputs/task2_{exp_name}"
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save sample predictions
    samples_file = os.path.join(output_dir, "sample_predictions.json")
    samples = [
        {
            'prediction': pred,
            'reference': ref
        }
        for pred, ref in list(zip(predictions, references))[:10]
    ]
    with open(samples_file, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Sample predictions saved to: {samples_file}")

else:
    # Regular training experiment
    from src.task2_codegen.config2 import CodeGenTrainingConfig
    from src.task2_codegen.train2 import train_codegen_model

    config = CodeGenTrainingConfig(
        model_name="Salesforce/codet5-small",
        output_dir=f"./outputs/task2_{exp_name}",
        logging_dir=f"./logs/task2_{exp_name}",
        results_dir=f"./results/task2_{exp_name}",

        # Experiment parameters
        train_size_fraction=train_size,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.1,

        # Training settings
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,

        # Data settings
        dataset_name="mbpp",

        seed=42,
        fp16=True,
    )

    print(f"\nModel: {config.model_name}")
    print(f"LoRA: alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"Training: {config.num_train_epochs} epochs")
    print(f"Output: {config.output_dir}")
    print()

    # Train
    trainer, datasets = train_codegen_model(config)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT '{exp_name}' COMPLETE!")
    print(f"Model saved to: {config.output_dir}")
    print("=" * 70)