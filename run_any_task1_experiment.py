#!/usr/bin/env python3
"""
Run any Task 1 experiment with parameters.

Usage:
    python run_any_task1_experiment.py <exp_name> <train_size> <lora_r>

Examples:
    python run_any_task1_experiment.py pretrained 0 0        # No training, just eval
    python run_any_task1_experiment.py size_30 0.3 8         # 30% data
    python run_any_task1_experiment.py size_50 0.5 8         # 50% data
    python run_any_task1_experiment.py baseline 1.0 8        # 100% data, r=8
    python run_any_task1_experiment.py lora_r4 1.0 4         # 100% data, r=4
    python run_any_task1_experiment.py lora_r16 1.0 16       # 100% data, r=16
"""

import sys
import os

if len(sys.argv) != 4:
    print("Usage: python run_any_task1_experiment.py <exp_name> <train_size> <lora_r>")
    print("\nExamples:")
    print("  python run_any_task1_experiment.py pretrained 0 0")
    print("  python run_any_task1_experiment.py size_30 0.3 8")
    print("  python run_any_task1_experiment.py baseline 1.0 8")
    sys.exit(1)

exp_name = sys.argv[1]
train_size = float(sys.argv[2])
lora_r = int(sys.argv[3])

print("=" * 70)
print(f"TASK 1 EXPERIMENT: {exp_name}")
print("=" * 70)
print(f"Train size: {train_size * 100:.0f}%")
print(f"LoRA rank: {lora_r}")
print("=" * 70)

# Special case: pretrained baseline (no training)
if train_size == 0 and lora_r == 0:
    print("\nRunning PRETRAINED BASELINE evaluation (no fine-tuning)...")

    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    from src.task1_qa.evaluate1 import evaluate_qa_model, save_evaluation_results
    from src.task1_qa.preprocess1 import prepare_qa_dataset
    from src.task1_qa.config1 import QATrainingConfig
    import torch

    # Load pretrained model
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare test dataset
    config = QATrainingConfig(model_name=model_name)
    from src.task1_qa.preprocess1 import load_and_split_squad

    dataset = load_and_split_squad(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    # Tokenize test set
    from src.task1_qa.preprocess1 import preprocess_squad_examples

    test_dataset = dataset["test"].map(
        lambda examples: preprocess_squad_examples(
            examples,
            tokenizer,
            max_length=384,
            doc_stride=128,
            is_training=False
        ),
        batched=True,
        remove_columns=dataset["test"].column_names,
        desc="Tokenizing test set"
    )

    # Load pretrained model (no LoRA)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Evaluate
    print(f"\nEvaluating pretrained {model_name} on {len(test_dataset)} examples...")

    from src.task1_qa.evaluate1 import (
        get_answer_from_logits,
        exact_match_score
    )
    from tqdm import tqdm
    import numpy as np

    predictions = []
    ground_truths = []
    exact_matches = []

    with torch.no_grad():
        for example in tqdm(test_dataset, desc="Evaluating"):
            input_ids = torch.tensor([example["input_ids"]]).to(device)
            attention_mask = torch.tensor([example["attention_mask"]]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            pred_answer = get_answer_from_logits(
                outputs.start_logits[0],
                outputs.end_logits[0],
                input_ids[0],
                tokenizer
            )

            start_pos = example["start_positions"]
            end_pos = example["end_positions"]

            if start_pos == 0 and end_pos == 0:
                true_answer = ""
            else:
                true_tokens = example["input_ids"][start_pos:end_pos + 1]
                true_answer = tokenizer.decode(true_tokens, skip_special_tokens=True)

            predictions.append(pred_answer)
            ground_truths.append(true_answer)

            em_score = exact_match_score(pred_answer, true_answer)
            exact_matches.append(em_score)

    avg_exact_match = np.mean(exact_matches) * 100

    results = {
        "exact_match": avg_exact_match,
        "num_examples": len(test_dataset),
        "predictions_sample": predictions[:10],
        "ground_truths_sample": ground_truths[:10],
    }

    print(f"\nEvaluation Results:")
    print(f"Exact Match: {avg_exact_match:.2f}%")
    print(f"Number of examples: {len(test_dataset)}")

    # Save results
    output_dir = f"outputs/task1_{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    save_evaluation_results(
        results,
        predictions,
        ground_truths,
        os.path.join(output_dir, "evaluation_results.json")
    )

    print(f"\nResults saved to: {output_dir}/evaluation_results.json")

else:
    # Regular training experiment
    from src.task1_qa.config1 import QATrainingConfig
    from src.task1_qa.train1 import train_qa_model

    config = QATrainingConfig(
        model_name="roberta-base",
        output_dir=f"./outputs/task1_{exp_name}",
        logging_dir=f"./logs/task1_{exp_name}",
        results_dir=f"./results/task1_{exp_name}",

        # Experiment parameters
        train_size_fraction=train_size,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.1,

        # Training settings
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=100,
        eval_steps=500,
        save_steps=500,

        # Data settings
        max_length=384,
        doc_stride=128,

        seed=42,
        fp16=True,
    )

    print(f"\nModel: {config.model_name}")
    print(f"LoRA: alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"Training: {config.num_train_epochs} epochs")
    print(f"Output: {config.output_dir}")
    print()

    # Train
    trainer, datasets = train_qa_model(config)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE! Starting evaluation...")
    print("=" * 70)

    # Evaluate the trained model
    from src.task1_qa.evaluate1 import evaluate_qa_model, save_evaluation_results
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    print(f"\nEvaluating model on {len(datasets['test'])} test examples...")
    results, preds, truths = evaluate_qa_model(
        model_path=config.output_dir,
        test_dataset=datasets['test'],
        tokenizer=tokenizer,
        is_peft=True
    )

    # Save evaluation results
    save_evaluation_results(
        results,
        preds,
        truths,
        os.path.join(config.output_dir, "evaluation_results.json")
    )

    print("\n" + "=" * 70)
    print(f"EXPERIMENT '{exp_name}' COMPLETE!")
    print(f"Model saved to: {config.output_dir}")
    print(f"Exact Match: {results['exact_match']:.2f}%")
    print("=" * 70)