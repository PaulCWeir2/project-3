#!/usr/bin/env python3
"""
Utility script to check progress and status of Task 2 experiments
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Check Task 2 experiment status")
    parser.add_argument("--output_dir", type=str, default="./task2_outputs")
    return parser.parse_args()


def check_experiment_status(exp_dir: Path):
    """Check the status of a single experiment"""
    status = {
        "name": exp_dir.name,
        "exists": exp_dir.exists(),
        "training_complete": False,
        "evaluation_complete": False,
        "has_args": False,
        "has_metrics": False,
        "train_loss": None,
        "val_loss": None,
        "bleu_score": None,
        "exec_rate": None,
    }

    if not status["exists"]:
        return status

    # Check for training completion
    final_model_path = exp_dir / "final_model"
    if final_model_path.exists():
        status["training_complete"] = True

    # Check for evaluation completion
    eval_metrics_path = exp_dir / "evaluation" / "evaluation_metrics.json"
    if eval_metrics_path.exists():
        status["evaluation_complete"] = True
        with open(eval_metrics_path) as f:
            metrics = json.load(f)
            status["bleu_score"] = metrics.get("average_bleu")
            status["exec_rate"] = metrics.get("execution_success_rate")

    # Check for training args
    args_path = exp_dir / "args.json"
    if args_path.exists():
        status["has_args"] = True

    # Check for training metrics
    train_metrics_path = exp_dir / "train_metrics.json"
    if train_metrics_path.exists():
        status["has_metrics"] = True
        with open(train_metrics_path) as f:
            metrics = json.load(f)
            status["train_loss"] = metrics.get("train_loss")

    return status


def print_status_table(all_status):
    """Print a formatted table of experiment statuses"""
    print("\n" + "=" * 100)
    print("TASK 2 EXPERIMENT STATUS")
    print("=" * 100)
    print(f"{'Experiment':<30} {'Training':<12} {'Evaluation':<12} {'BLEU':<10} {'Exec Rate':<10}")
    print("-" * 100)

    for status in all_status:
        name = status["name"]

        # Training status
        if status["training_complete"]:
            train_status = "✓ Complete"
        elif status["exists"]:
            train_status = "⚠ In Progress"
        else:
            train_status = "✗ Not Started"

        # Evaluation status
        if status["evaluation_complete"]:
            eval_status = "✓ Complete"
        elif status["training_complete"]:
            eval_status = "⚠ Pending"
        else:
            eval_status = "✗ Not Started"

        # Metrics
        bleu = f"{status['bleu_score']:.4f}" if status['bleu_score'] is not None else "N/A"
        exec_rate = f"{status['exec_rate'] * 100:.2f}%" if status['exec_rate'] is not None else "N/A"

        print(f"{name:<30} {train_status:<12} {eval_status:<12} {bleu:<10} {exec_rate:<10}")

    print("=" * 100)

    # Summary
    total = len(all_status)
    completed_training = sum(1 for s in all_status if s["training_complete"])
    completed_eval = sum(1 for s in all_status if s["evaluation_complete"])

    print(f"\nSummary:")
    print(f"  Total experiments: {total}")
    print(f"  Training completed: {completed_training}/{total}")
    print(f"  Evaluation completed: {completed_eval}/{total}")
    print(f"  Overall progress: {(completed_eval / total) * 100:.1f}%")
    print()


def print_detailed_status(all_status):
    """Print detailed information for each experiment"""
    print("\n" + "=" * 100)
    print("DETAILED EXPERIMENT INFORMATION")
    print("=" * 100)

    for status in all_status:
        print(f"\n{'=' * 60}")
        print(f"Experiment: {status['name']}")
        print(f"{'=' * 60}")

        if not status["exists"]:
            print("  Status: Directory does not exist")
            continue

        print(f"  Training: {'✓ Complete' if status['training_complete'] else '✗ Incomplete'}")
        print(f"  Evaluation: {'✓ Complete' if status['evaluation_complete'] else '✗ Incomplete'}")

        if status["bleu_score"] is not None:
            print(f"  BLEU Score: {status['bleu_score']:.4f}")

        if status["exec_rate"] is not None:
            print(f"  Execution Success Rate: {status['exec_rate'] * 100:.2f}%")

        if status["train_loss"] is not None:
            print(f"  Final Training Loss: {status['train_loss']:.4f}")

    print("\n" + "=" * 100 + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        sys.exit(1)

    # Expected experiments
    expected_experiments = [
        "exp1_pretrained",
        "exp2_finetuned_100pct_rank8",
        "exp3_finetuned_30pct_rank8",
        "exp4_finetuned_50pct_rank8",
        "exp5_finetuned_100pct_rank4",
        "exp6_finetuned_100pct_rank16",
        "exp7_finetuned_100pct_rank32",
    ]

    # Check status of all experiments
    all_status = []
    for exp_name in expected_experiments:
        exp_dir = output_dir / exp_name
        status = check_experiment_status(exp_dir)
        all_status.append(status)

    # Print summary table
    print_status_table(all_status)

    # Print detailed information
    print("\nFor detailed information, the following files contain results:")
    print("  - <exp_name>/args.json: Training arguments")
    print("  - <exp_name>/train_metrics.json: Training metrics")
    print("  - <exp_name>/loss_curves.png: Training/validation curves")
    print("  - <exp_name>/evaluation/evaluation_metrics.json: Evaluation results")
    print("  - <exp_name>/evaluation/predictions_detailed.json: Detailed predictions")
    print()

    # Check if all experiments are complete
    all_complete = all(s["evaluation_complete"] for s in all_status)
    if all_complete:
        print("✓ All experiments complete! You can now generate the summary report:")
        print(f"  python task2_generate_report.py --output_dir {args.output_dir}")
        print()
    else:
        incomplete = [s["name"] for s in all_status if not s["evaluation_complete"]]
        print("⚠ The following experiments are not yet complete:")
        for exp in incomplete:
            print(f"  - {exp}")
        print()


if __name__ == "__main__":
    main()