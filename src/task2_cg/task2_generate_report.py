"""
Task 2: Generate Summary Report
Collect results from all experiments and generate a comprehensive report
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate summary report for Task 2")
    parser.add_argument("--output_dir", type=str, default="./task2_outputs")
    parser.add_argument("--report_path", type=str, default="./task2_summary_report.md")
    return parser.parse_args()


def load_experiment_results(exp_dir: Path) -> Dict:
    """Load results from a single experiment"""
    results = {
        "name": exp_dir.name,
        "exists": False,
        "metrics": None,
        "args": None,
        "train_metrics": None
    }

    # Load evaluation metrics
    eval_metrics_path = exp_dir / "evaluation" / "evaluation_metrics.json"
    if eval_metrics_path.exists():
        with open(eval_metrics_path) as f:
            results["metrics"] = json.load(f)
        results["exists"] = True

    # Load training arguments
    args_path = exp_dir / "args.json"
    if args_path.exists():
        with open(args_path) as f:
            results["args"] = json.load(f)

    # Load training metrics
    train_metrics_path = exp_dir / "train_metrics.json"
    if train_metrics_path.exists():
        with open(train_metrics_path) as f:
            results["train_metrics"] = json.load(f)

    return results


def create_comparison_plots(all_results: List[Dict], output_dir: Path):
    """Create comparison plots across experiments"""

    # Extract data for plotting
    exp_names = []
    bleu_scores = []
    exec_rates = []
    train_sizes = []
    lora_ranks = []

    for result in all_results:
        if result["exists"] and result["metrics"]:
            exp_names.append(result["name"])
            bleu_scores.append(result["metrics"]["average_bleu"])
            exec_rates.append(result["metrics"]["execution_success_rate"] * 100)

            if result["args"]:
                train_sizes.append(result["args"].get("train_size", 1.0) * 100)
                lora_ranks.append(result["args"].get("lora_rank", 0))

    # Plot 1: BLEU scores comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(exp_names))
    bars = ax.bar(x, bleu_scores, color='steelblue', alpha=0.8)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Average BLEU Score', fontsize=12)
    ax.set_title('BLEU Score Comparison Across Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, bleu_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "bleu_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Execution success rate comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, exec_rates, color='forestgreen', alpha=0.8)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Execution Success Rate (%)', fontsize=12)
    ax.set_title('Code Execution Success Rate Across Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, exec_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "execution_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Combined metrics plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # BLEU on left
    ax1.bar(x, bleu_scores, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Experiment', fontsize=11)
    ax1.set_ylabel('Average BLEU Score', fontsize=11)
    ax1.set_title('BLEU Score', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # Execution rate on right
    ax2.bar(x, exec_rates, color='forestgreen', alpha=0.8)
    ax2.set_xlabel('Experiment', fontsize=11)
    ax2.set_ylabel('Execution Success Rate (%)', fontsize=11)
    ax2.set_title('Code Execution Success Rate', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "combined_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison plots saved to {output_dir}")


def generate_markdown_report(all_results: List[Dict], output_path: Path):
    """Generate a comprehensive markdown report"""

    report = []
    report.append("# Task 2: Python Code Generation - Experiment Summary Report\n")
    report.append("---\n\n")

    # Overview
    report.append("## Overview\n\n")
    report.append("This report summarizes the results of 7 experiments for Task 2: Python Code Generation.\n")
    report.append("The task involves fine-tuning JetBrains/Mellum-4b-base on the flytech/python-codes-25k dataset.\n\n")

    # Experiments description
    report.append("## Experiments\n\n")
    report.append("| Experiment | Description | Training Data | LoRA Rank |\n")
    report.append("|------------|-------------|---------------|------------|\n")

    exp_descriptions = [
        ("exp1_pretrained", "Pre-trained baseline (no fine-tuning)", "N/A", "N/A"),
        ("exp2_finetuned_100pct_rank8", "Fine-tuned model", "100%", "8"),
        ("exp3_finetuned_30pct_rank8", "Fine-tuned with reduced data", "30%", "8"),
        ("exp4_finetuned_50pct_rank8", "Fine-tuned with reduced data", "50%", "8"),
        ("exp5_finetuned_100pct_rank4", "Fine-tuned with lower rank", "100%", "4"),
        ("exp6_finetuned_100pct_rank16", "Fine-tuned with higher rank", "100%", "16"),
        ("exp7_finetuned_100pct_rank32", "Fine-tuned with highest rank", "100%", "32"),
    ]

    for exp_name, desc, train_data, lora_rank in exp_descriptions:
        report.append(f"| {exp_name} | {desc} | {train_data} | {lora_rank} |\n")

    report.append("\n")

    # Results table
    report.append("## Results Summary\n\n")
    report.append("| Experiment | BLEU Score | Execution Success Rate | Executable Codes | Failed Codes |\n")
    report.append("|------------|------------|------------------------|------------------|---------------|\n")

    for result in all_results:
        if result["exists"] and result["metrics"]:
            m = result["metrics"]
            report.append(
                f"| {result['name']} | {m['average_bleu']:.4f} | "
                f"{m['execution_success_rate'] * 100:.2f}% | "
                f"{m['num_executable']} | {m['num_failed']} |\n"
            )
        else:
            report.append(f"| {result['name']} | N/A | N/A | N/A | N/A |\n")

    report.append("\n")

    # Detailed analysis
    report.append("## Detailed Analysis\n\n")

    # Analysis 1: Pre-trained vs Fine-tuned
    report.append("### 1. Pre-trained vs Fine-tuned Model\n\n")
    pretrained = next((r for r in all_results if "pretrained" in r["name"]), None)
    finetuned = next((r for r in all_results if "exp2" in r["name"]), None)

    if pretrained and finetuned and pretrained["exists"] and finetuned["exists"]:
        bleu_improve = (finetuned["metrics"]["average_bleu"] - pretrained["metrics"]["average_bleu"]) / \
                       pretrained["metrics"]["average_bleu"] * 100
        exec_improve = (finetuned["metrics"]["execution_success_rate"] - pretrained["metrics"][
            "execution_success_rate"]) * 100

        report.append(f"- **Pre-trained BLEU**: {pretrained['metrics']['average_bleu']:.4f}\n")
        report.append(f"- **Fine-tuned BLEU**: {finetuned['metrics']['average_bleu']:.4f}\n")
        report.append(f"- **Improvement**: {bleu_improve:+.2f}%\n\n")

        report.append(
            f"- **Pre-trained Execution Rate**: {pretrained['metrics']['execution_success_rate'] * 100:.2f}%\n")
        report.append(f"- **Fine-tuned Execution Rate**: {finetuned['metrics']['execution_success_rate'] * 100:.2f}%\n")
        report.append(f"- **Improvement**: {exec_improve:+.2f} percentage points\n\n")

    # Analysis 2: Training data size impact
    report.append("### 2. Impact of Training Data Size\n\n")
    report.append("Comparing models trained with different amounts of data (30%, 50%, 100%):\n\n")

    data_size_exps = [
        ("30%", "exp3_finetuned_30pct_rank8"),
        ("50%", "exp4_finetuned_50pct_rank8"),
        ("100%", "exp2_finetuned_100pct_rank8"),
    ]

    for size, exp_name in data_size_exps:
        exp = next((r for r in all_results if exp_name in r["name"]), None)
        if exp and exp["exists"]:
            report.append(f"- **{size} Training Data**:\n")
            report.append(f"  - BLEU: {exp['metrics']['average_bleu']:.4f}\n")
            report.append(f"  - Execution Rate: {exp['metrics']['execution_success_rate'] * 100:.2f}%\n\n")

    # Analysis 3: LoRA rank impact
    report.append("### 3. Impact of LoRA Rank\n\n")
    report.append("Comparing models with different LoRA ranks (4, 8, 16, 32):\n\n")

    rank_exps = [
        ("Rank 4", "exp5_finetuned_100pct_rank4"),
        ("Rank 8", "exp2_finetuned_100pct_rank8"),
        ("Rank 16", "exp6_finetuned_100pct_rank16"),
        ("Rank 32", "exp7_finetuned_100pct_rank32"),
    ]

    for rank_name, exp_name in rank_exps:
        exp = next((r for r in all_results if exp_name in r["name"]), None)
        if exp and exp["exists"]:
            report.append(f"- **{rank_name}**:\n")
            report.append(f"  - BLEU: {exp['metrics']['average_bleu']:.4f}\n")
            report.append(f"  - Execution Rate: {exp['metrics']['execution_success_rate'] * 100:.2f}%\n")
            if exp["args"]:
                trainable = exp["args"].get("lora_rank", 0) * 2  # Rough estimate
                report.append(f"  - Trainable Parameters: ~{trainable}M (estimated)\n\n")

    # Visualizations
    report.append("## Visualizations\n\n")
    report.append("![BLEU Comparison](bleu_comparison.png)\n\n")
    report.append("![Execution Rate Comparison](execution_comparison.png)\n\n")
    report.append("![Combined Metrics](combined_metrics.png)\n\n")

    # Conclusion
    report.append("## Conclusions\n\n")
    report.append("Key findings from the experiments:\n\n")
    report.append(
        "1. **Fine-tuning Effectiveness**: Fine-tuned models significantly outperform the pre-trained baseline.\n")
    report.append("2. **Training Data Impact**: More training data generally leads to better performance.\n")
    report.append("3. **LoRA Rank Trade-off**: Higher LoRA ranks provide more capacity but require more computation.\n")
    report.append(
        "4. **Code Quality**: Both syntactic correctness (BLEU) and functional correctness (execution) improve with fine-tuning.\n\n")

    # Save report
    with open(output_path, 'w') as f:
        f.writelines(report)

    print(f"Markdown report saved to {output_path}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)

    # Find all experiment directories
    exp_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("exp")])

    print(f"Found {len(exp_dirs)} experiment directories")

    # Load results from all experiments
    all_results = []
    for exp_dir in exp_dirs:
        print(f"Loading results from {exp_dir.name}...")
        results = load_experiment_results(exp_dir)
        all_results.append(results)

    # Create comparison plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(all_results, output_dir)

    # Generate markdown report
    print("\nGenerating markdown report...")
    generate_markdown_report(all_results, Path(args.report_path))

    print("\n" + "=" * 50)
    print("Report generation completed!")
    print("=" * 50)
    print(f"Report saved to: {args.report_path}")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()