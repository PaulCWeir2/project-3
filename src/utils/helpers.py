"""
Utility functions shared across tasks
"""
import os
import json
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd


def plot_training_curves(
        log_history: List[Dict],
        output_path: str,
        metrics: List[str] = ["loss", "eval_loss"]
):
    """
    Plot training and validation loss curves.

    Args:
        log_history: Training log history from trainer
        output_path: Path to save the plot
        metrics: Metrics to plot
    """
    # Extract metrics from log history
    data = {metric: {"steps": [], "values": []} for metric in metrics}

    for entry in log_history:
        for metric in metrics:
            if metric in entry:
                step = entry.get("step", entry.get("epoch", 0))
                data[metric]["steps"].append(step)
                data[metric]["values"].append(entry[metric])

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for metric in metrics:
        if data[metric]["steps"]:
            ax.plot(
                data[metric]["steps"],
                data[metric]["values"],
                label=metric.replace("_", " ").title(),
                marker='o' if len(data[metric]["steps"]) < 50 else None
            )

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to {output_path}")


def compare_experiments(
        results_dict: Dict[str, Dict],
        output_path: str,
        metric_name: str = "exact_match"
):
    """
    Create comparison table/plot for different experiments.

    Args:
        results_dict: Dictionary mapping experiment names to results
        output_path: Path to save comparison
        metric_name: Metric to compare
    """
    # Create DataFrame
    data = []
    for exp_name, results in results_dict.items():
        data.append({
            "Experiment": exp_name,
            metric_name.replace("_", " ").title(): results.get(metric_name, 0)
        })

    df = pd.DataFrame(data)

    # Save as CSV
    csv_path = output_path.replace(".png", ".csv")
    df.to_csv(csv_path, index=False)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["Experiment"], df[metric_name.replace("_", " ").title()])
    ax.set_xlabel("Experiment")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"Comparison of {metric_name.replace('_', ' ').title()}")
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison saved to {output_path} and {csv_path}")


def load_training_logs(log_dir: str) -> List[Dict]:
    """
    Load training logs from a directory.

    Args:
        log_dir: Directory containing training logs

    Returns:
        List of log entries
    """
    log_files = [f for f in os.listdir(log_dir) if f.startswith("trainer_state")]

    if not log_files:
        print(f"No training logs found in {log_dir}")
        return []

    # Load the most recent log file
    log_file = sorted(log_files)[-1]
    log_path = os.path.join(log_dir, log_file)

    with open(log_path, 'r') as f:
        state = json.load(f)

    return state.get("log_history", [])


def create_experiment_summary(
        experiments: Dict[str, str],
        output_path: str
):
    """
    Create a summary markdown file for all experiments.

    Args:
        experiments: Dictionary mapping experiment names to result paths
        output_path: Path to save summary
    """
    summary = "# Experiment Results Summary\n\n"

    for exp_name, result_path in experiments.items():
        summary += f"## {exp_name}\n\n"

        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                results = json.load(f)

            metrics = results.get("metrics", {})
            for metric, value in metrics.items():
                if isinstance(value, float):
                    summary += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
                else:
                    summary += f"- **{metric.replace('_', ' ').title()}**: {value}\n"
        else:
            summary += f"Results file not found: {result_path}\n"

        summary += "\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(summary)

    print(f"Experiment summary saved to {output_path}")