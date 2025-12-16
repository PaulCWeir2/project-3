"""
Utility functions for Task 2
"""
import os
import matplotlib.pyplot as plt
import json
from typing import List, Dict


def plot_training_curves(log_dir: str, output_path: str):
    """
    Plot training and validation loss curves from tensorboard logs.
    (Requires tensorboard package)
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()

        # Get training loss
        train_loss = [(s.step, s.value) for s in ea.Scalars('train/loss')]
        eval_loss = [(s.step, s.value) for s in ea.Scalars('eval/loss')]

        plt.figure(figsize=(10, 6))

        if train_loss:
            steps, losses = zip(*train_loss)
            plt.plot(steps, losses, label='Training Loss')

        if eval_loss:
            steps, losses = zip(*eval_loss)
            plt.plot(steps, losses, label='Validation Loss')

        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)

        plt.savefig(output_path)
        print(f"✓ Training curve saved to: {output_path}")

    except Exception as e:
        print(f"Could not plot training curves: {e}")


def compare_experiments(results_dir: str = "./outputs"):
    """
    Create comparison plots for all experiments.
    """
    experiments = []

    # Collect all results
    for exp_dir in os.listdir(results_dir):
        results_file = os.path.join(results_dir, exp_dir, "evaluation_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
                experiments.append({
                    'name': exp_dir,
                    'bleu': data['metrics']['average_bleu'],
                    'exec_rate': data['metrics']['execution_success_rate']
                })

    if not experiments:
        print("No results found!")
        return

    # Sort by name
    experiments.sort(key=lambda x: x['name'])

    # Plot comparisons
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    names = [e['name'] for e in experiments]
    bleus = [e['bleu'] for e in experiments]
    exec_rates = [e['exec_rate'] for e in experiments]

    # BLEU scores
    ax1.bar(range(len(names)), bleus)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('BLEU Score')
    ax1.set_title('BLEU Score Comparison')
    ax1.grid(axis='y')

    # Execution rates
    ax2.bar(range(len(names)), exec_rates)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Execution Success Rate (%)')
    ax2.set_title('Execution Success Rate Comparison')
    ax2.grid(axis='y')

    plt.tight_layout()
    output_path = os.path.join(results_dir, "experiments_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")


if __name__ == "__main__":
    # Generate comparison plots
    compare_experiments()