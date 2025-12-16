"""
Run all Task 2 experiments
"""
import os
from config import Task2Config
from train import train_model
from evaluate import evaluate_model, save_results
from data_loader import prepare_dataset


def run_all_experiments():
    """
    Run all required experiments for Task 2:
    1. Pretrained baseline
    2. Various training sizes (30%, 50%, 100%)
    3. Various LoRA ranks (4, 8, 16)
    """

    print("\n" + "=" * 80)
    print("TASK 2: PYTHON CODE GENERATION EXPERIMENTS")
    print("=" * 80 + "\n")

    # Base config
    base_config = Task2Config()

    # Prepare dataset once (reuse for all experiments)
    print("Preparing dataset...")
    full_datasets = prepare_dataset(base_config)

    experiments = []

    # ===================================================================
    # EXPERIMENT 1: Pretrained Baseline (no fine-tuning)
    # ===================================================================
    experiments.append({
        'name': 'pretrained_baseline',
        'config': Task2Config(),
        'is_pretrained': True
    })

    # ===================================================================
    # EXPERIMENT 2-4: Training Size Variations (30%, 50%, 100%)
    # ===================================================================
    for size_frac in [0.3, 0.5, 1.0]:
        experiments.append({
            'name': f'size_{int(size_frac * 100)}',
            'config': Task2Config(
                train_size_fraction=size_frac,
                lora_r=8,
                lora_alpha=16
            ),
            'is_pretrained': False
        })

    # ===================================================================
    # EXPERIMENT 5-7: LoRA Rank Variations (4, 8, 16)
    # ===================================================================
    for rank in [4, 8, 16]:
        experiments.append({
            'name': f'lora_r{rank}',
            'config': Task2Config(
                train_size_fraction=1.0,
                lora_r=rank,
                lora_alpha=rank * 2
            ),
            'is_pretrained': False
        })

    # Run all experiments
    results_summary = []

    for i, exp in enumerate(experiments, 1):
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT {i}/{len(experiments)}: {exp['name']}")
        print(f"{'=' * 80}")

        config = exp['config']
        is_pretrained = exp['is_pretrained']

        # Set output directory
        config.output_dir = f"./outputs/{exp['name']}"

        # Train (or skip for pretrained)
        if is_pretrained:
            # Just prepare the base model for evaluation
            output_path = config.output_dir
            os.makedirs(output_path, exist_ok=True)

            print("\n🔵 Pretrained baseline - no training needed")

            # Evaluate pretrained model directly
            eval_output = evaluate_model(
                model_path=output_path,
                test_dataset=full_datasets['test'],
                config=config,
                is_pretrained=True
            )
        else:
            # Train model
            trainer, datasets = train_model(config, is_pretrained=False)

            # Evaluate trained model
            eval_output = evaluate_model(
                model_path=config.output_dir,
                test_dataset=datasets['test'],
                config=config,
                is_pretrained=False
            )

        # Save results
        results_path = os.path.join(config.output_dir, "evaluation_results.json")
        save_results(eval_output, results_path)

        # Add to summary
        results_summary.append({
            'experiment': exp['name'],
            'bleu': eval_output['metrics']['average_bleu'],
            'exec_rate': eval_output['metrics']['execution_success_rate']
        })

        print(f"\n✓ Experiment '{exp['name']}' complete!")

    # Print final summary
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<25} {'BLEU':<10} {'Exec Rate':<10}")
    print("-" * 80)
    for result in results_summary:
        print(f"{result['experiment']:<25} {result['bleu']:<10.4f} {result['exec_rate']:<10.2f}%")
    print("=" * 80 + "\n")

    # Save summary
    import json
    summary_path = "./outputs/experiments_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_all_experiments()