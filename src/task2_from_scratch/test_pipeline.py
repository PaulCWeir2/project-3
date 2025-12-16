"""
Quick test to verify the entire pipeline works
"""
from config import Task2Config
from train import train_model
from evaluate import evaluate_model, save_results


def quick_test():
    """Run a quick test with minimal data"""
    print("\n" + "=" * 60)
    print("QUICK PIPELINE TEST")
    print("=" * 60 + "\n")

    # Very small config for testing
    config = Task2Config(
        num_epochs=1,
        train_size_fraction=0.01,  # Only 1% of data
        batch_size=2,
        output_dir="./outputs/test_run"
    )

    # Train
    print("1. Testing training...")
    trainer, datasets = train_model(config)

    # Evaluate
    print("\n2. Testing evaluation...")
    eval_output = evaluate_model(
        model_path=config.output_dir,
        test_dataset=datasets['test'].select(range(10)),  # Only 10 examples
        config=config,
        is_pretrained=False
    )

    # Save
    print("\n3. Testing result saving...")
    save_results(eval_output, os.path.join(config.output_dir, "test_results.json"))

    print("\n" + "=" * 60)
    print("✓ PIPELINE TEST SUCCESSFUL!")
    print("=" * 60)


if __name__ == "__main__":
    quick_test()