"""
Run a single experiment with given parameters
Usage: python run_single_exp.py <name> <train_frac> <lora_r>
"""
import sys
from config import Task2Config
from train import train_model
from evaluate import evaluate_model, save_results
from data_loader import prepare_dataset

if len(sys.argv) != 4:
    print("Usage: python run_single_exp.py <name> <train_frac> <lora_r>")
    sys.exit(1)

name = sys.argv[1]
train_frac = float(sys.argv[2])
lora_r = int(sys.argv[3])

print(f"\n{'='*60}")
print(f"EXPERIMENT: {name}")
print(f"{'='*60}")

config = Task2Config(
    train_size_fraction=train_frac,
    lora_r=lora_r,
    lora_alpha=lora_r * 2,
    output_dir=f"./outputs/{name}"
)

# Special case: pretrained baseline (no training)
if train_frac == 0.0:
    print("Running pretrained baseline (no training)")
    datasets = prepare_dataset(config)
    eval_output = evaluate_model(
        model_path=config.output_dir,
        test_dataset=datasets['test'],
        config=config,
        is_pretrained=True
    )
else:
    # Normal training + evaluation
    trainer, datasets = train_model(config)
    eval_output = evaluate_model(
        model_path=config.output_dir,
        test_dataset=datasets['test'],
        config=config,
        is_peft=True
    )

# Save results
save_results(eval_output, f"{config.output_dir}/results.json")

print(f"\n✓ {name} complete!")
print(f"✓ BLEU: {eval_output['metrics']['average_bleu']:.4f}")
print(f"✓ Exec Rate: {eval_output['metrics']['execution_success_rate']:.2f}%")