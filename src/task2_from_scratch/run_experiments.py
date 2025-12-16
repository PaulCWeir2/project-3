"""
Simple script to submit all 7 experiments as separate jobs
"""
import subprocess
import sys

experiments = [
    ("pretrained", 0.0, 0),    # No training
    ("size_30", 0.3, 8),
    ("size_50", 0.5, 8),
    ("size_100", 1.0, 8),
    ("lora_r4", 1.0, 4),
    ("lora_r8", 1.0, 8),
    ("lora_r16", 1.0, 16),
]

print("Submitting all experiments...")

for name, train_frac, lora_r in experiments:
    cmd = f'sbatch --job-name=t2_{name} --output=logs/{name}_%j.log --gres=gpu:1 --wrap="uv run python run_single_exp.py {name} {train_frac} {lora_r}"'
    subprocess.run(cmd, shell=True)
    print(f"✓ Submitted: {name}")

print("\nAll jobs submitted! Check with: squeue -u $USER")