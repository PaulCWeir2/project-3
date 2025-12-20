"""
Minimal Quick Test - Just verify imports and dataset access
Run this first before the full test
"""

print("Quick Test: Verifying basic setup...\n")

# Test imports
print("1. Testing imports...")
try:
    import torch
    import transformers
    import datasets
    import peft
    import trl
    print("   ✓ All packages imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("\n   Please run: uv sync")
    exit(1)

# Test dataset access
print("\n2. Testing dataset access...")
try:
    from datasets import load_dataset
    # Use streaming to avoid downloading the whole dataset
    dataset = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
    sample = next(iter(dataset))
    print(f"   ✓ Dataset accessible")
    print(f"   Sample fields: {list(sample.keys())}")
    print(f"   Instruction: {sample['instruction'][:60]}...")
except Exception as e:
    print(f"   ✗ Dataset access failed: {e}")
    exit(1)

# Test PyTorch
print("\n3. Testing PyTorch...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print("   ✓ PyTorch ready")

print("\n✓ QUICK TEST PASSED!")
print("\nRun 'python test_local.py' for comprehensive testing")
print("or upload to cluster and run 'sbatch run_exp1.sh' to test training")