"""
Data loading and preprocessing for Task 2
"""
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Dict, Any
import os


def load_and_split_dataset(config) -> DatasetDict:
    """
    Load the python-codes dataset and split into train/val/test.

    Returns:
        DatasetDict with 'train', 'validation', 'test' splits
    """
    print(f"Loading dataset: {config.dataset_name}")

    # Load full dataset
    dataset = load_dataset(
        config.dataset_name,
        cache_dir=config.cache_dir,
        split="train"
    )

    print(f"Total examples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print(f"\nFirst example:")
    print(f"Instruction: {dataset[0]['instruction'][:100]}...")
    print(f"Output: {dataset[0]['output'][:100]}...")

    # Shuffle
    dataset = dataset.shuffle(seed=config.seed)

    # Calculate split indices
    n = len(dataset)
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))

    # Create splits
    splits = DatasetDict({
        'train': dataset.select(range(train_end)),
        'validation': dataset.select(range(train_end, val_end)),
        'test': dataset.select(range(val_end, n))
    })

    print(f"\nSplit sizes:")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Validation: {len(splits['validation'])}")
    print(f"  Test: {len(splits['test'])}")

    return splits


def format_example(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Format a single example for training.
    Creates a simple instruction-response format.
    """
    instruction = example['instruction']
    output = example['output']

    # Simple format that works well
    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return {'text': text}


def prepare_dataset(config, tokenizer=None) -> DatasetDict:
    """
    Main function to prepare dataset for training.

    Args:
        config: Task2Config object
        tokenizer: Optional tokenizer (will load if not provided)

    Returns:
        Formatted DatasetDict ready for SFTTrainer
    """
    # Load tokenizer if not provided
    if tokenizer is None:
        print(f"Loading tokenizer: {config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            cache_dir=config.cache_dir
        )

    # Load and split
    splits = load_and_split_dataset(config)

    # Apply train size fraction
    if config.train_size_fraction < 1.0:
        original_size = len(splits['train'])
        new_size = int(original_size * config.train_size_fraction)
        splits['train'] = splits['train'].select(range(new_size))
        print(f"\nApplying train_size_fraction={config.train_size_fraction}")
        print(f"  Reduced train from {original_size} to {new_size} examples")

    # Format all splits
    print("\nFormatting examples...")
    formatted_splits = DatasetDict()

    for split_name in ['train', 'validation', 'test']:
        formatted_splits[split_name] = splits[split_name].map(
            format_example,
            desc=f"Formatting {split_name}"
        )
        print(f"  {split_name}: {len(formatted_splits[split_name])} examples")

    # Show sample
    print("\n=== Sample Formatted Text ===")
    print(formatted_splits['train'][0]['text'][:300])
    print("...")
    print("=" * 50)

    return formatted_splits


if __name__ == "__main__":
    # Test the data loading
    from config import Task2Config

    config = Task2Config()
    datasets = prepare_dataset(config)

    print("\n✓ Data loading test successful!")
    print(f"Train size: {len(datasets['train'])}")