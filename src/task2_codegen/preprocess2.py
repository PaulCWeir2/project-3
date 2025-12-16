"""
Data preprocessing for Task 2: Code Generation
"""
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Dict, Any


def load_and_split_python_codes(
        dataset_name: str = "flytech/python-codes-25k",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        cache_dir: str = None
) -> DatasetDict:
    """
    Load and split the Python code dataset.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
        cache_dir: Directory to cache the dataset

    Returns:
        DatasetDict with train, validation, and test splits
    """
    print(f"Loading dataset: {dataset_name}")

    # Load dataset
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    # The dataset only has a train split, so we need to split it
    full_dataset = dataset["train"]

    # Shuffle
    full_dataset = full_dataset.shuffle(seed=seed)

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # Split the dataset
    train_split = full_dataset.select(range(train_size))
    val_split = full_dataset.select(range(train_size, train_size + val_size))
    test_split = full_dataset.select(range(train_size + val_size, total_size))

    print(f"Dataset sizes - Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")

    return DatasetDict({
        "train": train_split,
        "validation": val_split,
        "test": test_split
    })


def format_for_chat_template(
        examples: Dict[str, Any],
        tokenizer: AutoTokenizer
) -> Dict[str, str]:
    """
    Apply chat template to format instruction + response pairs.

    The dataset has 'prompt' and 'code' fields.  # ← Updated comment
    We format them as a conversation for the model.

    Args:
        examples: Batch of examples from the dataset
        tokenizer: Tokenizer with chat template

    Returns:
        Formatted text for each example
    """
    formatted_texts = []

    for instruction, output in zip(examples["instruction"], examples["output"]):  # ← Changed from instruction/output
        # Create a conversation format
        messages = [
            {"role": "user", "content": instruction},  # ← Changed from instruction
            {"role": "assistant", "content": output}  # ← Changed from output
        ]

        # Apply chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback format if no chat template
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"  # ← Changed

        formatted_texts.append(text)

    return {"text": formatted_texts}


def prepare_code_dataset(
        config,
        tokenizer: AutoTokenizer = None
) -> DatasetDict:
    """
    Main function to prepare the code generation dataset.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and split dataset
    print("Loading and splitting code dataset...")
    dataset = load_and_split_python_codes(
        dataset_name=config.dataset_name,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        cache_dir=config.cache_dir
    )

    # ADD THIS DEBUG
    print(f"DEBUG: Train size before fraction: {len(dataset['train'])}")

    # Apply train size fraction if specified
    if config.train_size_fraction < 1.0:
        train_size = int(len(dataset["train"]) * config.train_size_fraction)
        dataset["train"] = dataset["train"].select(range(train_size))
        print(f"Using {config.train_size_fraction * 100}% of training data: {train_size} examples")

    # ADD THIS DEBUG
    print(f"DEBUG: Train size after fraction: {len(dataset['train'])}")
    print(f"DEBUG: Sample instruction: {dataset['train'][0]['instruction'][:100]}")
    print(f"DEBUG: Sample output: {dataset['train'][0]['output'][:100]}")

    # Format for chat template
    print("Formatting dataset with chat template...")
    formatted_dataset = DatasetDict()

    for split in ["train", "validation", "test"]:
        formatted_dataset[split] = dataset[split].map(
            lambda examples: format_for_chat_template(examples, tokenizer),
            batched=True,
            desc=f"Formatting {split} set"
        )
        # ADD THIS DEBUG
        print(f"DEBUG: Formatted {split} size: {len(formatted_dataset[split])}")
        if len(formatted_dataset[split]) > 0:
            print(f"DEBUG: Sample formatted text from {split}: {formatted_dataset[split][0]['text'][:200]}")

    print("Dataset preparation complete!")
    return formatted_dataset


if __name__ == "__main__":
    # Test the preprocessing
    from config2 import CodeGenTrainingConfig
    from transformers import AutoTokenizer

    config = CodeGenTrainingConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    dataset = prepare_code_dataset(config, tokenizer)

    print("\nSample formatted text:")
    print(dataset["train"][0]["text"])