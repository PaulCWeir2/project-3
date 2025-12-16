"""
Data preprocessing for Task 1: Question Answering
"""
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
from typing import Dict, Any, List
import numpy as np


def load_and_split_squad(
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        cache_dir: str = None
) -> DatasetDict:
    """
    Load SQUAD-v2 dataset and split into train/val/test sets.

    Args:
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
        cache_dir: Directory to cache the dataset

    Returns:
        DatasetDict with train, validation, and test splits
    """
    # Load the full SQUAD-v2 dataset
    dataset = load_dataset("squad_v2", cache_dir=cache_dir)

    # SQUAD-v2 comes with train and validation sets
    # We'll combine them and re-split into train/val/test
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # Combine both splits
    combined = concatenate_datasets([train_dataset, val_dataset])

    # Shuffle the combined dataset
    combined = combined.shuffle(seed=seed)

    # Calculate split sizes
    total_size = len(combined)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # Split the dataset
    train_split = combined.select(range(train_size))
    val_split = combined.select(range(train_size, train_size + val_size))
    test_split = combined.select(range(train_size + val_size, total_size))

    return DatasetDict({
        "train": train_split,
        "validation": val_split,
        "test": test_split
    })


def preprocess_squad_examples(
        examples: Dict[str, Any],
        tokenizer: AutoTokenizer,
        max_length: int = 384,
        doc_stride: int = 128,
        is_training: bool = True
) -> Dict[str, Any]:
    """
    Tokenize questions and contexts, and identify answer positions.

    Args:
        examples: Batch of examples from the dataset
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        doc_stride: Stride for sliding window when context is too long
        is_training: Whether this is for training (affects answer position handling)

    Returns:
        Tokenized inputs with start and end positions
    """
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    # Tokenize with truncation and padding
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",  # Only truncate the context
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map overflow tokens back to original examples
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Initialize start and end positions
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["example_id"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Get the sequence ids to identify question vs context
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Map back to the original example
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        answers = examples["answers"][sample_index]

        # If no answer (impossible question in SQUAD-v2)
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Get answer start and end character positions
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find token start and end positions
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:  # Find start of context
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:  # Find end of context
                token_end_index -= 1

            # Check if answer is in this chunk
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Find exact token positions
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_qa_dataset(
        config,
        tokenizer: AutoTokenizer = None
) -> DatasetDict:
    """
    Main function to prepare the entire QA dataset.

    Args:
        config: QATrainingConfig object
        tokenizer: Tokenizer (will be loaded if not provided)

    Returns:
        Preprocessed DatasetDict
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load and split dataset
    print("Loading and splitting SQUAD-v2 dataset...")
    dataset = load_and_split_squad(
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        cache_dir=config.cache_dir
    )

    # Apply train size fraction if specified (for experiments)
    if config.train_size_fraction < 1.0:
        train_size = int(len(dataset["train"]) * config.train_size_fraction)
        dataset["train"] = dataset["train"].select(range(train_size))
        print(f"Using {config.train_size_fraction * 100}% of training data: {train_size} examples")

    print(
        f"Dataset sizes - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = DatasetDict()

    for split in ["train", "validation", "test"]:
        tokenized_datasets[split] = dataset[split].map(
            lambda examples: preprocess_squad_examples(
                examples,
                tokenizer,
                max_length=config.max_length,
                doc_stride=config.doc_stride,
                is_training=(split == "train")
            ),
            batched=True,
            remove_columns=dataset[split].column_names,
            desc=f"Tokenizing {split} set"
        )

    print("Dataset preparation complete!")
    return tokenized_datasets