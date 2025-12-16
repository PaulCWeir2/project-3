"""
Configuration for Task 1: Question Answering with RoBERTa
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class QATrainingConfig:
    """Configuration for QA model training"""

    # Model configuration
    model_name: str = "roberta-base"

    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list = None  # Will be set to ["query", "value"] by default

    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500

    # Data parameters
    max_length: int = 384
    doc_stride: int = 128
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Dataset size experiment (0.3, 0.5, 1.0 for 30%, 50%, 100%)
    train_size_fraction: float = 1.0

    # Paths
    output_dir: str = "./outputs/task1"
    cache_dir: Optional[str] = None
    logging_dir: str = "./logs/task1"
    results_dir: str = "./results/task1"

    # Other
    seed: int = 42
    fp16: bool = True  # Use mixed precision if available

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["query", "value"]