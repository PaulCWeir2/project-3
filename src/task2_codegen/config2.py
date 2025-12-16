"""
Configuration for Task 2: Code Generation with Mellum
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class CodeGenTrainingConfig:
    """Configuration for code generation model training"""

    # Model configuration
    model_name: str = "JetBrains/Mellum-4b-base"

    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list = None  # Will use default for causal LM

    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    max_seq_length: int = 512

    # Data parameters
    dataset_name: str = "flytech/python-codes-25k"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Dataset size experiment (0.3, 0.5, 1.0 for 30%, 50%, 100%)
    train_size_fraction: float = 1.0

    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Paths
    output_dir: str = "./outputs/task2"
    cache_dir: Optional[str] = None
    logging_dir: str = "./logs/task2"
    results_dir: str = "./results/task2"

    # Other
    seed: int = 42
    fp16: bool = True

    def __post_init__(self):
        if self.lora_target_modules is None:
            # Common target modules for causal LM (better coverage)
            # For most transformer models, these are the attention projection layers
            self.lora_target_modules = ["q_proj", "v_proj"]  # ← CHANGED from ["q", "v"]