"""
Configuration for Task 2: Code Generation
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Task2Config:
    """Configuration for code generation experiments"""

    # Model
    model_name: str = "JetBrains/Mellum-4b-base"

    # Dataset
    dataset_name: str = "flytech/python-codes-25k"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Training size experiment
    train_size_fraction: float = 1.0  # 0.3, 0.5, or 1.0

    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512

    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Paths
    output_dir: str = "./outputs"
    cache_dir: Optional[str] = "./cache"

    # Other
    seed: int = 42
    use_fp16: bool = True

    def get_experiment_name(self) -> str:
        """Generate experiment name based on config"""
        if self.train_size_fraction < 1.0:
            size_str = f"size_{int(self.train_size_fraction * 100)}"
        else:
            size_str = "full"
        return f"lora_r{self.lora_r}_{size_str}"