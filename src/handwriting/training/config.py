from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int = 42
    batch_size: int = 32
    epochs: int = 70
    learning_rate: float = 0.0005
    augmentation_copies: int = 5