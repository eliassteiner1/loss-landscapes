from dataclasses import dataclass

import torch


@dataclass
class DataSplit:
    n_total: int
    n_train: int

    @property
    def n_valid(self) -> int:
        return self.n_total - self.n_train


@dataclass
class TrainingConfig:
    n_epochs: int = 12
    batch_size: int = 128
    train_split: float = 0.8


def get_device() -> str:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"> using {device} as device")
    return device