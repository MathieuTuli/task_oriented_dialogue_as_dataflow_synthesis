"""
Author: Mathieu Tuli
GitHub: @MathieuTuli
Email: tuli.mathieu@gmail.com
"""
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

import torch


def get_data(name: str, root: Path, mini_batch_size: int, num_workers: int,
             dist: bool = False) -> Tuple[DataLoader, DataLoader]:
    train_loader = None
    val_loader = None
    return train_loader, val_loader
