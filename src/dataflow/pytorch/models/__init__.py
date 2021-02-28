"""
Author: Mathieu Tuli
Date: 27-02-2021
GitHub: @MathieuTuli
Email: tuli.mathieu@gmail.com
"""
from dataflow.pytorch.models.base import BaseTransformer

import torch
import sys


def get_network(name: str) -> torch.nn.Module:
    try:
        model_class = getattr(sys.modules[__name__], name)
        return model_class()
    except AttributeError:
        raise AttributeError(f"Class name {name} unknown")
