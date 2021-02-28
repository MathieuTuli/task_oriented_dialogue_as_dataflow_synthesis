from dataflow.pytorch.models.base import BaseTransformer

import torch
import sys


def get_network(name: str) -> torch.nn.Module:
    try:
        model_class = getattr(sys.modules[__name__], name)
        return model_class()
    except AttributeError:
        raise AttributeError(f"Class name {name} unknown")
