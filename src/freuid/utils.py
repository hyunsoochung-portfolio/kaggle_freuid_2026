"""Reproducibility helpers. Determinism is a competition requirement, not a luxury."""

from __future__ import annotations

import os
import random

import numpy as np


def pick_device():
    """Best available torch device: cuda > mps (Apple) > cpu."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python / numpy / torch so a config reproduces a ranked result."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
