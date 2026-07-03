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
    """Seed python / numpy / torch so a config reproduces a ranked result.

    deterministic=True trades speed for bit-exact repro (cudnn.benchmark off).
    Use deterministic=False for fast iteration (smoke tests, hparam search);
    switch back to True for the run you'll actually submit / report.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    except ImportError:
        pass
