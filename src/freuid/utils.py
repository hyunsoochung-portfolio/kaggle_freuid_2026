"""Reproducibility helpers. Determinism is a competition requirement, not a luxury."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/freuid/utils.py -> repo root


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no-op if missing).

    Doesn't overwrite variables already set in the environment. Checks ``path`` if given,
    else ``.env`` in the current working directory, else the repo root's ``.env`` -- so it
    works whether a script is run from the repo root or elsewhere.
    """
    candidates = [Path(path)] if path is not None else [Path(".env"), _REPO_ROOT / ".env"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value:
                os.environ.setdefault(key, value)
        return


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
