"""Training entrypoint.

    uv run python -m freuid.train --config configs/baseline.yaml

STUB: wire the loop once data.py knows the dataset layout. The skeleton fixes the
reproducible parts (seed, config, deterministic flags) that the competition requires.
"""

from __future__ import annotations

import argparse

from freuid.config import load_config
from freuid.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    print(f"[train] config '{cfg.name}' loaded; seed={cfg.seed}, backbone={cfg.backbone}")
    # TODO(team): datasets -> dataloaders -> model -> BCEWithLogitsLoss loop.
    #   - validate each epoch with freuid.metrics.evaluate(scores, labels)
    #   - checkpoint best AuDET to checkpoints/<cfg.name>.pt (gitignored)
    raise NotImplementedError("Training loop pending dataset wiring (see data.py).")


if __name__ == "__main__":
    main()
