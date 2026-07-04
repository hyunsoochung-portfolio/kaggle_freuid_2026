"""Resolved-config diff tool: prints every key that differs between two experiment YAMLs.

    python scripts/config_diff.py configs/baseline_v1.yaml configs/finetune_v0.yaml

Loads both YAMLs through ``freuid.config.load_config`` (the exact same resolution
train.py/infer.py use, including defaults for keys the YAML doesn't set), flattens the
dataclass + nested ``extra`` dict, and reports every key whose resolved value differs.
Used to confirm a new stage config changes only what it claims to.
"""

from __future__ import annotations

import argparse

from freuid.config import load_config


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def diff_configs(path_a: str, path_b: str) -> dict[str, tuple]:
    flat_a = _flatten(vars(load_config(path_a)))
    flat_b = _flatten(vars(load_config(path_b)))
    keys = sorted(set(flat_a) | set(flat_b))
    diffs = {}
    for k in keys:
        va, vb = flat_a.get(k, "<absent>"), flat_b.get(k, "<absent>")
        if va != vb:
            diffs[k] = (va, vb)
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_a")
    parser.add_argument("config_b")
    args = parser.parse_args()

    diffs = diff_configs(args.config_a, args.config_b)
    print(f"[config_diff] {args.config_a} vs {args.config_b}: {len(diffs)} differing key(s)")
    for k, (va, vb) in diffs.items():
        print(f"  {k}: {va!r} -> {vb!r}")


if __name__ == "__main__":
    main()
