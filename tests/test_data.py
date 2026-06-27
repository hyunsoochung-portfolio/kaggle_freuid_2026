"""Tests for the Leave-One-Domain-Out (lodo_split) validation split.

Uses a synthetic train_labels.csv so it runs without the real (gitignored) dataset.
load_labels only needs the id/type/label columns; image files need not exist.
"""

import pandas as pd
import pytest

from freuid.data import lodo_split


def _write_labels(tmp_path, rows):
    """Write a minimal train_labels.csv to tmp_path and return it as the data root."""
    df = pd.DataFrame(rows, columns=["id", "label", "type"])
    df["image_path"] = df["id"] + ".jpeg"
    df["is_digital"] = False
    df.to_csv(tmp_path / "train_labels.csv", index=False)
    return tmp_path


def test_lodo_holds_out_only_the_chosen_domain(tmp_path):
    root = _write_labels(
        tmp_path,
        [
            ("a", 0, "EGYPT/DL"),
            ("b", 1, "EGYPT/DL"),
            ("c", 0, "MAURITIUS/ID"),
            ("d", 1, "MAURITIUS/ID"),
            ("e", 0, "SPAIN/PP"),
            ("f", 1, "SPAIN/PP"),
        ],
    )
    train_ids, val_ids = lodo_split(root, "MAURITIUS/ID")

    assert val_ids == {"c", "d"}
    assert train_ids == {"a", "b", "e", "f"}
    assert train_ids.isdisjoint(val_ids)


def test_lodo_train_and_val_share_no_domain(tmp_path):
    rows = [
        ("a", 0, "EGYPT/DL"),
        ("b", 1, "EGYPT/DL"),
        ("c", 0, "MAURITIUS/ID"),
        ("d", 1, "MAURITIUS/ID"),
    ]
    root = _write_labels(tmp_path, rows)
    df = pd.DataFrame(rows, columns=["id", "label", "type"])
    train_ids, val_ids = lodo_split(root, "MAURITIUS/ID")

    train_types = set(df[df["id"].isin(train_ids)]["type"])
    val_types = set(df[df["id"].isin(val_ids)]["type"])
    assert train_types.isdisjoint(val_types)


def test_lodo_unknown_domain_raises(tmp_path):
    root = _write_labels(tmp_path, [("a", 0, "EGYPT/DL"), ("b", 1, "EGYPT/DL")])
    with pytest.raises(ValueError, match="not found"):
        lodo_split(root, "ATLANTIS/ID")


def test_lodo_single_class_domain_raises(tmp_path):
    root = _write_labels(
        tmp_path,
        [
            ("a", 0, "EGYPT/DL"),
            ("b", 1, "EGYPT/DL"),
            ("c", 0, "MAURITIUS/ID"),  # bona-fide only -> AuDET undefined
            ("d", 0, "MAURITIUS/ID"),
        ],
    )
    with pytest.raises(ValueError, match="single-class"):
        lodo_split(root, "MAURITIUS/ID")
