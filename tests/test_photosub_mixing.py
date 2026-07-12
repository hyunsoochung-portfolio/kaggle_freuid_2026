"""Tests for the train-time photosub integration: freuid.photosub.mixing (dataset row
selection, twin-pair batching), freuid.loss.pair_hinge_loss, and freuid.photosub.probes (probe
hooks). All synthetic: tiny in-memory data, no real dataset / regions cache / GPU / model needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from freuid.loss import pair_hinge_loss
from freuid.photosub.mixing import (
    PhotosubTwinDataset,
    TwinPairBatchSampler,
    count_pairs_in_batch,
    load_photosub_rows,
    mode_bucket,
    select_mixed_rows,
    unpack_photosub_batch,
)
from freuid.photosub.probes import _pct_rank, _require_probe_csv, probes_dir, run_probe_hooks


# ---------------------------------------------------------------------------
# select_mixed_rows
# ---------------------------------------------------------------------------

def _rows_df(specs: list[tuple[str, str, str]]) -> pd.DataFrame:
    """specs: list of (id, source_id, mode)."""
    return pd.DataFrame({
        "id": [s[0] for s in specs],
        "image_path": [f"/fake/{s[0]}.jpeg" for s in specs],
        "label": [1] * len(specs),
        "is_digital": [True] * len(specs),
        "type": ["EGYPT/DL"] * len(specs),
        "source_id": [s[1] for s in specs],
        "mode": [s[2] for s in specs],
        "params": ["{}"] * len(specs),
    })


class TestSelectMixedRows:
    def test_filters_out_rows_whose_source_is_not_in_train_ids(self):
        rows = _rows_df([("r1", "src_in", "A"), ("r2", "src_out", "A")])
        selected = select_mixed_rows(
            rows, train_ids={"src_in"}, mode_weights={"A": 1.0}, share=0.5, base_n_positive=1,
        )
        assert set(selected["id"]) == {"r1"}

    def test_share_zero_selects_nothing(self):
        rows = _rows_df([("r1", "src_in", "A")])
        selected = select_mixed_rows(
            rows, train_ids={"src_in"}, mode_weights={"A": 1.0}, share=0.0, base_n_positive=100,
        )
        assert selected.empty

    def test_mode_weights_respected_proportionally(self):
        # 100 positives base, share=0.2 -> n_target = 100*0.2/0.8 = 25 rows total.
        # weights A:0.8 B:0.2 -> ~20 A, ~5 B.
        specs = [(f"a{i}", f"src{i}", "A") for i in range(50)] + [(f"b{i}", f"src{i+50}", "B") for i in range(50)]
        rows = _rows_df(specs)
        train_ids = {s[1] for s in specs}
        selected = select_mixed_rows(
            rows, train_ids=train_ids, mode_weights={"A": 0.8, "B": 0.2}, share=0.2,
            base_n_positive=100, seed=0,
        )
        counts = selected["mode"].value_counts().to_dict()
        assert counts.get("A", 0) == 20
        assert counts.get("B", 0) == 5

    def test_insufficient_rows_in_a_mode_takes_all_available_without_raising(self):
        rows = _rows_df([("a1", "src1", "A")])  # only 1 row available for mode A
        selected = select_mixed_rows(
            rows, train_ids={"src1"}, mode_weights={"A": 1.0}, share=0.5, base_n_positive=100,
        )
        assert len(selected) == 1

    def test_d_main_and_d_ghost_bucket_into_d_for_weighting(self):
        specs = [(f"d{i}", f"src{i}", "D_main" if i % 2 == 0 else "D_ghost") for i in range(10)]
        rows = _rows_df(specs)
        train_ids = {s[1] for s in specs}
        selected = select_mixed_rows(
            rows, train_ids=train_ids, mode_weights={"D": 1.0}, share=0.5, base_n_positive=10,
        )
        assert len(selected) == 10  # n_target = 10*0.5/0.5 = 10, all of them eligible under "D"

    def test_share_out_of_range_raises(self):
        rows = _rows_df([("a1", "src1", "A")])
        with pytest.raises(ValueError, match="share"):
            select_mixed_rows(rows, train_ids={"src1"}, mode_weights={"A": 1.0}, share=1.0, base_n_positive=1)

    def test_all_zero_mode_weights_raises(self):
        rows = _rows_df([("a1", "src1", "A")])
        with pytest.raises(ValueError, match="mode_weights"):
            select_mixed_rows(rows, train_ids={"src1"}, mode_weights={"A": 0.0}, share=0.5, base_n_positive=1)


def test_mode_bucket():
    assert mode_bucket("D_main") == "D"
    assert mode_bucket("D_ghost") == "D"
    assert mode_bucket("A") == "A"
    assert mode_bucket("C") == "C"


def _rows_df_with_types(specs: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """specs: list of (id, source_id, mode, type)."""
    return pd.DataFrame({
        "id": [s[0] for s in specs],
        "image_path": [f"/fake/{s[0]}.jpeg" for s in specs],
        "label": [1] * len(specs),
        "is_digital": [True] * len(specs),
        "type": [s[3] for s in specs],
        "source_id": [s[1] for s in specs],
        "mode": [s[2] for s in specs],
        "params": ["{}"] * len(specs),
    })


class TestSelectMixedRowsTemplateCap:
    def test_per_template_cap_bounds_any_single_template(self):
        """50 MAURITIUS/ID + 50 EGYPT/DL rows, all mode D. Uncapped selection at n_target=60
        could, by chance, draw disproportionately from one template; a 0.6 cap should keep
        neither template above 36 (ceil(60*0.6))."""
        specs = (
            [(f"m{i}", f"srcm{i}", "D_ghost", "MAURITIUS/ID") for i in range(50)]
            + [(f"e{i}", f"srce{i}", "D_ghost", "EGYPT/DL") for i in range(50)]
        )
        rows = _rows_df_with_types(specs)
        train_ids = {s[1] for s in specs}
        selected = select_mixed_rows(
            rows, train_ids=train_ids, mode_weights={"D": 1.0}, share=0.375,
            base_n_positive=100, seed=0, per_template_cap=0.6,
        )
        # n_target = 100*0.375/0.625 = 60
        assert len(selected) == 60
        counts = selected["type"].value_counts()
        assert counts.max() <= 36  # ceil(60 * 0.6)
        assert set(counts.index) == {"MAURITIUS/ID", "EGYPT/DL"}

    def test_per_template_cap_none_reproduces_uncapped_selection_count(self):
        """per_template_cap=None (the default) must select the same COUNT as before this
        feature existed -- only the byte-identical guarantee that matters here, since the exact
        row identities depend on RNG draw order either way."""
        specs = [(f"m{i}", f"srcm{i}", "D_ghost", "MAURITIUS/ID") for i in range(20)]
        rows = _rows_df_with_types(specs)
        train_ids = {s[1] for s in specs}
        selected = select_mixed_rows(
            rows, train_ids=train_ids, mode_weights={"D": 1.0}, share=0.5,
            base_n_positive=20, seed=0, per_template_cap=None,
        )
        assert len(selected) == 20

    def test_per_template_cap_fills_over_cap_when_only_one_template_available(self):
        """A cap that can't be met without under-filling must still hit n_target (with a loud
        warning, not silently under-shooting) -- a fairness ceiling, not a hard quota."""
        specs = [(f"m{i}", f"srcm{i}", "D_ghost", "MAURITIUS/ID") for i in range(30)]
        rows = _rows_df_with_types(specs)
        train_ids = {s[1] for s in specs}
        selected = select_mixed_rows(
            rows, train_ids=train_ids, mode_weights={"D": 1.0}, share=1.0 / 3.0,
            base_n_positive=20, seed=0, per_template_cap=0.1,
        )
        # n_target = 20*(1/3)/(2/3) = 10, but only one template exists -- cap can't be respected
        assert len(selected) == 10

    def test_per_template_cap_applied_independently_per_mode(self):
        """Mode A (5 templates, no cap needed in practice) and mode D (2 ghost templates,
        needs the cap) are capped independently -- a cap tight for D shouldn't affect A's
        selection at all."""
        specs = (
            [(f"a{i}", f"srca{i}", "A", "GUINEA/DL") for i in range(10)]
            + [(f"m{i}", f"srcm{i}", "D_ghost", "MAURITIUS/ID") for i in range(10)]
            + [(f"e{i}", f"srce{i}", "D_ghost", "EGYPT/DL") for i in range(10)]
        )
        rows = _rows_df_with_types(specs)
        train_ids = {s[1] for s in specs}
        selected = select_mixed_rows(
            rows, train_ids=train_ids, mode_weights={"A": 0.5, "D": 0.5}, share=0.5,
            base_n_positive=20, seed=0, per_template_cap=0.6,
        )
        a_rows = selected[selected["mode"] == "A"]
        assert len(a_rows) == 10
        assert set(a_rows["type"]) == {"GUINEA/DL"}


class TestLoadPhotosubRows:
    def test_missing_file_returns_empty_dataframe_with_expected_columns(self, tmp_path):
        df = load_photosub_rows(tmp_path / "does_not_exist.csv")
        assert df.empty
        assert set(df.columns) >= {"id", "image_path", "label", "source_id", "mode"}

    def test_loads_existing_csv(self, tmp_path):
        csv_path = tmp_path / "rows.csv"
        _rows_df([("r1", "src1", "A")]).to_csv(csv_path, index=False)
        df = load_photosub_rows(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["source_id"] == "src1"


# ---------------------------------------------------------------------------
# PhotosubTwinDataset
# ---------------------------------------------------------------------------

class _FakeSample:
    def __init__(self, id_: str, label: int):
        self.id = id_
        self.label = label


class _FakeBaseDataset:
    """Stand-in for a plain FreuidDataset: has `.samples` and returns (tensor, label)."""

    def __init__(self, ids_labels: list[tuple[str, int]]):
        self.samples = [_FakeSample(i, lbl) for i, lbl in ids_labels]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.zeros(3, 4, 4), s.label


class _FakeSynthTamperWrapper:
    """Stand-in for SynthTamperWrapper: only `.base.samples`, not `.samples` directly."""

    def __init__(self, base: _FakeBaseDataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx]


def _tiny_photo(tmp_path, name: str) -> str:
    path = tmp_path / name
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    return str(path)


class TestPhotosubTwinDataset:
    def test_base_item_pair_group_id_is_own_index(self, tmp_path):
        base = _FakeBaseDataset([("b0", 0), ("b1", 0), ("b2", 1)])
        rows = pd.DataFrame(columns=["id", "image_path", "label", "source_id", "mode"])
        ds = PhotosubTwinDataset(base, rows, transform=lambda im: torch.zeros(3, 4, 4))
        for i in range(len(base)):
            assert ds.pair_group_id(i) == i
            assert not ds.is_photosub(i)

    def test_photosub_item_pair_group_id_is_source_base_index(self, tmp_path):
        base = _FakeBaseDataset([("b0", 0), ("b1", 0), ("b2", 0)])
        rows = pd.DataFrame({
            "id": ["gen1"], "image_path": [_tiny_photo(tmp_path, "gen1.jpeg")],
            "label": [1], "source_id": ["b1"], "mode": ["A"],
        })
        ds = PhotosubTwinDataset(base, rows, transform=lambda im: torch.zeros(3, 4, 4))
        assert len(ds) == 4
        photosub_idx = 3
        assert ds.is_photosub(photosub_idx)
        assert ds.pair_group_id(photosub_idx) == 1  # base index of "b1"
        img, label = ds[photosub_idx][0], ds[photosub_idx][1]
        assert label == 1

    def test_getitem_returns_pair_group_id_as_third_element(self, tmp_path):
        base = _FakeBaseDataset([("b0", 0)])
        rows = pd.DataFrame({
            "id": ["gen1"], "image_path": [_tiny_photo(tmp_path, "gen1.jpeg")],
            "label": [1], "source_id": ["b0"], "mode": ["A"],
        })
        ds = PhotosubTwinDataset(base, rows, transform=lambda im: torch.zeros(3, 4, 4))
        _, label0, pair0 = ds[0]
        _, label1, pair1 = ds[1]
        assert (label0, pair0) == (0, 0)
        assert (label1, pair1) == (1, 0)  # same group as its source

    def test_unresolvable_source_id_raises(self, tmp_path):
        base = _FakeBaseDataset([("b0", 0)])
        rows = pd.DataFrame({
            "id": ["gen1"], "image_path": [_tiny_photo(tmp_path, "gen1.jpeg")],
            "label": [1], "source_id": ["not_in_base"], "mode": ["A"],
        })
        with pytest.raises(ValueError, match="split-discipline"):
            PhotosubTwinDataset(base, rows, transform=lambda im: torch.zeros(3, 4, 4))

    def test_accepts_synth_tamper_wrapper_style_base(self, tmp_path):
        """base may only expose `.base.samples` (SynthTamperWrapper), not `.samples` directly."""
        inner = _FakeBaseDataset([("b0", 0), ("b1", 0)])
        wrapper = _FakeSynthTamperWrapper(inner)
        rows = pd.DataFrame({
            "id": ["gen1"], "image_path": [_tiny_photo(tmp_path, "gen1.jpeg")],
            "label": [1], "source_id": ["b1"], "mode": ["A"],
        })
        ds = PhotosubTwinDataset(wrapper, rows, transform=lambda im: torch.zeros(3, 4, 4))
        assert ds.pair_group_id(2) == 1

    def test_is_pairable_true_for_mode_a_b_false_for_c_d(self, tmp_path):
        base = _FakeBaseDataset([("b0", 0), ("b1", 0), ("b2", 0), ("b3", 0)])
        rows = pd.DataFrame({
            "id": ["genA", "genB", "genC", "genD"],
            "image_path": [_tiny_photo(tmp_path, f"gen{m}.jpeg") for m in "ABCD"],
            "label": [1, 1, 1, 1], "source_id": ["b0", "b1", "b2", "b3"],
            "mode": ["A", "B", "C", "D_ghost"],
        })
        ds = PhotosubTwinDataset(base, rows, transform=lambda im: torch.zeros(3, 4, 4))
        assert ds.is_pairable(4) is True   # A
        assert ds.is_pairable(5) is True   # B
        assert ds.is_pairable(6) is False  # C
        assert ds.is_pairable(7) is False  # D_ghost

    def test_is_pairable_false_for_base_items(self, tmp_path):
        base = _FakeBaseDataset([("b0", 0)])
        rows = pd.DataFrame(columns=["id", "image_path", "label", "source_id", "mode"])
        ds = PhotosubTwinDataset(base, rows, transform=lambda im: torch.zeros(3, 4, 4))
        assert ds.is_pairable(0) is False


# ---------------------------------------------------------------------------
# TwinPairBatchSampler
# ---------------------------------------------------------------------------

class _FakePairDataset:
    """Minimal stand-in satisfying TwinPairBatchSampler's protocol: pair_group_id(i),
    is_photosub(i), is_pairable(i), __len__. n_base plain items (group id = own idx) +
    n_photosub items whose group id round-robins over the base items. ``all_pairable=False``
    marks every photosub item as NOT pairable (simulating an all-C/D corpus), for the per-mode
    gating tests -- default True reproduces the pre-per-mode-gating behavior every other test
    in this class relies on."""

    def __init__(self, n_base: int, n_photosub: int, all_pairable: bool = True):
        self.n_base = n_base
        self.n_photosub = n_photosub
        self.all_pairable = all_pairable

    def __len__(self):
        return self.n_base + self.n_photosub

    def is_photosub(self, idx: int) -> bool:
        return idx >= self.n_base

    def is_pairable(self, idx: int) -> bool:
        if idx < self.n_base:
            return False
        return self.all_pairable

    def pair_group_id(self, idx: int) -> int:
        if idx < self.n_base:
            return idx
        return (idx - self.n_base) % self.n_base


class TestRehearsalFrequency:
    """photosub_v1 spec item 1: "rehearsal weighting asserting real fraud rows keep
    finetune_v0-era effective sampling frequency (loader test, not assumption)". The sampler's
    own docstring already admits displacement CAN drop a real row from an epoch entirely (to
    make room for a forced bona-fide twin) -- "the displaced item simply isn't seen this batch,
    same kind of coverage loss drop_last already accepts." This class does NOT assume that loss
    stays negligible; it MEASURES the actual per-epoch coverage of real (base, including a
    tagged "fraud" subset) rows under realistic photosub_v1-scale ratios and asserts the
    measured loss is small and bounded -- an assumption checked, not assumed.
    """

    def _realistic_dataset(self, n_base: int = 1000, share: float = 0.15, n_fraud: int = 300):
        """n_base real rows (n_fraud of them tagged "fraud", matching a typical ~30% base
        positive rate), n_photosub = n_base * share/(1-share) rows (mirroring select_mixed_rows'
        own n_target formula) whose source ids round-robin over the BONA-FIDE (non-fraud) base
        rows only -- photosub sources are always bona-fide, never fraud, matching pipeline.py's
        real construction."""
        n_photosub = int(round(n_base * share / (1.0 - share)))
        bonafide_base_idx = list(range(n_fraud, n_base))  # fraud rows are [0, n_fraud)
        ds = _FakePairDataset(n_base=n_base, n_photosub=n_photosub, all_pairable=True)
        # Override pair_group_id so photosub sources round-robin over BONA-FIDE rows only
        # (the real construction: a generated row's source_id is always a bona-fide TRAIN id).
        def bonafide_only_group_id(idx):
            if idx < n_base:
                return idx
            return bonafide_base_idx[(idx - n_base) % len(bonafide_base_idx)]
        ds.pair_group_id = bonafide_only_group_id
        return ds, n_fraud

    def test_fraud_row_coverage_loss_is_small_and_bounded(self):
        """Measured (not assumed): at photosub_v1-realistic settings (share=0.15,
        twin_pair_prob=0.5, n_base=1000/n_fraud=300), per-epoch fraud-row coverage comes out
        ~90-93% across seeds -- a REAL ~7-10% rehearsal-frequency cost from the displacement
        mechanism, not the near-zero loss a "same as finetune_v0" assumption would predict. This
        is the actual finding docs/photosub_v1_spec.md reports for this item -- accepted as a
        known, bounded, and now-quantified cost of twin-pair batching (same tradeoff class as
        drop_last, per TwinPairBatchSampler's own docstring), not silently assumed negligible.
        The bound below is set below the worst observed value with margin for seed variance,
        not tuned to just barely pass."""
        ds, n_fraud = self._realistic_dataset(n_base=1000, share=0.15, n_fraud=300)
        sampler = TwinPairBatchSampler(ds, batch_size=32, twin_pair_prob=0.5, seed=0)
        seen = set()
        for batch in sampler:
            seen.update(int(x) for x in batch)
        fraud_seen = sum(1 for i in range(n_fraud) if i in seen)
        coverage = fraud_seen / n_fraud
        assert coverage > 0.80, f"fraud-row coverage this epoch: {coverage:.3f} (n_fraud={n_fraud})"

    def test_fraud_row_coverage_matches_non_fraud_base_coverage(self):
        """The displacement mechanism has no reason to target fraud rows SPECIFICALLY more than
        any other non-photosub row -- fraud and non-fraud base coverage should be statistically
        similar, not systematically worse for fraud specifically."""
        ds, n_fraud = self._realistic_dataset(n_base=1000, share=0.15, n_fraud=300)
        sampler = TwinPairBatchSampler(ds, batch_size=32, twin_pair_prob=0.5, seed=0)
        seen = set()
        for batch in sampler:
            seen.update(int(x) for x in batch)
        fraud_coverage = sum(1 for i in range(n_fraud) if i in seen) / n_fraud
        nonfraud_coverage = sum(1 for i in range(n_fraud, 1000) if i in seen) / (1000 - n_fraud)
        assert abs(fraud_coverage - nonfraud_coverage) < 0.05

    def test_photosub_free_baseline_has_zero_coverage_loss_beyond_drop_last(self):
        """Sanity anchor: with twin_pair_prob=0.0 (no photosub-driven displacement at all,
        finetune_v0's own regime), coverage loss is EXACTLY drop_last's remainder -- confirms
        the measured loss above is attributable to twin-pairing, not some other bug in the
        sampler's permutation/batching logic."""
        ds, n_fraud = self._realistic_dataset(n_base=1000, share=0.15, n_fraud=300)
        sampler = TwinPairBatchSampler(ds, batch_size=32, twin_pair_prob=0.0, seed=0)
        seen = set()
        for batch in sampler:
            seen.update(int(x) for x in batch)
        n_total = len(ds)
        expected_dropped = n_total % 32
        assert n_total - len(seen) == expected_dropped


class TestTwinPairBatchSampler:
    def test_every_batch_has_fixed_size(self):
        ds = _FakePairDataset(n_base=40, n_photosub=10)
        sampler = TwinPairBatchSampler(ds, batch_size=8, twin_pair_prob=0.7, seed=0)
        for batch in sampler:
            assert len(batch) == 8
            assert len(set(batch)) == len(batch)  # no duplicate indices within a batch

    def test_deterministic_given_same_seed_and_epoch(self):
        ds = _FakePairDataset(n_base=40, n_photosub=10)
        s1 = TwinPairBatchSampler(ds, batch_size=8, twin_pair_prob=0.7, seed=0)
        s2 = TwinPairBatchSampler(ds, batch_size=8, twin_pair_prob=0.7, seed=0)
        assert list(s1) == list(s2)

    def test_different_epoch_gives_different_batches(self):
        ds = _FakePairDataset(n_base=40, n_photosub=10)
        s1 = TwinPairBatchSampler(ds, batch_size=8, twin_pair_prob=0.7, seed=0)
        s2 = TwinPairBatchSampler(ds, batch_size=8, twin_pair_prob=0.7, seed=0)
        s2.set_epoch(1)
        assert list(s1) != list(s2)

    def test_twin_pair_prob_one_always_pairs_when_possible(self):
        """With prob=1.0 every photosub item's twin should land in the same batch (each
        photosub item has a distinct group, and n_base is large enough that displacement always
        finds a non-forced slot)."""
        ds = _FakePairDataset(n_base=100, n_photosub=5)
        sampler = TwinPairBatchSampler(ds, batch_size=20, twin_pair_prob=1.0, seed=1)
        for batch in sampler:
            batch_set = set(batch)
            for idx in batch:
                if ds.is_photosub(idx):
                    twin = ds.pair_group_id(idx)
                    assert twin in batch_set, f"photosub item {idx}'s twin {twin} missing from its batch"

    def test_twin_pair_prob_zero_rate_matches_chance(self):
        """With prob=0.0 the sampler must never force a pairing -- observed co-occurrence
        should be close to the base rate you'd expect from pure random batching, not near 1.0."""
        ds = _FakePairDataset(n_base=200, n_photosub=20)
        n_paired = 0
        n_photosub_seen = 0
        for epoch in range(20):
            sampler = TwinPairBatchSampler(ds, batch_size=16, twin_pair_prob=0.0, seed=epoch)
            for batch in sampler:
                batch_set = set(batch)
                for idx in batch:
                    if ds.is_photosub(idx):
                        n_photosub_seen += 1
                        if ds.pair_group_id(idx) in batch_set:
                            n_paired += 1
        rate = n_paired / max(n_photosub_seen, 1)
        assert rate < 0.3  # well below prob=1.0's guaranteed pairing; chance-level for batch_size=16/220

    def test_pairing_rate_near_configured_prob(self):
        ds = _FakePairDataset(n_base=300, n_photosub=60)
        target_prob = 0.5
        n_paired = 0
        n_photosub_seen = 0
        for epoch in range(30):
            sampler = TwinPairBatchSampler(ds, batch_size=16, twin_pair_prob=target_prob, seed=100 + epoch)
            for batch in sampler:
                batch_set = set(batch)
                for idx in batch:
                    if ds.is_photosub(idx):
                        n_photosub_seen += 1
                        if ds.pair_group_id(idx) in batch_set:
                            n_paired += 1
        rate = n_paired / n_photosub_seen
        assert abs(rate - target_prob) < 0.15

    def test_non_pairable_items_never_seek_a_twin(self):
        """all_pairable=False (simulating an all-C/D photosub corpus) at twin_pair_prob=1.0 --
        no photosub item should ever land next to its own group id via the sampler's forced
        rearrangement (a random co-occurrence is still possible by chance, so this checks the
        rate stays at chance level, not exactly zero)."""
        ds = _FakePairDataset(n_base=300, n_photosub=60, all_pairable=False)
        sampler = TwinPairBatchSampler(ds, batch_size=16, twin_pair_prob=1.0, seed=0)
        n_paired = n_photosub_seen = 0
        for batch in sampler:
            batch_set = set(batch)
            for idx in batch:
                if ds.is_photosub(idx):
                    n_photosub_seen += 1
                    if ds.pair_group_id(idx) in batch_set:
                        n_paired += 1
        rate = n_paired / max(n_photosub_seen, 1)
        assert rate < 0.3  # chance level, same bound test_twin_pair_prob_zero_rate_matches_chance uses

    def test_pairable_items_still_pair_when_mixed_with_non_pairable(self):
        """A dataset with SOME pairable (A/B) and SOME non-pairable (C/D) photosub items: only
        the pairable half should show the twin_pair_prob=1.0 pairing rate; verified via two
        separate _FakePairDataset instances sharing the same base range is awkward, so this
        directly patches is_pairable per-index to simulate a mixed corpus."""
        ds = _FakePairDataset(n_base=300, n_photosub=60, all_pairable=True)
        # Odd photosub indices (base+1, base+3, ...) become non-pairable (simulated C/D).
        orig_is_pairable = ds.is_pairable

        def mixed_is_pairable(idx):
            if idx < ds.n_base:
                return False
            return (idx - ds.n_base) % 2 == 0

        ds.is_pairable = mixed_is_pairable
        sampler = TwinPairBatchSampler(ds, batch_size=16, twin_pair_prob=1.0, seed=0)
        pairable_paired = pairable_seen = nonpairable_paired = nonpairable_seen = 0
        for batch in sampler:
            batch_set = set(batch)
            for idx in batch:
                if not ds.is_photosub(idx):
                    continue
                if orig_is_pairable(idx) and mixed_is_pairable(idx):
                    pairable_seen += 1
                    if ds.pair_group_id(idx) in batch_set:
                        pairable_paired += 1
                elif not mixed_is_pairable(idx):
                    nonpairable_seen += 1
                    if ds.pair_group_id(idx) in batch_set:
                        nonpairable_paired += 1
        assert pairable_seen > 0 and nonpairable_seen > 0
        assert (pairable_paired / pairable_seen) > 0.8  # near-guaranteed at prob=1.0
        assert (nonpairable_paired / nonpairable_seen) < 0.3  # chance level only


# ---------------------------------------------------------------------------
# pair_hinge_loss / count_pairs_in_batch
# ---------------------------------------------------------------------------

class TestPairHingeLoss:
    def test_penalizes_when_tampered_logit_not_above_clean_by_margin(self):
        logits = torch.tensor([[0.5], [0.5]])  # tampered==clean logit -> full margin violated
        labels = torch.tensor([1, 0])
        pair_ids = torch.tensor([7, 7])
        loss = pair_hinge_loss(logits, labels, pair_ids, margin=1.0)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_zero_when_margin_already_satisfied(self):
        logits = torch.tensor([[5.0], [0.0]])  # tampered well above clean
        labels = torch.tensor([1, 0])
        pair_ids = torch.tensor([7, 7])
        loss = pair_hinge_loss(logits, labels, pair_ids, margin=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_zero_when_no_group_has_both_labels(self):
        logits = torch.tensor([[1.0], [2.0], [3.0]])
        labels = torch.tensor([1, 1, 0])
        pair_ids = torch.tensor([1, 2, 3])  # every item in its own singleton group
        loss = pair_hinge_loss(logits, labels, pair_ids, margin=1.0)
        assert loss.item() == pytest.approx(0.0)

    def test_gradient_flows_to_logits(self):
        logits = torch.tensor([[0.0], [0.0]], requires_grad=True)
        labels = torch.tensor([1, 0])
        pair_ids = torch.tensor([1, 1])
        loss = pair_hinge_loss(logits, labels, pair_ids, margin=1.0)
        loss.backward()
        assert logits.grad is not None
        assert torch.any(logits.grad != 0)

    def test_ignores_unrelated_batch_items(self):
        logits = torch.tensor([[0.5], [0.5], [10.0], [10.0]])
        labels = torch.tensor([1, 0, 1, 1])
        pair_ids = torch.tensor([7, 7, 8, 9])  # group 8/9 are singletons, no pairing
        loss = pair_hinge_loss(logits, labels, pair_ids, margin=1.0)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)  # only group 7 contributes


def test_count_pairs_in_batch():
    labels = torch.tensor([1, 0, 1, 1])
    pair_ids = torch.tensor([7, 7, 8, 9])  # only group 7 has both labels present
    assert count_pairs_in_batch(labels, pair_ids) == 1


def test_unpack_photosub_batch_moves_to_device():
    imgs = torch.zeros(2, 3, 4, 4)
    labels = torch.tensor([0, 1])
    pair_ids = torch.tensor([0, 1])
    out_imgs, out_labels, out_pairs = unpack_photosub_batch((imgs, labels, pair_ids), torch.device("cpu"))
    assert out_imgs.shape == imgs.shape
    assert torch.equal(out_labels, labels)
    assert torch.equal(out_pairs, pair_ids)


# ---------------------------------------------------------------------------
# probes.py
# ---------------------------------------------------------------------------

class TestPctRank:
    def test_matches_hand_computed_percentile(self):
        reference = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        values = np.array([0.35])
        pct = _pct_rank(values, reference)
        assert pct[0] == pytest.approx(60.0)  # 3/5 below 0.35

    def test_value_equal_to_every_reference_point_is_50th_percentile(self):
        reference = np.array([0.5, 0.5, 0.5])
        pct = _pct_rank(np.array([0.5]), reference)
        assert pct[0] == pytest.approx(50.0)


class TestProbesDir:
    def test_resolves_relative_to_data_dir_parent(self, tmp_path):
        data_dir = tmp_path / "repo" / "data"
        data_dir.mkdir(parents=True)
        assert probes_dir(str(data_dir)) == tmp_path / "repo" / "data" / "probes"


class TestProbeFailLoudly:
    def test_require_probe_csv_raises_with_helpful_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missed_frauds_deep_ids.csv"):
            _require_probe_csv(tmp_path, "deep")

    def test_run_probe_hooks_raises_before_touching_model_when_probes_missing(self, tmp_path):
        data_dir = tmp_path / "repo" / "data"
        data_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            run_probe_hooks(
                model=None, data_dir=str(data_dir), transform=None, device=None,
                val_scores=np.array([0.1, 0.9]), val_labels=np.array([0, 1]),
            )
