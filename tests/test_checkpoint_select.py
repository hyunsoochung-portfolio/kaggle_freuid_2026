"""Tests for freuid.photosub.checkpoint_select: gate-composite eligibility, latest-stable
tiebreak, and last-k weight averaging. All synthetic (no real probe CSVs, no GPU, no VESSL)."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from freuid.photosub.checkpoint_select import (
    CheckpointTracker,
    GateThresholds,
    average_state_dicts,
    evaluate_boundary_gate,
    evaluate_ceiling_gate,
    evaluate_deep9_gate,
    evaluate_gates,
    extract_boundary_baseline,
    extract_ceiling_baseline,
)


def _probe_metrics(
    deep_scores, threshold=0.5, boundary_below=10, boundary_total=59,
    ceiling_rows=(("EGYPT/DL", 50, 100), ("MAURITIUS/ID", 40, 100)),
):
    deep_detail = [{"id": f"d{i}", "score": s, "logit": s} for i, s in enumerate(deep_scores)]
    ceiling_df = pd.DataFrame(
        [{"template": t, "n_above": a, "n_total": n} for t, a, n in ceiling_rows],
    )
    return {
        "probe_threshold_score": threshold,
        "probe_deep_detail": deep_detail,
        "probe_boundary_below_threshold": boundary_below,
        "probe_boundary_total": boundary_total,
        "probe_ceiling_by_template": ceiling_df,
    }


class TestDeep9Gate:
    def test_passes_when_majority_above_and_no_regression(self):
        # baseline below every id's current score -- no regression
        baseline = {f"d{i}": 0.1 for i in range(9)}
        m = _probe_metrics([0.9] * 6 + [0.2] * 3, threshold=0.5)
        ok, detail = evaluate_deep9_gate(m, baseline)
        assert ok is True
        assert detail["n_above"] == 6

    def test_fails_when_not_majority_above(self):
        baseline = {f"d{i}": 0.3 for i in range(9)}
        m = _probe_metrics([0.9] * 3 + [0.2] * 6, threshold=0.5)
        ok, detail = evaluate_deep9_gate(m, baseline)
        assert ok is False
        assert detail["majority_ok"] is False

    def test_fails_when_any_id_below_its_own_finetune_v0_baseline(self):
        """Majority-above can pass while a SPECIFIC id regresses below its own frozen baseline
        -- the per-id regression check must catch that independently of the aggregate count."""
        baseline = {f"d{i}": 0.3 for i in range(9)}
        baseline["d0"] = 5.0  # this id's finetune_v0 baseline was very confident
        scores = [0.9] * 9
        m = _probe_metrics(scores, threshold=0.5)
        m["probe_deep_detail"][0]["logit"] = 0.9  # regressed well below its baseline of 5.0
        ok, detail = evaluate_deep9_gate(m, baseline)
        assert ok is False
        assert "d0" in detail["below_baseline_ids"]

    def test_empty_detail_fails_safe(self):
        m = _probe_metrics([], threshold=0.5)
        ok, detail = evaluate_deep9_gate(m, {})
        assert ok is False


class TestCeilingGate:
    def test_passes_with_no_baseline_yet(self):
        m = _probe_metrics([0.9] * 9)
        ok, detail = evaluate_ceiling_gate(m, baseline_by_template=None)
        assert ok is True

    def test_passes_when_within_tolerance(self):
        m = _probe_metrics([0.9] * 9, ceiling_rows=(("EGYPT/DL", 48, 100),))
        ok, _ = evaluate_ceiling_gate(m, {"EGYPT/DL": (50, 100)}, drop_tolerance=0.20)
        assert ok is True  # 4% relative drop, well under 20% tolerance

    def test_fails_on_material_drop(self):
        m = _probe_metrics([0.9] * 9, ceiling_rows=(("MAURITIUS/ID", 10, 100),))
        ok, detail = evaluate_ceiling_gate(m, {"MAURITIUS/ID": (40, 100)}, drop_tolerance=0.20)
        assert ok is False
        assert "MAURITIUS/ID" in detail["materially_dropped"]

    def test_each_template_checked_independently(self):
        """One template dropping materially fails the gate even if others are fine."""
        m = _probe_metrics(
            [0.9] * 9,
            ceiling_rows=(("EGYPT/DL", 49, 100), ("MAURITIUS/ID", 5, 100)),
        )
        ok, detail = evaluate_ceiling_gate(
            m, {"EGYPT/DL": (50, 100), "MAURITIUS/ID": (40, 100)}, drop_tolerance=0.20,
        )
        assert ok is False
        assert detail["materially_dropped"] == ["MAURITIUS/ID"]


class TestBoundaryGate:
    def test_passes_with_no_baseline_yet(self):
        m = _probe_metrics([0.9] * 9)
        ok, _ = evaluate_boundary_gate(m, baseline_above=None)
        assert ok is True

    def test_passes_when_above_count_holds(self):
        m = _probe_metrics([0.9] * 9, boundary_below=10, boundary_total=59)  # 49 above
        ok, _ = evaluate_boundary_gate(m, baseline_above=49)
        assert ok is True

    def test_fails_when_above_count_regresses(self):
        m = _probe_metrics([0.9] * 9, boundary_below=20, boundary_total=59)  # 39 above
        ok, detail = evaluate_boundary_gate(m, baseline_above=49)
        assert ok is False
        assert detail["n_above"] == 39


class TestExtractBaselines:
    def test_extract_ceiling_baseline(self):
        m = _probe_metrics([0.9] * 9, ceiling_rows=(("EGYPT/DL", 50, 100), ("GUINEA/DL", 20, 80)))
        base = extract_ceiling_baseline(m)
        assert base == {"EGYPT/DL": (50, 100), "GUINEA/DL": (20, 80)}

    def test_extract_boundary_baseline(self):
        m = _probe_metrics([0.9] * 9, boundary_below=10, boundary_total=59)
        assert extract_boundary_baseline(m) == 49


class TestCheckpointTracker:
    def test_eligible_epochs_filters_to_composite_pass(self):
        baseline = {f"d{i}": 0.3 for i in range(9)}
        tracker = CheckpointTracker(thresholds=GateThresholds(stability_window=1))
        m_good = _probe_metrics([0.9] * 9, threshold=0.5)
        m_bad = _probe_metrics([0.1] * 9, threshold=0.5)
        for epoch, m in [(1, m_good), (2, m_bad), (3, m_good)]:
            tracker.record_epoch(evaluate_gates(epoch, m, baseline, None, None, tracker.thresholds))
        assert tracker.eligible_epochs() == [1, 3]

    def test_latest_stable_requires_a_full_window(self):
        baseline = {f"d{i}": 0.3 for i in range(9)}
        thresholds = GateThresholds(stability_window=2)
        tracker = CheckpointTracker(thresholds=thresholds)
        m_good = _probe_metrics([0.9] * 9, threshold=0.5)
        m_bad = _probe_metrics([0.1] * 9, threshold=0.5)
        # epoch 1,2 good (stable pair) -- epoch 3 bad -- epoch 4 good but no epoch 5 yet to confirm
        for epoch, m in [(1, m_good), (2, m_good), (3, m_bad), (4, m_good)]:
            tracker.record_epoch(evaluate_gates(epoch, m, baseline, None, None, thresholds))
        assert tracker.latest_stable_epoch() == 1  # only confirmed stable window

    def test_latest_stable_none_when_nothing_eligible(self):
        baseline = {f"d{i}": 0.3 for i in range(9)}
        tracker = CheckpointTracker(thresholds=GateThresholds(stability_window=1))
        m_bad = _probe_metrics([0.1] * 9, threshold=0.5)
        tracker.record_epoch(evaluate_gates(1, m_bad, baseline, None, None, tracker.thresholds))
        assert tracker.latest_stable_epoch() is None

    def test_latest_stable_picks_the_later_of_two_valid_windows(self):
        baseline = {f"d{i}": 0.3 for i in range(9)}
        thresholds = GateThresholds(stability_window=2)
        tracker = CheckpointTracker(thresholds=thresholds)
        m_good = _probe_metrics([0.9] * 9, threshold=0.5)
        for epoch in range(1, 6):
            tracker.record_epoch(evaluate_gates(epoch, m_good, baseline, None, None, thresholds))
        assert tracker.latest_stable_epoch() == 4  # epochs 4,5 form the latest full window


class TestAverageStateDicts:
    def test_averages_floating_point_tensors(self):
        sd1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
        sd2 = {"w": torch.tensor([3.0, 4.0, 5.0])}
        out = average_state_dicts([sd1, sd2])
        assert torch.allclose(out["w"], torch.tensor([2.0, 3.0, 4.0]))

    def test_preserves_dtype(self):
        sd1 = {"w": torch.tensor([1.0, 2.0], dtype=torch.float16)}
        sd2 = {"w": torch.tensor([3.0, 4.0], dtype=torch.float16)}
        out = average_state_dicts([sd1, sd2])
        assert out["w"].dtype == torch.float16

    def test_non_floating_point_tensors_copied_not_averaged(self):
        sd1 = {"bn.num_batches_tracked": torch.tensor(10, dtype=torch.int64)}
        sd2 = {"bn.num_batches_tracked": torch.tensor(20, dtype=torch.int64)}
        out = average_state_dicts([sd1, sd2])
        assert out["bn.num_batches_tracked"].item() == 10  # copied from the first, not averaged

    def test_single_state_dict_returns_itself_effectively(self):
        sd = {"w": torch.tensor([1.0, 2.0, 3.0])}
        out = average_state_dicts([sd])
        assert torch.allclose(out["w"], sd["w"])

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            average_state_dicts([])

    def test_raises_on_mismatched_keys(self):
        sd1 = {"w": torch.tensor([1.0])}
        sd2 = {"v": torch.tensor([1.0])}
        with pytest.raises(ValueError):
            average_state_dicts([sd1, sd2])
