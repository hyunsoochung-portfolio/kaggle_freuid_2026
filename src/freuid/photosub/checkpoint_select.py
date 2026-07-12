"""Checkpoint-selection policy for photosub_v1: gate-composite eligibility, latest-stable
tiebreak, and optional last-k weight averaging.

No prior spec existed for this in the repo (checked: no reference to "last-k averaging",
"gate-composite", or "latest-stable" anywhere before this file) -- designed and implemented here
per docs/photosub_v1_spec.md's pre-registered gates, item 3.

Design, most-to-least novel:

1. **Gate-composite eligibility** (``evaluate_gates``): a pure function of one epoch's already-
   computed probe metrics (from ``freuid.photosub.probes.run_probe_hooks`` +
   ``threshold_watch``) plus the frozen finetune_v0-era deep-9 baseline logits -- no file I/O,
   no side effects, easy to unit test with synthetic inputs. Three gates, ALL must pass for an
   epoch to be "eligible":
     - deep9: majority (>=5/9) of the deep-9 ids score >= this epoch's present-calibrated
       threshold (``probe_threshold_score``, exact -- real val labels, no imputation), AND no
       id's current logit falls below its OWN frozen finetune_v0 baseline logit (regression
       guard, checked per-id not just in aggregate).
     - ceiling: no per-template above-threshold rate in ``probe_ceiling_by_template`` drops by
       more than ``ceiling_drop_tolerance`` (relative) from this RUN's own epoch-1 baseline for
       that template -- the APCER guard photosub_v0 needed (see docs/photosub_v1_spec.md).
     - boundary: the boundary-59 above-threshold count does not drop below this run's own
       epoch-1 baseline count.
   Epoch-1 is used as the in-run baseline (not an external finetune_v0 number) because these
   specific per-template/boundary-above-threshold metrics are new instrumentation with no
   finetune_v0-era measurement to compare against -- consistent with the existing clean_floor/
   ceiling mean-logit guards' own convention of tracking WITHIN one run, not against an
   external baseline.

2. **Latest-stable** (``CheckpointTracker.latest_stable_epoch``): the latest epoch E such that E
   and every epoch in [E, E + stability_window) are ALL gate-composite-eligible -- avoids
   checkpointing on a single noisy good epoch surrounded by failures (photosub_v0's own deep-9
   per-id logits "oscillated substantially epoch to epoch before this pattern stabilized" per
   docs/technical_report.md, so a one-epoch eligibility spike is a real, observed risk here, not
   a hypothetical one).

3. **Last-k averaging** (``average_state_dicts``): optional SWA-style arithmetic mean of the
   last K eligible checkpoints' weights -- offered as an alternative candidate, not a
   replacement for latest-stable; the caller (train.py) decides whether to save it and how to
   compare it against the latest-stable single checkpoint (e.g. via a fresh probe pass before
   final selection).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class GateThresholds:
    deep9_majority_frac: float = 0.5
    ceiling_drop_tolerance: float = 0.20  # relative drop tolerated per template
    stability_window: int = 2


@dataclass
class GateResult:
    epoch: int
    deep9_pass: bool
    deep9_detail: dict
    ceiling_pass: bool
    ceiling_detail: dict
    boundary_pass: bool
    boundary_detail: dict

    @property
    def composite_pass(self) -> bool:
        return self.deep9_pass and self.ceiling_pass and self.boundary_pass


def evaluate_deep9_gate(
    probe_metrics: dict, deep9_baseline_logits: dict[str, float],
    majority_frac: float = 0.5,
) -> tuple[bool, dict]:
    """``probe_metrics`` is ``run_probe_hooks``'s return dict (needs ``probe_threshold_score``
    and ``probe_deep_detail``). ``deep9_baseline_logits`` maps id -> finetune_v0-era baseline
    logit (the frozen ``logit`` column already in data/probes/missed_frauds_deep_ids.csv)."""
    threshold_score = probe_metrics["probe_threshold_score"]
    detail = probe_metrics["probe_deep_detail"]
    n = len(detail)
    if n == 0 or threshold_score is None:
        return False, {"reason": "no deep-9 detail or no threshold score", "n": n}

    n_above = sum(1 for d in detail if d["score"] >= threshold_score)
    below_baseline = [
        d["id"] for d in detail
        if d["id"] in deep9_baseline_logits and d["logit"] < deep9_baseline_logits[d["id"]]
    ]
    majority_ok = n_above >= majority_frac * n
    no_regression = len(below_baseline) == 0
    detail_out = {
        "n_above": n_above, "n_total": n, "majority_ok": majority_ok,
        "below_baseline_ids": below_baseline, "no_regression": no_regression,
        "threshold_score": threshold_score,
    }
    return (majority_ok and no_regression), detail_out


def evaluate_ceiling_gate(
    probe_metrics: dict, baseline_by_template: dict[str, tuple[int, int]] | None,
    drop_tolerance: float = 0.20,
) -> tuple[bool, dict]:
    """``baseline_by_template`` maps template -> (n_above, n_total) from this RUN's epoch-1
    (None on epoch 1 itself, when there's nothing to compare against yet -- passes trivially,
    since a first-epoch checkpoint being gate-composite-eligible is not a realistic outcome
    anyway given LLRD warmup, per CLAUDE.md)."""
    by_template = probe_metrics.get("probe_ceiling_by_template")
    if by_template is None or baseline_by_template is None:
        return True, {"reason": "no per-template data or no baseline yet (epoch 1)"}

    drops = {}
    for _, row in by_template.iterrows():
        template = row["template"]
        n_above, n_total = int(row["n_above"]), int(row["n_total"])
        base_above, base_total = baseline_by_template.get(template, (0, 0))
        if base_total == 0 or base_above == 0:
            continue
        base_rate = base_above / base_total
        cur_rate = n_above / max(1, n_total)
        rel_drop = (base_rate - cur_rate) / base_rate if base_rate > 0 else 0.0
        drops[template] = {
            "baseline_rate": base_rate, "current_rate": cur_rate, "relative_drop": rel_drop,
        }
    materially_dropped = {t: d for t, d in drops.items() if d["relative_drop"] > drop_tolerance}
    return len(materially_dropped) == 0, {
        "drops": drops, "materially_dropped": list(materially_dropped),
    }


def evaluate_boundary_gate(
    probe_metrics: dict, baseline_above: int | None,
) -> tuple[bool, dict]:
    """``run_probe_hooks`` reports ``probe_boundary_below_threshold`` (still missed) +
    ``probe_boundary_total`` -- "above" (caught) is the complement."""
    n_below = probe_metrics.get("probe_boundary_below_threshold")
    n_total = probe_metrics.get("probe_boundary_total")
    if n_below is None or n_below < 0 or n_total is None or baseline_above is None:
        return True, {"reason": "no threshold-watch data or no baseline yet (epoch 1)"}
    n_above = n_total - n_below
    ok = n_above >= baseline_above
    return ok, {"n_above": n_above, "baseline_above": baseline_above}


def extract_ceiling_baseline(probe_metrics: dict) -> dict[str, tuple[int, int]] | None:
    """Pulls (n_above, n_total) per template out of a completed epoch's
    ``probe_ceiling_by_template`` -- the caller (train.py) calls this once, on epoch 1, and
    threads the result into every later epoch's ``evaluate_gates`` call as ``ceiling_baseline``."""
    by_template = probe_metrics.get("probe_ceiling_by_template")
    if by_template is None:
        return None
    return {
        row["template"]: (int(row["n_above"]), int(row["n_total"]))
        for _, row in by_template.iterrows()
    }


def extract_boundary_baseline(probe_metrics: dict) -> int | None:
    n_below = probe_metrics.get("probe_boundary_below_threshold")
    n_total = probe_metrics.get("probe_boundary_total")
    if n_below is None or n_total is None:
        return None
    return n_total - n_below


def evaluate_gates(
    epoch: int, probe_metrics: dict, deep9_baseline_logits: dict[str, float],
    ceiling_baseline: dict[str, tuple[int, int]] | None, boundary_baseline: int | None,
    thresholds: GateThresholds | None = None,
) -> GateResult:
    thresholds = thresholds if thresholds is not None else GateThresholds()
    deep9_pass, deep9_detail = evaluate_deep9_gate(
        probe_metrics, deep9_baseline_logits, thresholds.deep9_majority_frac,
    )
    ceiling_pass, ceiling_detail = evaluate_ceiling_gate(
        probe_metrics, ceiling_baseline, thresholds.ceiling_drop_tolerance,
    )
    boundary_pass, boundary_detail = evaluate_boundary_gate(probe_metrics, boundary_baseline)
    return GateResult(
        epoch=epoch, deep9_pass=deep9_pass, deep9_detail=deep9_detail,
        ceiling_pass=ceiling_pass, ceiling_detail=ceiling_detail,
        boundary_pass=boundary_pass, boundary_detail=boundary_detail,
    )


@dataclass
class CheckpointTracker:
    """Accumulates per-epoch GateResults across a training run and answers the two selection
    questions: which epochs are eligible, and which eligible epoch is "latest-stable"."""

    thresholds: GateThresholds = field(default_factory=GateThresholds)
    history: list[GateResult] = field(default_factory=list)

    def record_epoch(self, result: GateResult) -> None:
        self.history.append(result)

    def eligible_epochs(self) -> list[int]:
        return [r.epoch for r in self.history if r.composite_pass]

    def latest_stable_epoch(self) -> int | None:
        """The latest epoch E such that E and the next ``stability_window - 1`` epochs are ALL
        eligible. Returns None if no such window exists yet (including: training hasn't run
        enough epochs past a candidate to confirm its stability)."""
        w = self.thresholds.stability_window
        by_epoch = {r.epoch: r.composite_pass for r in self.history}
        if not by_epoch:
            return None
        max_epoch = max(by_epoch)
        candidates = [
            e for e in by_epoch
            if e + w - 1 <= max_epoch and all(by_epoch.get(e + i, False) for i in range(w))
        ]
        return max(candidates) if candidates else None


def average_state_dicts(state_dicts: list[dict]) -> dict:
    """Arithmetic mean of a list of model state_dicts (SWA-style) -- all must share identical
    keys and shapes (asserted). Non-floating-point tensors (e.g. integer buffers) are copied
    from the FIRST state_dict unchanged rather than averaged, since averaging e.g. a batch-norm
    ``num_batches_tracked`` counter would be meaningless."""
    if not state_dicts:
        raise ValueError("average_state_dicts needs at least one state_dict")
    keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != keys:
            raise ValueError(f"state_dict {i} has different keys than state_dict 0")

    out = {}
    for k in keys:
        tensors = [sd[k] for sd in state_dicts]
        if torch.is_floating_point(tensors[0]):
            stacked = torch.stack([t.float() for t in tensors], dim=0)
            out[k] = stacked.mean(dim=0).to(tensors[0].dtype)
        else:
            out[k] = tensors[0].clone()
    return out
