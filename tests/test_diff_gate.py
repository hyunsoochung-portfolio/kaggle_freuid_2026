"""Tests for scripts/analysis/diff_gate.py and the MC machinery it reuses from
official_score_reconciliation.py: MC reproducibility under a fixed seed, and the gate
threshold/verdict logic in isolation (no full 7,821-id pipeline -- small synthetic inputs, for
speed, matching what a CI-style pre-submission check needs to run fast).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from diff_gate import gate_verdict  # noqa: E402
from official_score_reconciliation import mc_expected_metrics  # noqa: E402


def _synthetic_inputs(seed: int = 0, n_present: int = 200, n_placeholder: int = 50):
    rng = np.random.default_rng(seed)
    p_fraud = rng.uniform(0.0, 1.0, size=n_present)
    score = rng.uniform(0.0, 1.0, size=n_present)
    return p_fraud, score, n_placeholder


class TestMCReproducibility:
    def test_same_seed_gives_identical_results(self):
        p_fraud, score, n_placeholder = _synthetic_inputs()
        r1 = mc_expected_metrics(p_fraud, score, 0.5, 0.42316, n_placeholder, n_draws=50, seed=7)
        r2 = mc_expected_metrics(p_fraud, score, 0.5, 0.42316, n_placeholder, n_draws=50, seed=7)
        assert r1 == r2  # exact dict equality -- every float must match bit-for-bit

    def test_different_seeds_give_different_draws(self):
        p_fraud, score, n_placeholder = _synthetic_inputs()
        r1 = mc_expected_metrics(p_fraud, score, 0.5, 0.42316, n_placeholder, n_draws=50, seed=1)
        r2 = mc_expected_metrics(p_fraud, score, 0.5, 0.42316, n_placeholder, n_draws=50, seed=2)
        # Different seeds should (almost certainly) not produce an identical mean to full
        # float precision -- if they did, the seed wouldn't actually be threading through.
        assert r1["freuid_mean"] != r2["freuid_mean"]

    def test_more_draws_reduces_standard_error(self):
        """Sanity check on the MC itself: std should not systematically grow with n_draws
        (it estimates a fixed population std), so std/sqrt(n) -- the standard error actually
        used by the gate -- must shrink as n_draws grows."""
        p_fraud, score, n_placeholder = _synthetic_inputs()
        r_small = mc_expected_metrics(
            p_fraud, score, 0.5, 0.42316, n_placeholder, n_draws=20, seed=3,
        )
        r_large = mc_expected_metrics(
            p_fraud, score, 0.5, 0.42316, n_placeholder, n_draws=200, seed=3,
        )
        se_small = r_small["freuid_std"] / np.sqrt(20)
        se_large = r_large["freuid_std"] / np.sqrt(200)
        assert se_large < se_small

    def test_reproducible_across_process_semantics(self):
        """Calling twice in immediate succession (simulating two separate script invocations
        with the same --seed) must reproduce the exact mean/std -- the property a
        pre-submission gate depends on to be trustworthy run-to-run."""
        p_fraud, score, n_placeholder = _synthetic_inputs(seed=99)
        results = [
            mc_expected_metrics(p_fraud, score, 0.5, 0.42316, n_placeholder, n_draws=30, seed=42)
            for _ in range(3)
        ]
        assert all(r == results[0] for r in results)


def _mc_df_row(scheme: str, role: str, freuid_mean: float, freuid_std: float = 0.0004) -> dict:
    return {
        "scheme": scheme, "role": role, "freuid_mean": freuid_mean, "freuid_std": freuid_std,
        "audet_mean": 0.1, "audet_std": 0.001,
        "apcer_at_bpcer_mean": 0.5, "apcer_at_bpcer_std": 0.001,
    }


class TestGateVerdict:
    def _mc_df(self, deltas: dict[str, float], ref_freuid: float = 0.5) -> pd.DataFrame:
        rows = []
        for scheme, delta in deltas.items():
            rows.append(_mc_df_row(scheme, "reference", ref_freuid))
            rows.append(_mc_df_row(scheme, "candidate", ref_freuid + delta))
        return pd.DataFrame(rows)

    def test_pass_when_candidate_strictly_better_everywhere(self):
        mc_df = self._mc_df({"blended": -0.05, "zone_only": -0.05, "verdict_only": -0.05})
        gate = gate_verdict(mc_df, n_draws=200)
        assert gate["verdict"] == "PASS"
        assert gate["n_pass"] == 3

    def test_fail_when_candidate_significantly_worse_in_blended(self):
        # A large delta (0.05) against a small std (0.0004/sqrt(200) SE) is many std -- a real,
        # significant regression in the headline scheme.
        mc_df = self._mc_df({"blended": 0.05, "zone_only": 0.05, "verdict_only": 0.05})
        gate = gate_verdict(mc_df, n_draws=200)
        assert gate["verdict"] == "FAIL"
        assert gate["n_pass"] == 0

    def test_warn_when_blended_passes_but_others_dont(self):
        mc_df = self._mc_df({"blended": -0.05, "zone_only": 0.05, "verdict_only": 0.05})
        gate = gate_verdict(mc_df, n_draws=200)
        assert gate["verdict"] == "WARN"
        assert gate["n_pass"] == 1

    def test_noise_level_delta_does_not_spuriously_fail(self):
        """A delta far smaller than its standard error (e.g. two independent MC estimates of
        the SAME underlying distribution) must PASS, not FAIL -- regression test for the bug
        this module's own self-comparison caught: a strict delta<=0 check fails ~50% of the
        time on pure noise, which is not an acceptable gate for a truly-tied candidate."""
        mc_df = self._mc_df({"blended": 1e-6, "zone_only": 1e-6, "verdict_only": -1e-6})
        gate = gate_verdict(mc_df, n_draws=200)
        assert gate["verdict"] == "PASS"

    def test_per_scheme_table_has_expected_columns(self):
        mc_df = self._mc_df({"blended": -0.01, "zone_only": -0.01, "verdict_only": -0.01})
        gate = gate_verdict(mc_df, n_draws=200)
        assert set(gate["per_scheme"].columns) >= {
            "scheme", "reference_freuid", "candidate_freuid", "delta", "delta_se", "delta_z",
            "candidate_not_worse",
        }
