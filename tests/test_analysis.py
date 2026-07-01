"""Tests for eval/analysis: calibration, power, and feedback-responsiveness, all on
synthetic ground truth we control. Pure numpy, deterministic."""

from __future__ import annotations

import numpy as np
import pytest

from eval.analysis.calibration import (bootstrap_ece_ci, ci_coverage,
                                       confidence_reliability)
from eval.analysis.power import (n_for_coverage_test, n_for_paired_accuracy,
                                 n_from_accuracy_gap)
from eval.analysis.responsiveness import responsiveness_test


# ── calibration ──────────────────────────────────────────────────────────────
def test_ci_coverage_recovers_nominal():
    n = 1000
    lo = np.zeros(n); hi = np.ones(n)
    gt = np.concatenate([np.full(900, 0.5), np.full(100, 2.0)])   # 900 in, 100 out
    out = ci_coverage(lo, hi, gt)
    assert out["coverage"] == pytest.approx(0.90)
    assert out["wilson"][0] <= 0.90 <= out["wilson"][1]
    assert abs(out["gap"]) < 1e-9


def test_confidence_reliability_calibrated_low_ece():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < conf).astype(float)   # perfectly calibrated
    out = confidence_reliability(conf, y)
    assert out["ece"] < 0.06


def test_confidence_reliability_miscalibrated_high_ece():
    conf = np.full(400, 0.95)
    y = np.array([1.0] * 200 + [0.0] * 200)              # acc 0.5 vs stated 0.95
    out = confidence_reliability(conf, y)
    assert out["ece"] == pytest.approx(0.45, abs=1e-6)


def test_bootstrap_ece_ci_brackets_point():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, 800)
    y = (rng.uniform(0, 1, 800) < conf).astype(float)
    out = bootstrap_ece_ci(conf, y, B=400, seed=2)
    assert out["ci"][0] <= out["ece"] <= out["ci"][1]


# ── power ────────────────────────────────────────────────────────────────────
def test_coverage_power_monotone_in_margin():
    big = n_for_coverage_test(0.05)      # small margin -> many records
    small = n_for_coverage_test(0.15)    # large margin -> few
    assert big > small > 0


def test_paired_accuracy_power():
    assert n_for_paired_accuracy(0.4, 0.1) > 0
    assert n_from_accuracy_gap(0.9, 0.7, p_disc=0.4) > 0
    with pytest.raises(ValueError):
        n_for_paired_accuracy(0.2, 0.2)   # no detectable effect


# ── feedback-responsiveness ──────────────────────────────────────────────────
def _blind(n, k, seed):
    rng = np.random.default_rng(seed)
    fixed = rng.integers(0, k, n)
    best = rng.integers(0, k, n)          # outcomes independent of the (fixed) choice
    return fixed, fixed.copy(), best


def _responsive(n, k, seed):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=(n, k))
    perm = np.array([rng.permutation(k) for _ in range(n)])
    permout = np.take_along_axis(true, perm, axis=1)
    rec = true.argmax(1)
    return rec, permout.argmax(1), rec.copy()


def test_blind_agent_is_flagged_blind():
    rec, rec_perm, best = _blind(300, 4, 0)
    out = responsiveness_test(rec, rec_perm, best_fed=best, n_perm=2000, k_options=4)
    assert out["T"] == 0.0                # permuting outcomes never changes the choice
    assert out["p_value"] > 0.2           # agreement no better than chance
    assert out["S"] == pytest.approx(0.25, abs=0.1)


def test_responsive_agent_is_flagged_responsive():
    rec, rec_perm, best = _responsive(300, 4, 1)
    out = responsiveness_test(rec, rec_perm, best_fed=best, n_perm=2000, k_options=4)
    assert out["S"] == 1.0                # argmax of true feed == true best, always
    assert out["p_value"] < 0.01          # far beyond chance
    assert out["T"] == pytest.approx(0.75, abs=0.1)   # ceiling 1 - 1/K = 0.75


def test_responsiveness_is_graded_monotone():
    k = 4
    ts = []
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        rng = np.random.default_rng(int(rho * 100))
        rb, rpb, bb = _blind(400, k, 7)
        rr, rpr, br = _responsive(400, k, 8)
        mask = rng.random(400) < rho
        rec = np.where(mask, rr, rb)
        rec_perm = np.where(mask, rpr, rpb)
        ts.append(responsiveness_test(rec, rec_perm, n_perm=1, k_options=k)["T"])
    assert all(b >= a - 0.05 for a, b in zip(ts, ts[1:])), ts   # T rises with responsiveness
