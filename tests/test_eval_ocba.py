"""Tests for the OCBA baseline arm (eval/baselines.ocba) — research item R1.

The allocation RULE is pure, so it's unit-tested directly against the closed-form
OCBA behavior (concentrate on critical designs). The end-to-end arm is checked on
the cached braess GT for budget discipline, GT-prefix fairness, and correctness.
"""

from __future__ import annotations

from eval import baselines as B
from eval import run_eval
from eval.candidates import DEV_DECISIONS
from eval.gt_sweep import load_or_compute


# ── the OCBA allocation rule (pure, no sim) ───────────────────────────────────

def test_alloc_sums_to_one_and_best_leads():
    props = B._ocba_alloc(means=[10.0, 9.0, 8.0], sds=[1.0, 1.0, 1.0])
    assert abs(sum(props) - 1.0) < 1e-9
    assert all(p >= 0 for p in props)
    assert props[0] == max(props)                 # the best design gets the most


def test_alloc_closer_competitor_gets_more_than_far_one():
    # equal variance, best=idx0: competitor at gap 1 should outrank competitor at gap 2
    props = B._ocba_alloc(means=[10.0, 9.0, 8.0], sds=[1.0, 1.0, 1.0])
    assert props[1] > props[2]


def test_alloc_higher_variance_competitor_gets_more():
    # same gap to best, but idx1 is noisier than idx2 → idx1 gets more budget
    props = B._ocba_alloc(means=[10.0, 9.0, 9.0], sds=[1.0, 2.0, 0.5])
    assert props[1] > props[2]


def test_alloc_degenerate_is_valid_distribution():
    props = B._ocba_alloc(means=[5.0, 5.0, 5.0], sds=[1.0, 1.0, 1.0])
    assert abs(sum(props) - 1.0) < 1e-9
    assert all(p > 0 for p in props)             # no crash, finite, positive


def test_alloc_single_candidate():
    assert B._ocba_alloc([3.0], [1.0]) == [1.0]


# ── end-to-end on the cached braess GT (small budgets, fast) ──────────────────

def test_ocba_picks_optimum_within_budget():
    decision = DEV_DECISIONS["braess_dev"]
    gt = load_or_compute(decision, list(range(24)))           # cached, no sim
    ad = B.ocba(decision, budget=12, gt=gt)                    # total = 6*2 = 12 rollouts
    assert ad.arm == "ocba"
    assert ad.picked == "mid_cross_aisle"                      # the GT optimum
    assert ad.rollouts_used <= 12                             # never overspends
    # per-candidate spend never exceeds the GT resolution; seeds are a GT prefix
    assert all(s in gt["seeds"] for s in ad.seeds_used)
    assert ad.ci90 is None                                    # OCBA states no CI


def test_ocba_requires_gt():
    import pytest
    with pytest.raises(ValueError):
        B.ocba(DEV_DECISIONS["braess_dev"], budget=12, gt=None)


def test_run_eval_includes_ocba_arm():
    table, graded = run_eval.run(["braess_dev"], seeds=24, budget=12)
    assert "ocba" in graded and "grid_search" in graded       # both CPU arms run
    g = graded["ocba"][0]
    assert g["picked"] == "mid_cross_aisle" and g["correct"] is True
    assert g["rollouts"] <= 12
