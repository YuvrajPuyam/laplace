"""Guard tests for the eval fairness invariant (spec §5.4 H2).

The three-way ablation only means something if every arm is compared at EQUAL
budget against the ground truth, drawing CRN seeds from a shared prefix of the
GT seed list and never sampling more seeds per candidate than the GT itself did.

The bug these tests lock out: grid_search used range(0, budget//n) seeds (~100
at the default budget) while the GT used 24 — so brute force trivially re-derived
the ground truth on a seed-count artifact rather than on strategy, and "agent
beats uniform spend" was decided before either arm ran. See docs/HANDOFF.md §3.
"""

from __future__ import annotations

from eval import run_eval
from eval.baselines import (
    DEFAULT_GT_SEEDS,
    ArmDecision,
    fair_seed_prefix,
    grid_search,
)
from eval.candidates import DEV_DECISIONS


# ── pure fairness logic (no sim) ──────────────────────────────────────────────

def test_prefix_caps_at_gt_resolution():
    """A generous budget cannot make an arm out-resolve the ground truth."""
    gt = list(range(24))
    seeds = fair_seed_prefix(budget=200, n_candidates=2, gt_seeds=gt)
    assert seeds == list(range(24))          # 100/candidate requested → capped to 24
    assert len(seeds) <= len(gt)


def test_prefix_is_a_true_prefix_not_range_zero():
    """Seeds are a prefix of the *actual* GT list, even if it doesn't start at 0
    or isn't contiguous — so the comparison stays CRN-paired with the GT."""
    gt = [5, 6, 7, 8, 9, 10, 11, 12]
    seeds = fair_seed_prefix(budget=8, n_candidates=2, gt_seeds=gt)
    assert seeds == [5, 6, 7, 8]             # prefix of gt, NOT [0,1,2,3]
    assert seeds == gt[: len(seeds)]


def test_prefix_respects_budget_when_under_gt():
    """When the budget fits inside the GT seed set, the arm spends exactly it."""
    gt = list(range(24))
    seeds = fair_seed_prefix(budget=20, n_candidates=2, gt_seeds=gt)
    assert seeds == list(range(10))          # 20 // 2 = 10, well under 24


def test_prefix_is_at_least_one_seed():
    seeds = fair_seed_prefix(budget=1, n_candidates=5, gt_seeds=list(range(24)))
    assert seeds == [0]                       # max(1, 1//5) = 1


def test_prefix_defaults_to_dev_gt_count():
    """A direct call with no GT list falls back to the dev GT seed count."""
    seeds = fair_seed_prefix(budget=1000, n_candidates=2, gt_seeds=None)
    assert seeds == list(range(DEFAULT_GT_SEEDS))


# ── grid_search end to end honors the invariant ───────────────────────────────

def test_grid_search_cannot_out_resolve_the_gt():
    """With a huge budget but a 6-seed GT, grid_search must spend exactly the
    6-seed prefix (12 rollouts over 2 candidates) — not 100/candidate."""
    decision = DEV_DECISIONS["braess_dev"]
    gt_seeds = list(range(6))
    ad = grid_search(decision, budget=200, gt_seeds=gt_seeds)

    assert isinstance(ad, ArmDecision)
    assert ad.seeds_used == gt_seeds                 # shared prefix, capped
    assert ad.rollouts_used == len(gt_seeds) * len(decision.candidates) == 12
    assert ad.picked in {c.label for c in decision.candidates}
    assert "capped" in ad.notes                      # surfaced, not silent


def test_grid_search_seeds_are_subset_of_gt():
    """Under budget, the seeds spent are a strict prefix of the GT seed list."""
    decision = DEV_DECISIONS["braess_dev"]
    gt_seeds = list(range(24))
    ad = grid_search(decision, budget=8, gt_seeds=gt_seeds)
    assert ad.seeds_used == [0, 1, 2, 3]             # 8 // 2 candidates = 4
    assert set(ad.seeds_used) <= set(gt_seeds)
    assert ad.rollouts_used == 8


# ── run_eval threads the GT seed list to every arm ────────────────────────────

def test_run_eval_grades_grid_at_capped_budget():
    """run_eval hands each arm the loaded GT's seed list; the grid arm's spend is
    its fair prefix, and it still recovers the cached braess_dev optimum."""
    table, graded = run_eval.run(["braess_dev"], seeds=24, budget=12)
    g = graded["grid_search"][0]
    assert g["rollouts"] == 12                        # 12 // 2 = 6 seeds/candidate
    assert g["gt_optimum"] == "mid_cross_aisle"
    assert g["picked"] == g["gt_optimum"]             # fair fix preserves quality
    assert g["correct"] is True
    assert table["grid_search"]["total_rollouts"] == 12
