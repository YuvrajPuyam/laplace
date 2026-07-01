"""Tests for the decision-relevant fidelity metrics + the rho* locator + pre-registration,
all on synthetic ground truth we control (no GPU/physics needed)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.fidelity_metrics import (decision_flip_rate, srcc,
                                          treatment_effect_fidelity, _wilson)
from experiments.fidelity_sweep import (DEFAULT_PREREG, freeze_prereg, load_prereg,
                                        locate_rho_star, verify_prereg)


# ── treatment-effect fidelity ────────────────────────────────────────────────
def test_effect_agree_when_des_equals_phys():
    rng = np.random.default_rng(0)
    d_phys = 0.5 + rng.normal(0, 0.2, 200)        # clearly positive effect
    out = treatment_effect_fidelity(d_phys.copy(), d_phys.copy(), delta=0.05)
    assert out["rec_correct"] and out["label"] == "AGREE"
    assert out["decide_des"] == 1 and out["decide_phys"] == 1


def test_effect_disagree_on_opposite_sign():
    rng = np.random.default_rng(1)
    d_phys = 0.5 + rng.normal(0, 0.15, 200)
    out = treatment_effect_fidelity(-d_phys, d_phys, delta=0.05)
    assert out["label"] == "DISAGREE" and not out["rec_correct"]


def test_invented_effect_is_a_failure():
    # physics: no real effect (within delta band); DES: invents a large positive effect.
    rng = np.random.default_rng(2)
    d_phys = rng.normal(0, 0.01, 300)             # ~0, inside delta
    d_des = 0.5 + rng.normal(0, 0.1, 300)
    out = treatment_effect_fidelity(d_des, d_phys, delta=0.05)
    assert out["decide_phys"] == 0 and out["decide_des"] == 1
    assert not out["rec_correct"]                 # inventing an effect is wrong
    assert out["rel_eff_err"] is None             # ratio guarded (|E_phys| <= delta)


# ── decision-flip rate ───────────────────────────────────────────────────────
def test_flip_rate_counts_and_wilson():
    # 5 decisions, 3 configs each; flip exactly 2 by making DES prefer a worse config.
    decisions = []
    for j in range(5):
        phys = np.array([1.0, 2.0, 3.0])          # config 2 is best under physics
        des = phys.copy()
        if j < 2:                                 # flip: DES prefers config 0
            des = np.array([9.0, 2.0, 3.0])
        decisions.append((des, phys))
    out = decision_flip_rate(decisions)
    assert out["n_flips"] == 2 and out["flip_rate"] == pytest.approx(0.4)
    lo, hi = out["flip_rate_ci"]
    assert lo <= 0.4 <= hi


def test_near_tie_flip_has_low_regret():
    # DES flips to a config that is essentially tied under physics -> tiny regret.
    decisions = [(np.array([3.001, 3.0]), np.array([2.999, 3.0])) for _ in range(4)]
    out = decision_flip_rate(decisions)
    assert out["flip_rate"] == pytest.approx(1.0)        # always "flips"
    # ABSOLUTE regret is the honest "barely costs anything" measure. (Normalized regret
    # is ~1.0 here precisely because the whole decision is low-stakes: picking the worse
    # of two near-tied configs spends 100% of a tiny spread - which is why we lead with
    # absolute regret + flip rate, and read normalized regret only with the spread.)
    assert out["mean_regret"] < 0.05


def test_wilson_bounds():
    lo, hi = _wilson(0, 10)
    assert lo >= 0.0 and hi <= 1.0 and lo < hi


# ── SRCC: Pearson vs Spearman do different jobs ──────────────────────────────
def test_srcc_recovers_linear_correlation():
    rng = np.random.default_rng(3)
    des = rng.normal(0, 1, 40)
    phys = 2.0 * des + rng.normal(0, 0.3, 40)
    out = srcc(des, phys)
    assert out["pearson_srcc"] > 0.9
    lo, hi = out["pearson_ci"]
    assert lo <= out["pearson_srcc"] <= hi


def test_srcc_spearman_robust_to_monotone_nonlinearity():
    des = np.linspace(0.1, 3.0, 30)
    phys = np.exp(des)                             # monotone but very nonlinear
    out = srcc(des, phys)
    assert out["spearman"] > 0.99                  # ordering preserved perfectly
    assert out["pearson_srcc"] < out["spearman"]   # linear corr is lower


# ── rho* locator ─────────────────────────────────────────────────────────────
def test_rho_star_step_crossing():
    grid = [0.05, 0.1, 0.15, 0.2, 0.25]
    point = [0.0, 0.0, 0.0, 0.2, 0.3]             # breaks at 0.2
    lo = [-0.02, -0.02, -0.02, 0.15, 0.25]        # CI lower also past tau from 0.2
    out = locate_rho_star(grid, lo, point, tau=0.05)
    assert out["rho_star_lower"] == 0.2
    assert out["bracket"] == (0.15, 0.2)
    assert out["monotonic_spearman"] > 0.0


def test_rho_star_persistence_rejects_noisy_bin():
    grid = [0.05, 0.1, 0.15, 0.2, 0.25]
    point = [0.0, 0.2, 0.0, 0.0, 0.3]             # a spurious spike at 0.1
    lo = [-0.02, 0.10, -0.02, -0.02, 0.25]        # crosses at 0.1 but does NOT persist
    out = locate_rho_star(grid, lo, point, tau=0.05)
    assert out["rho_star_lower"] == 0.25          # only the persistent crossing counts


# ── pre-registration integrity ───────────────────────────────────────────────
def test_prereg_freeze_verify_roundtrip():
    frozen = freeze_prereg(DEFAULT_PREREG)
    assert verify_prereg(frozen)
    tampered = copy.deepcopy(frozen)
    tampered["decision_rule"]["failure_threshold_tau"] = 0.99
    assert not verify_prereg(tampered)            # any edit breaks the hash


def test_prereg_file_loads_and_is_intact():
    spec = load_prereg("experiments/prereg_physx_breakdown_v1.json")
    assert spec["preregistration_id"] == "physx_breakdown_v1"
    assert spec["decision_rule"]["failure_threshold_tau"] == 0.05
