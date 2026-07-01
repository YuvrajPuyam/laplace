"""Tests for the DES-vs-PhysX agreement core (experiments/physx_compare.py).

Pure math, no Isaac. The Isaac/PhysX run that produces phys_legs is GPU-gated and
unverifiable here; this locks down the 'agree within X%' computation so the headline
number is trustworthy once the Isaac side lands.
"""

from __future__ import annotations

import math

from experiments.physx_compare import (
    leg_agreement,
    metric_agreement,
    relative_errors,
    summarize,
)


def test_relative_errors_shared_positive_keys_only():
    des = {("r", 0): 10.0, ("r", 1): 0.0, ("r", 2): 20.0}
    phys = {("r", 0): 11.0, ("r", 1): 5.0, ("r", 2): 19.0}   # ("r",3) absent
    rel = relative_errors(des, phys)
    assert set(rel) == {("r", 0), ("r", 2)}                 # 0-time leg dropped
    assert math.isclose(rel[("r", 0)], 0.1)
    assert math.isclose(rel[("r", 2)], -0.05)


def test_leg_agreement_uniform_bias():
    # physics uniformly 6% slower -> mean abs rel = signed mean = 6%, tight CI
    des = {("r0", i): 10.0 + i for i in range(30)}
    phys = {k: v * 1.06 for k, v in des.items()}
    a = leg_agreement(des, phys)
    assert a["n_legs"] == 30
    assert math.isclose(a["mean_abs_rel_pct"], 6.0, abs_tol=1e-6)
    assert math.isclose(a["signed_mean_rel_pct"], 6.0, abs_tol=1e-6)
    assert a["agree_within_pct"] == 6.0
    lo, hi = a["ci90_abs_rel_pct"]
    assert math.isclose(lo, 6.0, abs_tol=1e-6) and math.isclose(hi, 6.0, abs_tol=1e-6)


def test_leg_agreement_signed_vs_absolute():
    # symmetric errors cancel in the signed mean but not the absolute mean
    des = {("a", 0): 10.0, ("a", 1): 10.0}
    phys = {("a", 0): 11.0, ("a", 1): 9.0}                  # +10% and -10%
    a = leg_agreement(des, phys)
    assert math.isclose(a["signed_mean_rel_pct"], 0.0, abs_tol=1e-9)
    assert math.isclose(a["mean_abs_rel_pct"], 10.0, abs_tol=1e-9)


def test_leg_agreement_empty():
    a = leg_agreement({}, {})
    assert a["n_legs"] == 0 and a["agree_within_pct"] is None


def test_metric_agreement_relative_delta():
    m = metric_agreement(120.0, 113.0)
    assert math.isclose(m["rel_delta_pct"], (113 - 120) / 120 * 100, abs_tol=0.01)
    assert metric_agreement(0.0, 5.0)["rel_delta_pct"] is None


def test_summarize_headline_direction():
    des = {("r0", i): 10.0 + i for i in range(20)}
    phys = {k: v * 1.06 for k, v in des.items()}
    out = summarize(des, phys,
                    des_metrics={"throughput_orders_per_hr": 120.0},
                    phys_metrics={"throughput_orders_per_hr": 113.0})
    assert "within 6.0%" in out["headline"]
    assert "physics slower" in out["headline"]
    assert out["metrics"]["throughput_orders_per_hr"]["rel_delta_pct"] < 0
