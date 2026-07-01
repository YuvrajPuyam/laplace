"""Tests for the CPU halves of the PhysX-validation runner (renderer/physx_run).

`extract` (DES rollout -> per-robot leg plan + constant-speed DES times) and `compare`
(-> the "agree within X%" agreement report) are CPU-only and fully testable here. The
GPU `drive` half is isolated and exercised on the cluster, so it is NOT imported by these
tests - and we assert that, so a broken Isaac install can never break this suite.
"""

from __future__ import annotations

import math
import sys

from renderer.physx_run import _load_config, compare, extract


def _plan(scenario: str = "baseline_small", seed: int = 0, window: float | None = None):
    config, name = _load_config(scenario, None)
    return extract(config, name, seed, window_min=window, include_warmup=False)


def test_extract_wellformed_plan():
    plan = _plan()
    assert plan["n_legs"] > 0 and plan["n_robots"] >= 1
    for leg in plan["legs"]:
        assert len(leg["a"]) == 2 and len(leg["b"]) == 2
        assert leg["des_time_s"] >= 0.0
        d = math.hypot(leg["b"][0] - leg["a"][0], leg["b"][1] - leg["a"][1])
        if leg["speed_mps"] > 0:                       # des time == constant-speed traversal
            assert abs(leg["des_time_s"] - d / leg["speed_mps"]) < 1e-3
    # des_legs index keys are exactly the (amr, leg) pairs in legs
    keyset = {(leg["amr"], leg["leg"]) for leg in plan["legs"]}
    assert {(a, lg) for a, lg, _ in plan["des_legs"]} == keyset


def test_extract_deterministic():
    # same (config, seed) -> identical leg plan (CRN / determinism non-negotiable)
    assert _plan(seed=1)["des_legs"] == _plan(seed=1)["des_legs"]


def test_window_caps_legs():
    assert _plan(window=1.0)["n_legs"] <= _plan(window=None)["n_legs"]


def test_compare_roundtrips_headline():
    plan = _plan()
    phys = {"phys_legs": [[a, lg, round(t * 1.05, 4)] for a, lg, t in plan["des_legs"]],
            "phys_metrics": {}}
    out = compare(plan, phys)                          # physics uniformly 5% slower
    assert out["legs"]["n_legs"] > 0
    assert 4.9 <= out["legs"]["agree_within_pct"] <= 5.1
    assert out["legs"]["signed_mean_rel_pct"] > 0      # signed bias: physics slower
    assert out["legs"]["gridlock_legs"] == 0           # uniform 5% slower -> no gridlock tail
    assert "agree on a typical leg" in out["headline"]
    assert out["coverage"]["compared"] <= plan["n_legs"]


def test_compare_handles_partial_coverage():
    # a GPU run that only finished some legs still yields an honest number over the overlap
    plan = _plan()
    half = plan["des_legs"][: max(1, len(plan["des_legs"]) // 2)]
    phys = {"phys_legs": [[a, lg, t] for a, lg, t in half]}   # exact match on the overlap
    out = compare(plan, phys)
    assert out["legs"]["agree_within_pct"] == 0.0      # identical -> 0% error
    assert out["coverage"]["compared"] <= len(half)


def test_cpu_halves_need_no_isaac():
    _plan()                                            # exercise extract
    assert "isaacsim" not in sys.modules               # never pulled the GPU dependency
