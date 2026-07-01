"""DES-vs-PhysX agreement — the "agree within X%" verification (Option 3, CPU side).

This is the pure, testable core of the authoritative-Isaac check (docs/critique-
response.md §3): given the fast-DES leg timings and the PhysX-measured leg timings
for the SAME plan (same robots, same legs, same CRN seed), compute how well the
fast sim's kinematic abstraction agrees with full physics, with a bootstrap CI.

The Isaac/PhysX side (renderer/physx_run.py, GPU-gated) drives the robots and
writes per-leg arrival times; this module consumes them. Keeping the agreement
MATH separate and unit-tested means the headline number is trustworthy regardless
of the (unverifiable-here) Isaac run.

FAIR-COMPARISON PROTOCOL (must hold for the number to mean anything):
  1. Same plan: PhysX replays the DES's decisions (which robot, which leg, when),
     so any timing gap is the kinematic-vs-dynamics gap, not a different policy.
  2. Frozen tuning: PhysX drive params (max accel, wheel/contact) are calibrated on
     a HELD-OUT config and frozen BEFORE the validated run — otherwise "agreement"
     measures the tuning, not the abstraction.
  3. Honest direction: report the gap whichever way it goes. If physics weakens the
     effect, say so — that is a real result about the abstraction.

This is sim-vs-SIM verification, NOT sim-to-real. PhysX is a higher-fidelity model,
not ground truth.
"""

from __future__ import annotations

import numpy as np


def relative_errors(des: dict, phys: dict) -> dict[tuple, float]:
    """Signed relative error (phys - des)/des per leg, over the shared keys with a
    positive DES time. Keys identify a leg, e.g. (robot_id, leg_index)."""
    return {k: (phys[k] - des[k]) / des[k]
            for k in sorted(set(des) & set(phys)) if des.get(k, 0) > 0}


def _bootstrap_ci(values: np.ndarray, fn, n_boot: int, seed: int,
                  lo: float = 5, hi: float = 95) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    stats = fn(values[idx], axis=1)
    return (float(np.percentile(stats, lo)), float(np.percentile(stats, hi)))


def leg_agreement(des: dict, phys: dict, *, n_boot: int = 2000,
                  seed: int = 12345, gridlock_factor: float = 3.0) -> dict:
    """Per-leg travel-time agreement between DES and PhysX.

    The per-leg error distribution is HEAVY-TAILED: most legs agree closely (free
    flow), but a few physically gridlock — a robot, as a solid body, blocks behind
    another at a choke point for many seconds, which the DES free-flow edge model
    cannot represent at all. A plain mean is dominated by that tail and misreports the
    typical agreement. So the headline is the MEDIAN abs relative error (robust), with
    the gridlock tail surfaced SEPARATELY as its own fidelity-gap finding.

    Returns the median + mean abs relative error, the free-flow fraction (legs within
    20%), the signed median (systematic accel lag: + = physics slower), the gridlock
    count/fraction (legs ≥ ``gridlock_factor``× the DES time), the MAE in time units,
    and a bootstrap CI90 on the MEDIAN abs relative error."""
    rel = relative_errors(des, phys)
    if not rel:
        return {"n_legs": 0, "median_abs_rel_pct": None, "mean_abs_rel_pct": None,
                "signed_median_rel_pct": None, "signed_mean_rel_pct": None,
                "frac_within_20pct": None, "p90_abs_rel_pct": None,
                "gridlock_legs": 0, "gridlock_frac": None,
                "mae_time": None, "ci90_abs_rel_pct": [None, None],
                "agree_within_pct": None}
    keys = list(rel)
    r = np.array([rel[k] for k in keys], float)
    abs_r = np.abs(r)
    diffs = np.array([phys[k] - des[k] for k in keys], float)
    ratio = np.array([phys[k] / des[k] for k in keys], float)
    n_grid = int(np.count_nonzero(ratio >= gridlock_factor))
    ci = _bootstrap_ci(abs_r, lambda a, axis: np.median(a, axis=axis) * 100, n_boot, seed)
    return {
        "n_legs": len(keys),
        "median_abs_rel_pct": round(float(np.median(abs_r)) * 100, 3),
        "mean_abs_rel_pct": round(float(abs_r.mean()) * 100, 3),
        "signed_median_rel_pct": round(float(np.median(r)) * 100, 3),
        "signed_mean_rel_pct": round(float(r.mean()) * 100, 3),
        "frac_within_20pct": round(float(np.mean(abs_r <= 0.20)), 3),
        "p90_abs_rel_pct": round(float(np.percentile(abs_r, 90)) * 100, 3),
        "gridlock_legs": n_grid,
        "gridlock_frac": round(n_grid / len(keys), 3),
        "gridlock_factor": gridlock_factor,
        "mae_time": round(float(np.abs(diffs).mean()), 4),
        "ci90_abs_rel_pct": [round(ci[0], 3), round(ci[1], 3)],
        "agree_within_pct": round(float(np.median(abs_r)) * 100, 1),
    }


def metric_agreement(des_value: float, phys_value: float) -> dict:
    """Agreement on an aggregate metric (e.g. throughput, p95): signed relative
    delta of the PhysX value vs the DES value."""
    if des_value == 0:
        return {"des": des_value, "phys": phys_value, "rel_delta_pct": None}
    return {"des": round(float(des_value), 4), "phys": round(float(phys_value), 4),
            "rel_delta_pct": round((phys_value - des_value) / des_value * 100, 2)}


def summarize(des_legs: dict, phys_legs: dict, *,
              des_metrics: dict | None = None, phys_metrics: dict | None = None,
              n_boot: int = 2000, seed: int = 12345) -> dict:
    """The full agreement report. ``*_metrics`` are optional aggregate dicts (same
    keys) such as {"throughput_orders_per_hr": ..., "p95_order_latency_min": ...}."""
    out = {"legs": leg_agreement(des_legs, phys_legs, n_boot=n_boot, seed=seed)}
    if des_metrics and phys_metrics:
        out["metrics"] = {k: metric_agreement(des_metrics[k], phys_metrics[k])
                          for k in sorted(set(des_metrics) & set(phys_metrics))}
    lg = out["legs"]
    if lg["agree_within_pct"] is not None:
        out["headline"] = (
            f"fast-sim and full-physics PhysX agree on a typical leg to within "
            f"{lg['agree_within_pct']}% (median abs error, CI90 "
            f"{lg['ci90_abs_rel_pct'][0]}–{lg['ci90_abs_rel_pct'][1]}%; "
            f"{lg['frac_within_20pct'] * 100:.0f}% of n={lg['n_legs']} legs within 20%); "
            f"systematic bias {lg['signed_median_rel_pct']:+}% (physics "
            f"{'slower' if lg['signed_median_rel_pct'] > 0 else 'faster'}), the "
            f"contact drag and finite acceleration the kinematic DES omits. Separately, "
            f"physical aisle-blocking "
            f"gridlocks {lg['gridlock_legs']}/{lg['n_legs']} legs "
            f"({lg['gridlock_frac'] * 100:.0f}%, ≥{lg['gridlock_factor']:g}× the DES "
            f"time) — congestion the free-flow DES cannot represent.")
    return out


if __name__ == "__main__":  # tiny manual demo with synthetic data
    des = {("r0", i): 10.0 + i for i in range(20)}
    phys = {k: v * 1.06 for k, v in des.items()}      # physics ~6% slower (accel)
    import json
    print(json.dumps(summarize(des, phys,
          des_metrics={"throughput_orders_per_hr": 120.0},
          phys_metrics={"throughput_orders_per_hr": 113.0}), indent=2))
