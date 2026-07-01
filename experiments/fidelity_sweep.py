"""experiments/fidelity_sweep.py - density-sweep orchestration + the rho* locator.

Characterizes WHERE the DES abstraction breaks down vs robot density: sweep rho over a
PRE-REGISTERED grid; at each rho aggregate a decision-relevant agreement metric (default:
normalized regret, higher = worse) with a CI; locate rho* = the first density where the
abstraction is CONFIDENTLY and PERSISTENTLY past a pre-registered failure threshold.

Pre-registration (freeze the thresholds/grid/rule BEFORE the sweep, with a content hash)
makes rho* a falsifiable prediction rather than a post-hoc line. The GPU side produces the
DES-vs-PhysX samples; this module is the pure-CPU analysis + integrity layer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.fidelity_metrics import _ranks


# ── pre-registration: freeze / verify a content hash ──────────────────────────
def _canonical(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def freeze_prereg(spec: dict) -> dict:
    """Return a copy with frozen_hash = sha256 of the canonical spec (hash field nulled)."""
    d = dict(spec)
    d["frozen_hash"] = None
    h = hashlib.sha256(_canonical(d).encode("utf-8")).hexdigest()
    d["frozen_hash"] = h
    return d


def verify_prereg(spec: dict) -> bool:
    h = spec.get("frozen_hash")
    if not h:
        return False
    d = dict(spec)
    d["frozen_hash"] = None
    return hashlib.sha256(_canonical(d).encode("utf-8")).hexdigest() == h


def load_prereg(path: str | Path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if not verify_prereg(spec):
        raise ValueError(f"prereg hash mismatch at {path} - edited after freeze")
    return spec


# ── the rho* breakdown-boundary locator ───────────────────────────────────────
def locate_rho_star(rho_grid, metric_lo, metric_point, tau: float) -> dict:
    """rho* = first rho whose CI lower bound is past tau AND stays past for all larger
    rho (persistence guards against a single noisy bin). Also reports the point-estimate
    crossing, the bracketing interval (grid only localizes to a bin), and a monotonicity
    check (a single rho* is the wrong model if agreement is non-monotone in density)."""
    rho = list(rho_grid)
    lo = list(metric_lo)
    pt = list(metric_point)
    G = len(rho)
    if not (len(lo) == G == len(pt)):
        raise ValueError("rho_grid, metric_lo, metric_point must be equal length")

    rho_lower, bracket, idx = None, None, None
    for g in range(G):
        if lo[g] > tau and all(lo[g2] > tau for g2 in range(g, G)):
            rho_lower = rho[g]
            bracket = (rho[g - 1] if g > 0 else None, rho[g])
            idx = g
            break

    rho_point = next((rho[g] for g in range(G) if pt[g] > tau), None)
    mono = (float(np.corrcoef(_ranks(np.array(rho, float)),
                              _ranks(np.array(pt, float)))[0, 1]) if G >= 3 else None)
    return {
        "rho_star_lower": rho_lower,    # confident + persistent breakdown
        "rho_star_point": rho_point,    # best-estimate breakdown
        "bracket": bracket,             # grid bin around rho_star_lower
        "crossing_index": idx,
        "monotonic_spearman": mono,     # > 0 means agreement worsens with density (expected)
        "tau": tau,
    }


def sweep_summary(per_rho: list[dict], tau: float) -> dict:
    """`per_rho` = [{rho, metric_point, metric_lo, metric_hi}, ...] sorted by rho.
    Ties the per-rho aggregates into the rho* verdict."""
    per_rho = sorted(per_rho, key=lambda r: r["rho"])
    out = locate_rho_star([r["rho"] for r in per_rho],
                          [r["metric_lo"] for r in per_rho],
                          [r["metric_point"] for r in per_rho], tau)
    out["per_rho"] = per_rho
    return out


# Default pre-registration spec for the PhysX breakdown study (freeze before running).
DEFAULT_PREREG = {
    "preregistration_id": "physx_breakdown_v1",
    "primary_metric": "normalized_regret",
    "metric_orientation": "higher_is_worse",
    "secondary_metrics": ["decision_flip_rate", "pearson_srcc", "spearman", "rel_effect_error"],
    "rho_grid": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    "rho_definition": "active_robots / reachable_floor_cells",
    "decision_rule": {
        "failure_threshold_tau": 0.05,
        "crossing_rule": "first rho with ci_lo > tau, persisting for all larger rho",
        "ci_bound_used": "lower", "persistence_required": True,
    },
    "effect_fidelity": {"negligible_effect_band_delta": 0.02,
                        "delta_units": "fraction_of_baseline_throughput"},
    "decision_tie_margin": 0.01,
    "bootstrap": {"n_boot": 2000, "ci_pct": [5, 95],
                  "resample_unit_effect": "seed_index",
                  "resample_unit_sweep": "config_within_rho_bin", "rng_seed": 12345},
    "srcc": {"min_configs": 15, "predictivity_threshold": 0.80},
    "generalization": {"layout_families": ["grid_aisle", "fishbone", "cross_dock"],
                       "train_families": ["grid_aisle"],
                       "test_families": ["fishbone", "cross_dock"]},
    "seeds": {"n_seeds_per_config": 64},
}


if __name__ == "__main__":   # emit the frozen prereg file
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1
               else "experiments/prereg_physx_breakdown_v1.json")
    out.write_text(json.dumps(freeze_prereg(DEFAULT_PREREG), indent=2), encoding="utf-8")
    print(f"wrote frozen prereg -> {out}")
