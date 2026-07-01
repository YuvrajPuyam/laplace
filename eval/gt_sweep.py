"""Ground-truth sweep runner (WS4).

For each decision (eval/candidates.py), run every candidate over a fixed set of
CRN-paired seeds, score it, and determine the ground-truth optimum with a paired
significance check. Results are cached and versioned by the base config_hash +
seed list, so `make gt-sweeps` is reproducible and cheap to re-run.

This is pure simulation (no LLM, no Max budget): it establishes the GT the eval
arms (LLM-alone, grid-search, agent) are graded against.

  python -m eval.gt_sweep                 # sweep+cache all dev decisions
  python -m eval.gt_sweep --scenario braess_dev --seeds 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.candidates import DEV_DECISIONS, Decision
from eval.held_scenarios import HELD_DECISIONS
from sim.config import apply_patch, config_hash, fill_defaults, load_config
from sim.runner import run_many

ALL_DECISIONS = {**DEV_DECISIONS, **HELD_DECISIONS}

CACHE_DIR = Path("eval/gt_cache")


def _candidate_config(base: dict, patch: dict) -> dict:
    cfg = fill_defaults(apply_patch(base, patch)) if patch else fill_defaults(base)
    return cfg


def _bootstrap_p(diffs: np.ndarray, n_boot: int = 5000, seed: int = 12345) -> float:
    """One-sided paired bootstrap: P(mean paired advantage <= 0). Lower = the
    optimum's lead over the runner-up is more robust. Deterministic (fixed RNG)."""
    if diffs.size == 0:
        return 1.0
    if np.allclose(diffs, diffs[0]):                       # zero-variance lead
        return 0.0 if diffs[0] > 0 else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    return float((means <= 0).mean())


def sweep_decision(decision: Decision, seeds: list[int]) -> dict:
    base = load_config(decision.base_path())
    rows = []
    for cand in decision.candidates:
        cfg = _candidate_config(base, cand.patch)
        results = run_many(cfg, seeds, write_log=False)
        per_seed_metric = [r["metrics"][decision.metric_name] for r in results]
        per_seed_score = np.array(
            [decision.score(r["metrics"], cand.params) for r in results], float)
        rows.append({
            "label": cand.label, "params": cand.params,
            "config_hash": config_hash(cfg),
            "mean_metric": round(float(np.mean(per_seed_metric)), 4),
            "mean_score": float(np.mean(per_seed_score)),
            "_score_by_seed": per_seed_score,
        })

    order = sorted(range(len(rows)), key=lambda i: rows[i]["mean_score"], reverse=True)
    opt, runner = rows[order[0]], rows[order[1]] if len(order) > 1 else None
    p_value, significant = 1.0, False
    if runner is not None:
        diffs = opt["_score_by_seed"] - runner["_score_by_seed"]
        p_value = _bootstrap_p(diffs)
        significant = p_value < 0.01

    for r in rows:
        r.pop("_score_by_seed")
    return {
        "scenario_id": decision.scenario_id,
        "question": decision.question,
        "metric_name": decision.metric_name,
        "rationale": decision.rationale,
        "base_config_hash": config_hash(base),
        "seeds": seeds,
        "candidates": rows,
        "gt_optimum": opt["label"],
        "runner_up": runner["label"] if runner else None,
        "p_value": round(p_value, 5),
        "significant_p<0.01": significant,
    }


def cache_path(scenario_id: str) -> Path:
    return CACHE_DIR / f"{scenario_id}.gt.json"


def load_or_compute(decision: Decision, seeds: list[int], force: bool = False) -> dict:
    p = cache_path(decision.scenario_id)
    if p.exists() and not force:
        cached = json.loads(p.read_text(encoding="utf-8"))
        base_hash = config_hash(load_config(decision.base_path()))
        labels = [c.label for c in decision.candidates]
        if (cached.get("base_config_hash") == base_hash
                and cached.get("seeds") == seeds
                and [c["label"] for c in cached.get("candidates", [])] == labels):
            return cached
    result = sweep_decision(decision, seeds)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gt_sweep")
    ap.add_argument("--scenario", default=None, choices=sorted(ALL_DECISIONS))
    ap.add_argument("--suite", default="dev", choices=("dev", "held", "all"),
                    help="which decision set to sweep when --scenario is omitted")
    ap.add_argument("--seeds", type=int, default=24, help="number of CRN seeds")
    ap.add_argument("--force", action="store_true", help="ignore cache")
    args = ap.parse_args(argv)

    seeds = list(range(args.seeds))
    suite = {"dev": DEV_DECISIONS, "held": HELD_DECISIONS, "all": ALL_DECISIONS}[args.suite]
    ids = [args.scenario] if args.scenario else sorted(suite)
    for sid in ids:
        res = load_or_compute(ALL_DECISIONS[sid], seeds, force=args.force)
        sig = "p<0.01 (sig)" if res["significant_p<0.01"] else f"p={res['p_value']}"
        print(f"[gt] {sid}: optimum={res['gt_optimum']} vs {res['runner_up']} "
              f"({sig}) | {res['metric_name']}")
        for c in res["candidates"]:
            star = " *" if c["label"] == res["gt_optimum"] else ""
            print(f"       {c['label']:>16}: {res['metric_name']}="
                  f"{c['mean_metric']}{star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
