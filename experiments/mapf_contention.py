"""MAPF contention-curve anchor — does Laplace's congestion model bend the way the
MAPF literature's throughput-vs-fleet curve does? (CRITIQUE_BACKLOG A3.)

The lifelong-MAPF literature reports system throughput RISING and saturating with
agent count, then DECLINING past an optimal density (congestion collapse). This is
the external anchor on Laplace's congestion realism — the critique's exact weak
point — and, unlike a public-DC throughput number, it is real and obtainable.

We reproduce the SHAPE (not the magnitude): sweep fleet size at saturated demand on
a capacity-constrained layout (CRN-paired seeds) and check for the
rising -> peak -> decline signature.

ANCHORS (cite for the shape; absolute levels are NOT comparable — Laplace has pick/
pack service times, Poisson arrivals, and greedy routing that pure MAPF lacks):
  - RHCR (Li, Tinka, Kiesel, Durham, Kumar, Koenig, AAAI 2021, "Lifelong MAPF in
    Large-Scale Warehouses") — the rising/concave body. Fulfillment 33x46/16%:
    2.33 / 3.56 / 4.55 goals/timestep at m = 60 / 100 / 140.
  - Lifelong-MAPF survey (Jiang, Zhang et al., arXiv:2404.16162, Fig 7) and RL-RH-PP
    (Table 3) — the peak-then-decline tail (throughput peaks, then drops as density
    climbs past optimal; RHCR's own tables stay monotone because they are bounded by
    solver runtime, not by a throughput peak — do NOT cite RHCR for the collapse).

NOTE: Laplace caps fleet at 12 (Contract A). If no decline appears within the cap,
that is reported honestly (a fast DES could push past it; the contract bounds us).

  python -m experiments.mapf_contention [scenario] --demand 6 --seeds 12 --fleet 1 2 3 ... 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.config import apply_patch, fill_defaults
from sim.runner import run_many


def sweep(scenario: str, fleets: list[int], demand: float | None,
          seeds: list[int]) -> list[dict]:
    base = json.loads(Path(f"eval/dev_scenarios/{scenario}.config.json").read_text())
    rows = []
    for n in fleets:
        patch = {"fleet.amr_count": n}
        if demand is not None:
            patch["demand.arrival_rate_per_min"] = demand
        cfg = fill_defaults(apply_patch(base, patch))
        results = run_many(cfg, seeds, write_log=False)
        thr = np.array([r["metrics"]["throughput_orders_per_hr"] for r in results])
        util = np.mean([r["metrics"]["amr_utilization_pct"] for r in results])
        comp = np.array([r["metrics"]["orders_completed"] for r in results], float)
        ab = np.array([r["metrics"]["orders_abandoned"] for r in results], float)
        rows.append({
            "fleet": n,
            "throughput": round(float(thr.mean()), 2),
            "throughput_sd": round(float(thr.std(ddof=1)) if len(thr) > 1 else 0.0, 2),
            "utilization_pct": round(float(util), 1),
            "abandonment_pct": round(100 * float(ab.sum() / (comp.sum() + ab.sum()))
                                     if (comp.sum() + ab.sum()) else 0.0, 1),
        })
    return rows


def diagnose(rows: list[dict]) -> dict:
    thr = [r["throughput"] for r in rows]
    peak_i = int(np.argmax(thr))
    peak = rows[peak_i]
    declined = peak_i < len(rows) - 1 and thr[-1] < thr[peak_i]
    drop_pct = round(100 * (thr[peak_i] - thr[-1]) / thr[peak_i], 1) if thr[peak_i] else 0.0
    if declined:
        sig = (f"rising -> peak at fleet={peak['fleet']} ({peak['throughput']}/hr) "
               f"-> decline ({drop_pct}% drop by fleet={rows[-1]['fleet']}) "
               f"= the MAPF contention signature (congestion collapse past optimal)")
    elif peak_i == len(rows) - 1:
        sig = (f"monotone rising through the fleet cap (peak at the max swept "
               f"fleet={peak['fleet']}); the decline is past Laplace's fleet cap of "
               f"{rows[-1]['fleet']} (Contract A) — matches RHCR's rising body, "
               f"collapse not reached within the cap")
    else:
        sig = (f"rising then flat (saturated at fleet~{peak['fleet']}, "
               f"{peak['throughput']}/hr); plateau without a clear decline in range")
    return {"peak_fleet": peak["fleet"], "peak_throughput": peak["throughput"],
            "declines_after_peak": declined, "drop_pct_to_last": drop_pct,
            "signature": sig}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mapf_contention")
    ap.add_argument("scenario", nargs="?", default="braess_dev")
    ap.add_argument("--demand", type=float, default=6.0,
                    help="orders/min; high enough to saturate so the fleet, not "
                         "demand, is the bottleneck (the regime the curve lives in)")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--fleet", type=int, nargs="*",
                    default=list(range(1, 13)))
    args = ap.parse_args(argv)

    seeds = list(range(args.seeds))
    rows = sweep(args.scenario, args.fleet, args.demand, seeds)
    diag = diagnose(rows)

    print(f"\nMAPF contention curve — {args.scenario} @ demand {args.demand}/min, "
          f"{args.seeds} CRN seeds")
    print(f"  {'fleet':>5} {'thru/hr':>9} {'±sd':>6} {'util%':>6} {'aband%':>7}")
    for r in rows:
        star = "  <- peak" if r["fleet"] == diag["peak_fleet"] else ""
        print(f"  {r['fleet']:>5} {r['throughput']:>9} {r['throughput_sd']:>6} "
              f"{r['utilization_pct']:>6} {r['abandonment_pct']:>7}{star}")
    print(f"\n  SIGNATURE: {diag['signature']}")
    print("  ANCHOR: shape vs RHCR (Li et al. AAAI 2021, rising body) + lifelong-MAPF "
          "survey arXiv:2404.16162 Fig 7 (peak-then-decline). Shape, not magnitude.")

    out = Path("eval/results/mapf_contention.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"scenario": args.scenario, "demand": args.demand, "seeds": args.seeds,
         "curve": rows, "diagnosis": diag}, indent=2), encoding="utf-8")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
