"""Braess candidate probe: paired effect of opening the mid shortcut.

Usage: python scripts/braess_probe.py [n_seeds] [arrival_rate] [amr_count]
Runs base vs shortcut (capacity-1 A3_15<->A4_15) on paired seeds and prints
the compare_configs-style stats for throughput and p95 latency. Dev tooling
only — not part of the eval harness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.stats import paired_compare  # noqa: E402
from sim.config import fill_defaults  # noqa: E402
from sim.runner import run_many  # noqa: E402

def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else None
    amrs = int(sys.argv[3]) if len(sys.argv) > 3 else None

    base = json.loads(Path("eval/dev_scenarios/braess_dev.config.json").read_text())
    if rate:
        base["demand"]["arrival_rate_per_min"] = rate
    if amrs:
        base["fleet"]["amr_count"] = amrs
    base = fill_defaults(base)

    shortcut = json.loads(json.dumps(base))
    shortcut["layout"]["extra_edges"] = [
        {"from": "A3_15", "to": "A4_15", "bidirectional": True}]
    shortcut["layout"]["edge_overrides"] = [
        {"edge": "A3_15->A4_15", "capacity": 1, "max_speed_mps": 0.2}]

    seeds = list(range(n))
    # modest worker count: this box has 15 GB RAM and often runs Isaac alongside
    rb = run_many(base, seeds, log_dir="logs/braess_probe", write_log=False,
                  max_workers=4)
    rs = run_many(shortcut, seeds, log_dir="logs/braess_probe", write_log=False,
                  max_workers=4)

    for metric in ("throughput_orders_per_hr", "p95_order_latency_min",
                   "amr_utilization_pct"):
        a = [r["metrics"][metric] for r in rb]
        b = [r["metrics"][metric] for r in rs]
        st = paired_compare(a, b)
        rel = 100 * st["diff_mean"] / st["mean_a"] if st["mean_a"] else 0
        print(f"{metric:28s} base={st['mean_a']:8.2f} shortcut={st['mean_b']:8.2f} "
              f"diff={st['diff_mean']:+7.2f} ({rel:+5.1f}%) p={st['p_value']:.2e} "
              f"[{st['method']}]")

    for label, rr in (("base", rb), ("shortcut", rs)):
        comp = sum(r["metrics"]["orders_completed"] for r in rr) / len(rr)
        ab = sum(r["metrics"]["orders_abandoned"] for r in rr) / len(rr)
        print(f"{label:9s} mean completed={comp:7.1f} abandoned={ab:6.1f} "
              f"({100 * ab / (comp + ab):4.1f}% of arrivals)")

    cong = {}
    for r in rs:
        for e in r["metrics"]["edge_congestion_top5"]:
            cong.setdefault(e["edge"], []).append(e["occupancy_pct"])
    top = sorted(cong.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))[:5]
    print("\nshortcut-config congestion (mean occupancy% when in top5):")
    for edge, vals in top:
        print(f"  {edge:20s} {sum(vals)/len(vals):5.1f}%  (in top5 {len(vals)}/{n} seeds)")


if __name__ == "__main__":
    main()
