"""Routing-robustness probe for the Braess headline.

The critique's sharpest attack: is the Braess throughput drop a property of
warehouses, or of the *greedy* shortest_path router funnelling everyone onto the
capacity-1 edge? The agent's own report flagged that a congestion_aware router was
not tested. This probe settles it: run base vs the narrow cap-1 shortcut under
BOTH routing policies on the SAME CRN seeds, and report the paired throughput
effect under each (plus the choke-edge occupancy that explains it).

CRN note: the arrival stream depends only on seed+demand, never on layout/fleet/
routing, so all four cells (base|shortcut x shortest_path|congestion_aware) share
one arrival stream per seed — the effect under each policy is a clean paired diff.

Usage: python scripts/routing_robustness.py [n_seeds]   (default 24)
Dev tooling — not part of the eval harness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.stats import paired_compare  # noqa: E402
from sim.config import fill_defaults  # noqa: E402
from sim.runner import run_many  # noqa: E402

CHOKE = "A3_15->A4_15"


def _variant(base: dict, *, shortcut: bool, routing: str) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg["fleet"]["routing"] = routing
    if shortcut:
        cfg["layout"]["extra_edges"] = [
            {"from": "A3_15", "to": "A4_15", "bidirectional": True}]
        cfg["layout"]["edge_overrides"] = [
            {"edge": "A3_15->A4_15", "capacity": 1, "max_speed_mps": 0.2}]
    return cfg


def _choke_occupancy(results: list[dict]) -> tuple[float, int]:
    """Mean occupancy% of the cap-1 choke edge across seeds where it's in top5."""
    vals = [e["occupancy_pct"] for r in results
            for e in r["metrics"]["edge_congestion_top5"] if e["edge"] == CHOKE]
    return (sum(vals) / len(vals) if vals else 0.0), len(vals)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seeds = list(range(n))
    base = fill_defaults(
        json.loads(Path("eval/dev_scenarios/braess_dev.config.json").read_text()))

    print(f"braess_dev — routing-robustness of the Braess effect (n={n} CRN seeds)")
    print(f"  fleet={base['fleet']['amr_count']} AMRs, demand="
          f"{base['demand']['arrival_rate_per_min']}/min, horizon="
          f"{base['horizon']['sim_minutes']}min\n")
    print(f"  shortcut = narrow {CHOKE} (capacity 1, 0.2 m/s)\n")

    for routing in ("shortest_path", "congestion_aware"):
        rb = run_many(_variant(base, shortcut=False, routing=routing), seeds,
                      write_log=False, max_workers=4)
        rs = run_many(_variant(base, shortcut=True, routing=routing), seeds,
                      write_log=False, max_workers=4)

        a = [r["metrics"]["throughput_orders_per_hr"] for r in rb]
        b = [r["metrics"]["throughput_orders_per_hr"] for r in rs]
        st = paired_compare(a, b)
        rel = 100 * st["diff_mean"] / st["mean_a"] if st["mean_a"] else 0.0
        occ, k = _choke_occupancy(rs)

        def _ab(rr: list[dict]) -> float:
            comp = sum(r["metrics"]["orders_completed"] for r in rr)
            ab = sum(r["metrics"]["orders_abandoned"] for r in rr)
            return 100 * ab / (comp + ab) if (comp + ab) else 0.0

        print(f"-- routing = {routing}")
        print(f"   throughput   base={st['mean_a']:7.2f}  shortcut={st['mean_b']:7.2f}"
              f"  diff={st['diff_mean']:+7.2f} ({rel:+5.1f}%)  p={st['p_value']:.2e}"
              f"  [{st['method']}]")
        print(f"   abandonment  base={_ab(rb):5.1f}%  shortcut={_ab(rs):5.1f}%")
        print(f"   choke {CHOKE} occupancy (shortcut): {occ:5.1f}%"
              f"  (in top5 {k}/{n} seeds)\n")


if __name__ == "__main__":
    main()
