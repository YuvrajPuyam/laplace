"""Class-B diagnosis design probe: are the three candidate causes DISCRIMINABLE?

For a diagnosis scenario to be well-posed (spec §5.1), the planted cause must
produce a signature the agent can tell apart from the other candidates by
experiment. This computes, CRN-paired on the dc_pickzone_med baseline, the metric
signature of: healthy, edge-capacity↓ (the planted cause for diag_edge), service-
variance↑, and demand↑ — and prints the discriminating metrics so we can confirm
they differ (and tune until they do) before building the agent task.

  python -m experiments.diag_explore [n_seeds]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sim.config import apply_patch, fill_defaults
from sim.runner import run_many

BASE = "dc_pickzone_med"

# Choke a long stretch of the pos-0 cross-aisle (where pick→pack legs must cross),
# so left/centre legs single-file through it — hard to detour around.
_EDGE_CHOKE = [{"edge": e, "capacity": 1, "max_speed_mps": 0.4} for e in
               ("A2_00->A3_00", "A3_00->A4_00", "A4_00->A5_00", "A5_00->A6_00",
                "A3_00->A2_00", "A4_00->A3_00", "A5_00->A4_00", "A6_00->A5_00")]


def causes(amrs: int, demand: float, demand_up: float) -> dict:
    """Candidate-cause patches over a (possibly re-based) healthy config. The base
    fleet/demand are set high enough that the system is near capacity, so the edge
    choke actually bites (at low load AMRs just detour around it)."""
    base = {"fleet.amr_count": amrs, "demand.arrival_rate_per_min": demand}
    return {
        "healthy": base,
        "edge_cap_down": {**base, "layout.edge_overrides": _EDGE_CHOKE},
        "service_variance_up": {**base, "stations.pack": [
            {"id": "K1", "node": "A1_00", "slots": 2, "service_lognorm": [0.45, 1.1]},
            {"id": "K2", "node": "A5_00", "slots": 2, "service_lognorm": [0.45, 1.1]},
            {"id": "K3", "node": "A9_00", "slots": 2, "service_lognorm": [0.45, 1.1]}]},
        "demand_up": {**base, "demand.arrival_rate_per_min": round(demand * demand_up, 2)},
    }


def signature(results: list[dict]) -> dict:
    m = [r["metrics"] for r in results]
    # worst station wait p95 (service-variance fingerprint)
    sw = []
    for r in m:
        d = r.get("station_wait_p95_min") or {}
        sw.append(max(d.values()) if d else 0.0)
    # worst edge occupancy (congestion fingerprint)
    eo = []
    for r in m:
        top = r.get("edge_congestion_top5") or []
        eo.append(max((e["occupancy_pct"] for e in top), default=0.0))
    comp = sum(r["orders_completed"] for r in m)
    ab = sum(r["orders_abandoned"] for r in m)
    return {
        "throughput": round(float(np.mean([r["throughput_orders_per_hr"] for r in m])), 1),
        "p95_latency": round(float(np.mean([r["p95_order_latency_min"] for r in m])), 1),
        "worst_station_wait_p95": round(float(np.mean(sw)), 1),
        "worst_edge_occupancy_pct": round(float(np.mean(eo)), 1),
        "abandonment_pct": round(100 * ab / (comp + ab), 1) if (comp + ab) else 0.0,
    }


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    amrs = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    demand = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
    seeds = list(range(n))
    base = json.loads(Path(f"eval/dev_scenarios/{BASE}.config.json").read_text())

    print(f"Class-B discriminability probe — {BASE}, {amrs} AMRs, demand {demand}/min, "
          f"{n} CRN seeds\n")
    print(f"  {'cause':<20}{'thru/hr':>9}{'p95':>7}{'stn_wait_p95':>14}"
          f"{'edge_occ%':>11}{'aband%':>8}")
    sigs = {}
    for name, patch in causes(amrs, demand, 1.4).items():
        cfg = fill_defaults(apply_patch(base, patch))
        sig = signature(run_many(cfg, seeds, write_log=False))
        sigs[name] = sig
        print(f"  {name:<20}{sig['throughput']:>9}{sig['p95_latency']:>7}"
              f"{sig['worst_station_wait_p95']:>14}{sig['worst_edge_occupancy_pct']:>11}"
              f"{sig['abandonment_pct']:>8}")

    h = sigs["healthy"]
    print("\n  fingerprints vs healthy (what each cause moves MOST):")
    for name, s in sigs.items():
        if name == "healthy":
            continue
        print(f"    {name:<20} d_thru={s['throughput']-h['throughput']:+6.1f}  "
              f"d_stn_wait={s['worst_station_wait_p95']-h['worst_station_wait_p95']:+6.1f}  "
              f"d_edge_occ={s['worst_edge_occupancy_pct']-h['worst_edge_occupancy_pct']:+6.1f}  "
              f"d_aband={s['abandonment_pct']-h['abandonment_pct']:+6.1f}")
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/diag_signatures.json").write_text(json.dumps(sigs, indent=2))
    print("\n  -> eval/results/diag_signatures.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
