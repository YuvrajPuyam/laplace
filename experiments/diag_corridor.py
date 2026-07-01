"""Design + tune a FORCED-CORRIDOR layout for diag_edge (Class B).

dc_pickzone_med is edge-robust (binding constraint = stations/fleet), so an edge
choke can't bite there. This builds a purpose-built layout where a single cross-aisle
IS the only path (picks left, packs right, one cross-aisle), so a slow cap-1 choke on
its middle segment forces serialization — a genuine edge bottleneck the agent can
diagnose. Confirms healthy is OK + the choke bites + it's discriminable from the
service-variance and demand causes.

  python -m experiments.diag_corridor [n_seeds] [aisles] [amrs] [demand]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sim.config import apply_patch, fill_defaults
from sim.runner import run_many

from experiments.diag_explore import signature


def corridor(aisles: int, amrs: int, demand: float) -> dict:
    """Healthy forced-corridor config: picks on the left aisles, packs on the right,
    one cross-aisle at pos 15 — every pick->pack leg must traverse the corridor."""
    right = aisles
    picks = [{"id": f"P{i}", "node": f"A{i}_15", "slots": 1,
              "service_lognorm": [-0.7, 0.3]} for i in (1, 2)]            # left, fast
    packs = [{"id": "K1", "node": f"A{right - 1}_15", "slots": 2, "service_lognorm": [-0.5, 0.3]},
             {"id": "K2", "node": f"A{right}_15", "slots": 2, "service_lognorm": [-0.5, 0.3]}]
    return {
        "schema_version": "1.0", "scenario_id": "corridor_med",
        "layout": {"grid": {"aisles": aisles, "aisle_length_m": 30, "cross_aisles": [15]},
                   "extra_edges": [], "edge_overrides": []},
        "stations": {"pick": picks, "pack": packs,
                     "charge": [{"id": "C1", "node": "A1_00", "slots": 2}],
                     "dock": [{"id": "D1", "node": f"A{right}_00"}]},
        "fleet": {"amr_count": amrs, "speed_mps": 1.5, "battery_capacity_m": 8000,
                  "charge_minutes": 15, "routing": "shortest_path"},
        "demand": {"arrival_rate_per_min": demand, "pack_assignment": "shortest_queue"},
        "horizon": {"sim_minutes": 300, "warmup_minutes": 30},
    }


def causes(aisles: int) -> dict:
    mid = aisles // 2  # the middle corridor segment A{mid}_15 <-> A{mid+1}_15
    return {
        "healthy": {},
        "edge_cap_down": {"layout.edge_overrides": [
            {"edge": f"A{mid}_15->A{mid + 1}_15", "capacity": 1, "max_speed_mps": 0.2}]},
        "service_variance_up": {"stations.pack": [
            {"id": "K1", "node": f"A{aisles - 1}_15", "slots": 2, "service_lognorm": [-0.5, 1.2]},
            {"id": "K2", "node": f"A{aisles}_15", "slots": 2, "service_lognorm": [-0.5, 1.2]}]},
        "demand_up": {"demand.arrival_rate_per_min": None},  # filled below
    }


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    aisles = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    amrs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    demand = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    seeds = list(range(n))
    base = corridor(aisles, amrs, demand)
    cs = causes(aisles)
    cs["demand_up"] = {"demand.arrival_rate_per_min": round(demand * 1.4, 2)}

    print(f"corridor diag probe — {aisles} aisles, {amrs} AMRs, demand {demand}/min, "
          f"{n} CRN seeds (choke = A{aisles//2}_15<->A{aisles//2+1}_15 @ cap1/0.2mps)\n")
    print(f"  {'cause':<20}{'thru/hr':>9}{'p95':>7}{'stn_wait_p95':>14}{'edge_occ%':>11}{'aband%':>8}")
    sigs = {}
    for name, patch in cs.items():
        sig = signature(run_many(fill_defaults(apply_patch(base, patch) if patch else base),
                                 seeds, write_log=False))
        sigs[name] = sig
        print(f"  {name:<20}{sig['throughput']:>9}{sig['p95_latency']:>7}"
              f"{sig['worst_station_wait_p95']:>14}{sig['worst_edge_occupancy_pct']:>11}"
              f"{sig['abandonment_pct']:>8}")
    h = sigs["healthy"]
    print("\n  fingerprints vs healthy:")
    for name, s in sigs.items():
        if name == "healthy":
            continue
        print(f"    {name:<20} d_thru={s['throughput']-h['throughput']:+6.1f}  "
              f"d_stn_wait={s['worst_station_wait_p95']-h['worst_station_wait_p95']:+6.1f}  "
              f"d_edge_occ={s['worst_edge_occupancy_pct']-h['worst_edge_occupancy_pct']:+6.1f}  "
              f"d_aband={s['abandonment_pct']-h['abandonment_pct']:+6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
