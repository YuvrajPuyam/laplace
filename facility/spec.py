"""FacilitySpec + the adapter that lowers a real footprint to Contract A.

Contract A is frozen and IS a parameterized parallel-aisle warehouse — exactly
the Roodbergen & De Koster multi-block layout family. So grounding in a real
footprint is a *parameterization + provenance* job, not a schema change:
choose citable real dimensions, place stations the way a real pick zone is
laid out (pick faces distributed through the racks; pack/ship/charge at the
front depot), and record the source.
"""

from __future__ import annotations

import dataclasses
import math

from sim.config import fill_defaults, validate_config
from sim.navgraph import node_name


@dataclasses.dataclass
class FacilitySpec:
    facility_id: str          # -> scenario_id (lowercase, a-z0-9_)
    name: str                 # human-readable
    source: str               # citation for the layout/dimensions
    notes: str                # how the real footprint maps to the grid
    pick_aisles: int          # parallel pick aisles (Contract A: 2..10)
    aisle_length_m: int       # metres front-to-back (10..60)
    cross_aisle_count: int    # 2 = front+back only; 3 = +1 middle block; ...
    n_pick_stations: int      # pick faces represented (1..8)
    n_pack_stations: int      # pack/ship stations at the depot (1..6)
    n_chargers: int           # charge slots (1..4)
    amr_count: int            # starting fleet (the lever the agent will tune)
    speed_mps: float = 1.5
    battery_capacity_m: int = 8000
    charge_minutes: int = 15
    routing: str = "shortest_path"
    arrival_rate_per_min: float = 2.0   # calibrated later from real order data
    pack_assignment: str = "shortest_queue"
    sim_minutes: int = 480
    warmup_minutes: int = 30
    demand_source: str = ""             # citation for the arrival rate


def _spread(n: int, lo: int, hi: int) -> list[int]:
    """n evenly spaced integer positions in [lo, hi] (inclusive of interior)."""
    if n <= 1:
        return [round((lo + hi) / 2)]
    return [round(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def _cross_positions(count: int, length: int) -> list[int]:
    count = max(2, count)
    return sorted(set(_spread(count, 0, length)))


def _sample_even(seq: list[int], k: int) -> list[int]:
    """k evenly-spaced picks from a sorted sequence (deduped, order kept)."""
    if not seq:
        return []
    if k <= 1:
        return [seq[len(seq) // 2]]
    idx = sorted({round(i * (len(seq) - 1) / (k - 1)) for i in range(k)})
    return [seq[i] for i in idx]


def _pick_grid(n: int, aisles: int, length: int,
               positions: list[int] | None = None) -> list[tuple[int, int]]:
    """Distribute n pick faces across interior aisles x interior positions —
    a real pick zone has faces throughout the racks, not lumped at one spot.

    `positions` (optional): real rack-row depths (sim positions) extracted from
    a scanned warehouse. When given, pick faces sit on ACTUAL rack rows instead
    of the synthetic 6..length-6 spread — the scan-to-sim grounding."""
    cols = min(n, max(1, aisles - 2))            # interior aisles
    rows = max(1, math.ceil(n / cols))
    aisle_ids = _spread(cols, 2, aisles - 1) if aisles > 2 else [1]
    if positions:
        pos_ids = _sample_even(sorted(set(positions)), rows)
    else:
        pos_ids = _spread(rows, 6, max(7, length - 6))
    pts = [(a, p) for p in pos_ids for a in aisle_ids]
    return pts[:n]


def assemble_config(
    *, scenario_id: str, aisles: int, aisle_length_m: int, cross: list[int],
    n_pick: int, n_pack: int, n_charge: int,
    amr_count: int, speed_mps: float, battery_capacity_m: int,
    charge_minutes: float, routing: str,
    arrival_rate_per_min: float, pack_assignment: str,
    sim_minutes: int, warmup_minutes: int,
    aisle_spacing_m: float | None = None,
    pick_positions: list[int] | None = None,
) -> dict:
    """Assemble a validated, defaults-filled Contract A config from explicit
    parameters. The single place stations are laid out, so the named-facility
    adapter and the scan-to-sim extractor produce structurally identical
    scenarios. `cross` is the exact cross-aisle position list (Contract A),
    letting the extractor place cross-aisles where the real racks leave gaps
    rather than at evenly-spaced defaults.

    `aisle_spacing_m` (optional) is the real inter-aisle pitch. When None the
    field is OMITTED entirely so the config stays hash-identical to a pre-field
    scenario (the sim falls back to its 3.0 m convention); when set, it makes
    cross-aisle dynamics geometrically faithful to a scanned footprint.
    """
    a, L = aisles, aisle_length_m

    # pick faces distributed through the racks (on real rack rows if scanned)
    pick = []
    used = set()
    for i, (ai, pos) in enumerate(_pick_grid(n_pick, a, L, positions=pick_positions)):
        node = node_name(ai, pos)
        if node in used:                 # nudge off a collision
            pos = min(pos + 1, L)
            node = node_name(ai, pos)
        used.add(node)
        pick.append({"id": f"P{i + 1}", "node": node, "slots": 1,
                     "service_lognorm": [0.3, 0.35]})

    # depot at the front (position 0): pack/ship, then chargers, then dock
    front = sorted(set(_spread(n_pack, 1, a)))
    pack = [{"id": f"K{i + 1}", "node": node_name(front[i % len(front)], 0),
             "slots": 2, "service_lognorm": [0.45, 0.4]}
            for i in range(n_pack)]
    # de-collide pack nodes by bumping position
    seen = {}
    for st in pack:
        seen[st["node"]] = seen.get(st["node"], -1) + 1
        if seen[st["node"]]:
            a_idx = int(st["node"][1:].split("_")[0])
            st["node"] = node_name(a_idx, seen[st["node"]])

    charge = [{"id": f"C{i + 1}", "node": node_name(1, i), "slots": 2}
              for i in range(n_charge)]
    dock = [{"id": "D1", "node": node_name(a, 0)}]

    grid = {"aisles": a, "aisle_length_m": L, "cross_aisles": cross}
    if aisle_spacing_m is not None:
        grid["aisle_spacing_m"] = aisle_spacing_m

    config = {
        "schema_version": "1.0",
        "scenario_id": scenario_id,
        "layout": {
            "grid": grid,
            "extra_edges": [], "edge_overrides": [],
        },
        "stations": {"pick": pick, "pack": pack, "charge": charge, "dock": dock},
        "fleet": {"amr_count": amr_count, "speed_mps": speed_mps,
                  "battery_capacity_m": battery_capacity_m,
                  "charge_minutes": charge_minutes, "routing": routing},
        "demand": {"arrival_rate_per_min": arrival_rate_per_min,
                   "pack_assignment": pack_assignment},
        "horizon": {"sim_minutes": sim_minutes,
                    "warmup_minutes": warmup_minutes},
    }
    validate_config(config)
    return fill_defaults(config)


def facility_to_config(spec: FacilitySpec) -> tuple[dict, dict]:
    """Return (Contract A config, provenance record)."""
    a, L = spec.pick_aisles, spec.aisle_length_m
    cross = _cross_positions(spec.cross_aisle_count, L)

    config = assemble_config(
        scenario_id=spec.facility_id, aisles=a, aisle_length_m=L, cross=cross,
        n_pick=spec.n_pick_stations, n_pack=spec.n_pack_stations,
        n_charge=spec.n_chargers,
        amr_count=spec.amr_count, speed_mps=spec.speed_mps,
        battery_capacity_m=spec.battery_capacity_m,
        charge_minutes=spec.charge_minutes, routing=spec.routing,
        arrival_rate_per_min=spec.arrival_rate_per_min,
        pack_assignment=spec.pack_assignment,
        sim_minutes=spec.sim_minutes, warmup_minutes=spec.warmup_minutes,
    )
    pick = config["stations"]["pick"]

    provenance = {
        "facility_id": spec.facility_id, "name": spec.name,
        "layout_source": spec.source, "demand_source": spec.demand_source,
        "notes": spec.notes,
        "footprint_m": {"width": round((a - 1) * 3.0, 1), "length": float(L)},
        "cross_aisles": cross, "n_pick_faces": len(pick),
        "fleet_amr_count": spec.amr_count,
        "arrival_rate_per_min": spec.arrival_rate_per_min,
    }
    return config, provenance
