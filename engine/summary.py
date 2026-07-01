"""get_scene_summary content: prose summary + machine-readable editable bounds.

No metrics anywhere in here — the agent must not get performance numbers
without spending rollouts (tools.md §1). Bounds are hand-derived from
Contract A and must be updated if (and only if) the schema ever changes.
"""

from __future__ import annotations

import math

# dot-path -> {"min","max"} | {"enum"} | {"max_items"} (arrays)
EDITABLE_BOUNDS = {
    "layout.grid.aisles": {"min": 2, "max": 10},
    "layout.grid.aisle_length_m": {"min": 10, "max": 60},
    "layout.grid.cross_aisles": {"max_items": 6, "item_min": 0, "item_max": 60},
    "layout.extra_edges": {"max_items": 8},
    "layout.edge_overrides": {"max_items": 16},
    "stations.pick": {"min_items": 1, "max_items": 8},
    "stations.pack": {"min_items": 1, "max_items": 6},
    "stations.charge": {"min_items": 1, "max_items": 4},
    "stations.dock": {"min_items": 1, "max_items": 2},
    "fleet.amr_count": {"min": 1, "max": 12},
    "fleet.speed_mps": {"min": 0.5, "max": 3.0},
    "fleet.battery_capacity_m": {"min": 1000, "max": 20000},
    "fleet.charge_minutes": {"min": 1, "max": 60},
    "fleet.routing": {"enum": ["shortest_path", "congestion_aware"]},
    "fleet.dispatch": {"enum": ["nearest_idle", "atc", "covert"]},
    "fleet.congestion_penalty": {"min": 0.0, "max": 5.0},
    "fleet.charge_threshold_pct": {"min": 0.05, "max": 0.5},
    "demand.arrival_rate_per_min": {"min": 0.5, "max": 8.0},
    "demand.pack_assignment": {"enum": ["round_robin", "shortest_queue"]},
    "horizon.sim_minutes": {"min": 60, "max": 1440},
    "horizon.warmup_minutes": {"min": 0, "max": 120},
}

# Patches may touch only these roots; identity fields are never editable.
EDITABLE_ROOTS = ("layout", "stations", "fleet", "demand", "horizon")


def _mean_service_min(lognorm: list[float]) -> float:
    mu, sigma = lognorm
    return math.exp(mu + sigma * sigma / 2)


def scene_summary_text(cfg: dict) -> str:
    g = cfg["layout"]["grid"]
    st = cfg["stations"]
    fl = cfg["fleet"]
    dm = cfg["demand"]
    hz = cfg["horizon"]

    lines = [
        f"Warehouse '{cfg['scenario_id']}': {g['aisles']} parallel aisles of "
        f"{g['aisle_length_m']} m, cross-aisles at positions "
        f"{', '.join(str(c) for c in g['cross_aisles'])}. Default edges carry "
        f"capacity 2 and are bidirectional.",
    ]
    if cfg["layout"]["extra_edges"]:
        ee = ", ".join(f"{e['from']}{'<->' if e.get('bidirectional', True) else '->'}{e['to']}"
                       for e in cfg["layout"]["extra_edges"])
        lines.append(f"Extra edges beyond the grid: {ee}.")
    if cfg["layout"]["edge_overrides"]:
        ov = "; ".join(
            f"{o['edge']}" + "".join(
                f", {k}={o[k]}" for k in ("capacity", "one_way", "max_speed_mps") if k in o)
            for o in cfg["layout"]["edge_overrides"])
        lines.append(f"Edge overrides: {ov}.")

    picks = ", ".join(
        f"{s['id']}@{s['node']} ({s['slots']} slot{'s' if s['slots'] > 1 else ''}, "
        f"mean {_mean_service_min(s['service_lognorm']):.1f} min)"
        for s in st["pick"])
    packs = ", ".join(
        f"{s['id']}@{s['node']} ({s['slots']} slot{'s' if s['slots'] > 1 else ''}, "
        f"mean {_mean_service_min(s['service_lognorm']):.1f} min)"
        for s in st["pack"])
    charges = ", ".join(f"{s['id']}@{s['node']} ({s['slots']} slots)" for s in st["charge"])
    docks = ", ".join(f"{s['id']}@{s['node']}" for s in st["dock"])
    lines.append(f"Pick stations: {picks}. Pack stations: {packs}. "
                 f"Charging: {charges}. Dock: {docks}.")

    lines.append(
        f"Fleet: {fl['amr_count']} AMRs at {fl['speed_mps']} m/s, battery "
        f"{fl['battery_capacity_m']} m of travel, {fl['charge_minutes']} min to "
        f"recharge, routing={fl['routing']}.")
    lines.append(
        f"Demand: Poisson arrivals at {dm['arrival_rate_per_min']}/min; each "
        f"order picks at a uniformly random pick station, then delivers to a "
        f"pack station chosen by {dm['pack_assignment']}.")
    lines.append(
        f"Horizon: {hz['sim_minutes']} sim-minutes, first {hz['warmup_minutes']} "
        f"excluded from metrics as warmup.")
    return " ".join(lines)
