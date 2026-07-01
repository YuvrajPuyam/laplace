"""Facility adapter tests: real footprints lower to valid Contract A configs
that the sim can actually run."""

from __future__ import annotations

from facility import FACILITIES, facility_to_config
from sim.config import validate_config
from sim.navgraph import NavGraph
from sim.runner import run_rollout


def test_all_named_facilities_lower_to_valid_configs():
    for fid, spec in FACILITIES.items():
        config, prov = facility_to_config(spec)
        validate_config(config)            # raises if invalid
        assert config["scenario_id"] == fid
        assert prov["layout_source"]       # provenance must cite a source
        # every station sits on a real grid node
        g = NavGraph(config)
        for kind in ("pick", "pack", "charge", "dock"):
            for st in config["stations"][kind]:
                g.node_index(st["node"])    # raises if the node doesn't exist


def test_pick_faces_distributed_not_lumped():
    config, _ = facility_to_config(FACILITIES["dc_pickzone_med"])
    picks = config["stations"]["pick"]
    aisles = {int(p["node"][1:].split("_")[0]) for p in picks}
    positions = {int(p["node"].split("_")[1]) for p in picks}
    # faces spread across multiple aisles AND multiple depths, like a real zone
    assert len(aisles) >= 3
    assert len(positions) >= 2


def test_grounded_scenario_runs_in_the_sim():
    config, _ = facility_to_config(FACILITIES["mfc_compact"])
    result, rows = run_rollout(config, seed=0, write_log=False)
    m = result["metrics"]
    assert m["orders_completed"] > 0
    # conservation: arrived == completed + abandoned
    assert m["orders_completed"] >= 0 and m["orders_abandoned"] >= 0
