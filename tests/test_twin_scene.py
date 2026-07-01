"""WS6 domain-agnostic tests (main env, no isaacsim).

Cover the contract (TwinScene round-trips), the warehouse adapter (correct
geometry from a real scenario), and — the point of the whole design — that a
hand-authored TwinScene from a DIFFERENT domain builds with no warehouse code.
"""

from __future__ import annotations

import json

from engine.store import ScenarioStore
from renderer.catalog import DEFAULT_CATALOG, resolve
from renderer.domains import warehouse
from renderer.twin_scene import Agent, Lane, Node, Prop, TwinScene
from sim.config import apply_patch

SHORTCUT_PATCH = {
    "layout.extra_edges": [{"from": "A3_15", "to": "A4_15", "bidirectional": True}],
    "layout.edge_overrides": [{"edge": "A3_15->A4_15", "capacity": 1,
                               "max_speed_mps": 0.2}],
}


def _braess_scene(with_shortcut: bool = False):
    cfg = ScenarioStore(dirs=("eval/dev_scenarios",)).get("braess_dev")
    if with_shortcut:
        cfg = apply_patch(cfg, SHORTCUT_PATCH)
    return warehouse.to_scene(cfg)


def test_twinscene_roundtrip(tmp_path):
    scene = TwinScene(
        domain="warehouse", scenario_id="x",
        floor={"min_x": 0, "min_y": 0, "max_x": 15, "max_y": 30, "margin": 3.0},
        nodes=[Node("A1_00", 0.0, 0.0)],
        lanes=[Lane("A1_00->A1_01", [(0.0, 0.0), (0.0, 1.0)])],
        props=[Prop("P1", "pick_station", 6.0, 5.0, label="P1")],
        agents=[Agent("amr_00", "amr", "A4_30")])
    path = tmp_path / "s.json"
    scene.to_json(path)
    back = TwinScene.from_json(path)
    assert back.to_dict() == scene.to_dict()
    assert back.lanes[0].polyline == [(0.0, 0.0), (0.0, 1.0)]
    assert back.node_xy("A1_00") == (0.0, 0.0)


def test_warehouse_adapter_geometry():
    scene = _braess_scene()
    assert scene.domain == "warehouse"
    # 6 aisles x 31 positions (0..30)
    assert len(scene.nodes) == 6 * 31
    # picks P1-P3 on aisle 3 (x=6), packs K1-K2 on aisle 4 (x=9)
    picks = [p for p in scene.props if p.type == "pick_station"]
    packs = [p for p in scene.props if p.type == "pack_station"]
    assert {p.id for p in picks} == {"P1", "P2", "P3"}
    assert all(abs(p.x - 6.0) < 1e-9 for p in picks)
    assert all(abs(p.x - 9.0) < 1e-9 for p in packs)
    # 9 AMRs, all starting at the dock node
    amrs = [a for a in scene.agents if a.type == "amr"]
    assert len(amrs) == 9
    assert {a.start_node for a in amrs} == {"A4_30"}


def test_warehouse_shortcut_lane_present():
    scene = _braess_scene(with_shortcut=True)
    shortcut = [l for l in scene.lanes if l.kind == "shortcut"]
    assert shortcut, "the A3_15<->A4_15 shortcut should be a 'shortcut' lane"
    lane = shortcut[0]
    assert lane.capacity == 1
    # horizontal segment between aisle 3 (x=6) and aisle 4 (x=9) at y=15
    xs = sorted(p[0] for p in lane.polyline)
    assert xs == [6.0, 9.0]
    assert all(p[1] == 15.0 for p in lane.polyline)
    # and a congestion close-up camera was added for it
    assert any(c.name == "congestion_closeup" for c in scene.cameras)


def test_foreign_domain_scene_needs_no_warehouse_code():
    """A 'hospital' twin authored by hand uses the same contract + fallbacks,
    proving the renderer path is domain-agnostic."""
    scene = TwinScene(
        domain="hospital", scenario_id="er_triage",
        floor={"min_x": 0, "min_y": 0, "max_x": 20, "max_y": 12, "margin": 2.0},
        nodes=[Node("W_00", 0.0, 0.0), Node("W_01", 5.0, 0.0)],
        lanes=[Lane("W_00->W_01", [(0.0, 0.0), (5.0, 0.0)], kind="corridor")],
        props=[Prop("BED1", "bed", 2.0, 4.0), Prop("GATE", "gate", 10.0, 0.0)],
        agents=[Agent("porter_00", "porter", "W_00")])
    d = json.loads(json.dumps(scene.to_dict()))
    assert TwinScene.from_dict(d).domain == "hospital"
    # unknown types resolve to generic fallbacks, never a gap
    assert resolve("bed", {}, is_agent=False) is DEFAULT_CATALOG["_prop_default"]
    assert resolve("porter", {}, is_agent=True) is DEFAULT_CATALOG["_agent_default"]
