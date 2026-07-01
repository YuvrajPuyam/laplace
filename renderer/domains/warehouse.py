"""Warehouse domain pack: Contract A config -> TwinScene.

Runs in the MAIN python environment (it imports the sim's navgraph for exact
geometry, so the renderer never re-derives grid math and can never drift from
the simulator). It emits a plain TwinScene JSON that the Isaac venv consumes.

This file IS the template for a new domain: produce nodes (with xy), lanes
(edge polylines), props (stations), and agents (mobile entities) from
whatever your domain's config looks like, plus a CATALOG mapping your types
to assets. Everything downstream is domain-agnostic.
"""

from __future__ import annotations

from sim.config import fill_defaults
from sim.navgraph import NavGraph

from ..catalog import WAREHOUSE_CATALOG, WAREHOUSE_REALISTIC_CATALOG, catalog_to_dict
from ..twin_scene import Agent, CameraPreset, Lane, Node, Prop, TwinScene, default_cameras

CATALOG = WAREHOUSE_CATALOG

DOMAIN = "warehouse"

# station group -> (prop type, z lift so the box sits on the floor)
_STATION_TYPES = {
    "pick": "pick_station",
    "pack": "pack_station",
    "charge": "charger",
    "dock": "dock",
}


def _lane_kind(graph: NavGraph, pool: dict, cross_positions: set[int]) -> str:
    ax, ay = graph.node_xy[pool["a"]]
    bx, by = graph.node_xy[pool["b"]]
    if ax != bx and ay == by:  # horizontal: a cross-aisle or a shortcut
        return "cross_aisle" if int(ay) in cross_positions else "shortcut"
    if ax != bx or ay != by and pool["length"] > 1.5:
        return "shortcut"
    return "aisle"


def to_scene(config: dict, realistic: bool = False) -> TwinScene:
    cfg = fill_defaults(config)
    graph = NavGraph(cfg)
    cross_positions = set(cfg["layout"]["grid"]["cross_aisles"])

    nodes = [Node(id=name, x=xy[0], y=xy[1])
             for name, xy in zip(graph.node_names, graph.node_xy)]

    lanes: list[Lane] = []
    for pi, pool in enumerate(graph.pools):
        ax, ay = graph.node_xy[pool["a"]]
        bx, by = graph.node_xy[pool["b"]]
        kind = _lane_kind(graph, pool, cross_positions)
        lanes.append(Lane(
            id=graph.edge_id(pool["a"], pool["b"]),
            polyline=[(ax, ay), (bx, by)],
            width=0.5 if kind == "aisle" else 0.7,
            bidirectional=not pool["one_way"],
            capacity=pool["capacity"],
            kind=kind))

    props: list[Prop] = []
    for group, prop_type in _STATION_TYPES.items():
        for st in cfg["stations"][group]:
            x, y = graph.node_xy[graph.node_index(st["node"])]
            props.append(Prop(id=st["id"], type=prop_type, x=x, y=y, label=st["id"]))

    dock_node = cfg["stations"]["dock"][0]["node"]
    agents = [Agent(id=f"amr_{i:02d}", type="amr", start_node=dock_node)
              for i in range(cfg["fleet"]["amr_count"])]

    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    floor = {"min_x": min(xs), "min_y": min(ys),
             "max_x": max(xs), "max_y": max(ys), "margin": 3.0}

    cameras: list[CameraPreset] = default_cameras(floor)
    # a congestion close-up on the shortcut, if one exists (the Braess lever)
    shortcut = next((l for l in lanes if l.kind == "shortcut"), None)
    if shortcut:
        mx = sum(p[0] for p in shortcut.polyline) / len(shortcut.polyline)
        my = sum(p[1] for p in shortcut.polyline) / len(shortcut.polyline)
        cameras.append(CameraPreset("congestion_closeup",
                                    eye=(mx, my - 6.0, 5.0), target=(mx, my, 0.0)))

    catalog = WAREHOUSE_REALISTIC_CATALOG if realistic else CATALOG
    return TwinScene(domain=DOMAIN, scenario_id=cfg["scenario_id"], floor=floor,
                     nodes=nodes, lanes=lanes, props=props, agents=agents,
                     cameras=cameras, catalog=catalog_to_dict(catalog))
