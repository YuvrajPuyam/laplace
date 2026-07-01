"""TwinScene — the domain-agnostic, renderer-agnostic scene contract.

This is the seam that makes the digital twin retargetable. A *domain pack*
(e.g. renderer/domains/warehouse.py) turns its own config into a TwinScene; a
*renderer* (Isaac in renderer/build_stage.py, or a future Three.js viewer)
consumes a TwinScene and never knows which domain produced it. Swapping the
twin to a hospital, factory, or airport means writing a new domain pack that
emits TwinScene + an asset catalog — the renderer is untouched.

Coordinates are a right-handed plane in METERS: (x, y) ground plane, z up.
The renderer maps y -> world and lifts props/agents in z. Everything here is
plain data: dataclasses that round-trip through JSON, no third-party imports,
no isaacsim, no sim dependency. Either Python environment can build or read
a scene.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

SCHEMA_VERSION = "twin-1.0"


@dataclasses.dataclass
class Node:
    """A waypoint in the navgraph. Agents occupy and move between nodes;
    most are not rendered, but they anchor lane geometry and agent positions."""
    id: str
    x: float
    y: float
    kind: str = "waypoint"  # waypoint | junction | dock | station_anchor


@dataclasses.dataclass
class Lane:
    """A traversable edge, rendered as a floor marking. `polyline` is the
    full ground path in meters (>=2 points); straight lanes have 2."""
    id: str
    polyline: list[tuple[float, float]]
    width: float = 0.6
    bidirectional: bool = True
    capacity: int = 2
    kind: str = "aisle"  # aisle | cross_aisle | shortcut | corridor


@dataclasses.dataclass
class Prop:
    """A stationary object: a station, shelf, machine, bed, gate. `type` is a
    semantic key resolved by the renderer's asset catalog (catalog.py)."""
    id: str
    type: str
    x: float
    y: float
    z: float = 0.0
    rot_deg: float = 0.0
    label: str = ""


@dataclasses.dataclass
class Agent:
    """A mobile entity: AMR, forklift, cart, person. `type` -> asset catalog.
    `start_node` is where it sits before its first event (replay rule)."""
    id: str
    type: str
    start_node: str


@dataclasses.dataclass
class CameraPreset:
    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]


@dataclasses.dataclass
class TwinScene:
    domain: str                       # which domain pack produced this
    scenario_id: str
    floor: dict                       # {"min_x","min_y","max_x","max_y","margin"}
    nodes: list[Node] = dataclasses.field(default_factory=list)
    lanes: list[Lane] = dataclasses.field(default_factory=list)
    props: list[Prop] = dataclasses.field(default_factory=list)
    agents: list[Agent] = dataclasses.field(default_factory=list)
    cameras: list[CameraPreset] = dataclasses.field(default_factory=list)
    # asset catalog travels WITH the scene (type -> AssetSpec as plain dict),
    # so the renderer needs no domain code — scene.json is self-describing.
    catalog: dict = dataclasses.field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    # ---- node lookup (used by the replay driver) --------------------------
    def node_xy(self, node_id: str) -> tuple[float, float]:
        for n in self.nodes:
            if n.id == node_id:
                return (n.x, n.y)
        raise KeyError(node_id)

    # ---- serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "TwinScene":
        def tup(seq):
            return [tuple(p) for p in seq]
        return cls(
            domain=d["domain"], scenario_id=d["scenario_id"], floor=d["floor"],
            nodes=[Node(**n) for n in d.get("nodes", [])],
            lanes=[Lane(id=l["id"], polyline=tup(l["polyline"]),
                       width=l.get("width", 0.6),
                       bidirectional=l.get("bidirectional", True),
                       capacity=l.get("capacity", 2), kind=l.get("kind", "aisle"))
                   for l in d.get("lanes", [])],
            props=[Prop(**p) for p in d.get("props", [])],
            agents=[Agent(**a) for a in d.get("agents", [])],
            cameras=[CameraPreset(name=c["name"], eye=tuple(c["eye"]),
                                 target=tuple(c["target"]))
                     for c in d.get("cameras", [])],
            catalog=d.get("catalog", {}),
            schema_version=d.get("schema_version", SCHEMA_VERSION))

    @classmethod
    def from_json(cls, path: str | Path) -> "TwinScene":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def default_cameras(floor: dict) -> list[CameraPreset]:
    """Sensible presets for any planar facility, derived from floor bounds."""
    cx = (floor["min_x"] + floor["max_x"]) / 2
    cy = (floor["min_y"] + floor["max_y"]) / 2
    span = max(floor["max_x"] - floor["min_x"], floor["max_y"] - floor["min_y"], 1.0)
    return [
        CameraPreset("overview",
                     eye=(cx, cy - span * 0.9, span * 1.1),
                     target=(cx, cy, 0.0)),
        CameraPreset("top_down",
                     eye=(cx, cy, span * 1.6),
                     target=(cx, cy, 0.0)),
    ]
