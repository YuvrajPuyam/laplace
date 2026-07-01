"""Replay-sufficiency reconstruction (events.schema.md, binding on WS1).

Reconstructs every AMR's (x, y) position at any time t from the event log
plus the config alone — the exact rule the renderer (WS5/WS6) uses:

- Between amr_depart_edge at t0 and its next event: linear interpolation
  along the edge geometry at speed_mps from t0, clamped at the far node.
- While queued: parked at the queue anchor of `at`, offset pos * 0.8 m
  back along the approach direction.
- While in service / charging: parked at the station node, offset by slot.
- Before its first event: parked at the dock, slot-offset by AMR index.
"""

from __future__ import annotations

import json

from . import events as ev
from .navgraph import NavGraph

_QUEUE_SPACING_M = 0.8
# A station's parallel slots are physical docking BAYS. A robot is ~0.8 m wide but ~1.25 m
# LONG, so depending on how it is oriented two bays must sit >= a robot LENGTH + clearance
# apart or the render overlaps them (they are NOT a stack on one point). The event schema pins
# only "offset by slot", not the magnitude, so this is a renderer-side constant; both the
# Three.js viewer and the Isaac render consume it.
_SLOT_SPACING_M = 1.4


class ReplayState:
    """Per-AMR piecewise position track built from one pass over the log."""

    def __init__(self, rows: list[ev.Row], config: dict):
        self.g = NavGraph(config)
        g = self.g
        dock_node = g.node_index(config["stations"]["dock"][0]["node"])
        self.station_node = {}
        self.station_slots = {}
        for kind in ("pick", "pack", "charge"):
            for s in config["stations"][kind]:
                self.station_node[s["id"]] = g.node_index(s["node"])
                self.station_slots[s["id"]] = s.get("slots", 1)

        # track: amr_id -> list of (t, segment) where segment is
        # ("park", x, y) or ("move", x0, y0, x1, y1, speed_m_per_min)
        self.tracks: dict[str, list[tuple[float, tuple]]] = {}
        n_amrs = config["fleet"]["amr_count"]
        for i in range(n_amrs):
            amr_id = f"amr_{i:02d}"
            x, y = g.node_xy[dock_node]
            # dock slot offset: spread back along the aisle by AMR index
            self.tracks[amr_id] = [(0.0, ("park", x, y - i * _SLOT_SPACING_M))]

        import math
        last_dir: dict[str, tuple[float, float]] = {}
        for t, etype, amr_id, event, loc, payload in rows:
            if etype != "amr":
                continue
            p = json.loads(payload)
            if event == ev.AMR_DEPART_EDGE:
                u_name, v_name = p["edge"].split("->")
                x0, y0 = g.node_xy[g.node_index(u_name)]
                x1, y1 = g.node_xy[g.node_index(v_name)]
                self.tracks[amr_id].append(
                    (t, ("move", x0, y0, x1, y1, p["speed_mps"] * 60.0)))
                d = math.hypot(x1 - x0, y1 - y0)
                if d > 0:
                    last_dir[amr_id] = ((x1 - x0) / d, (y1 - y0) / d)
            elif event == ev.AMR_ENTER_QUEUE:
                at = p["at"]
                if p["kind"] == "station":
                    nx, ny = g.node_xy[self.station_node[at]]
                else:  # edge queue: anchor at the tail node of the edge
                    u_name, _ = at.split("->")
                    nx, ny = g.node_xy[g.node_index(u_name)]
                # offset back along the approach direction
                dx, dy = last_dir.get(amr_id, (0.0, 1.0))
                off = p["pos"] * _QUEUE_SPACING_M
                self.tracks[amr_id].append(
                    (t, ("park", nx - dx * off, ny - dy * off)))
            elif event in (ev.AMR_EXIT_QUEUE,):
                at = p["at"]
                if "->" in at:
                    u_name, _ = at.split("->")
                    nx, ny = g.node_xy[g.node_index(u_name)]
                else:
                    nx, ny = g.node_xy[self.station_node[at]]
                self.tracks[amr_id].append((t, ("park", nx, ny)))
            elif event in (ev.SERVICE_START, ev.CHARGE_START):
                nx, ny = g.node_xy[self.station_node[p["station"]]]
                slot = p.get("slot", 0)
                # stack serviced bots SINGLE-FILE back along their approach direction - a service
                # lane feeding the station - rather than side by side. Reads as a queue: slot 0 is
                # docked at the station face, the rest line up behind it (>= a robot length apart).
                dx, dy = last_dir.get(amr_id, (0.0, 1.0))
                off = slot * _SLOT_SPACING_M
                self.tracks[amr_id].append((t, ("park", nx - dx * off, ny - dy * off)))
            # service_end / charge_end: stays parked at station node

    def position(self, amr_id: str, t: float) -> tuple[float, float]:
        track = self.tracks[amr_id]
        # last segment starting at or before t (linear scan ok for tests;
        # bisect if perf ever matters)
        seg_t, seg = track[0]
        for st, s in track:
            if st <= t:
                seg_t, seg = st, s
            else:
                break
        if seg[0] == "park":
            return seg[1], seg[2]
        _, x0, y0, x1, y1, speed = seg
        import math
        edge_len = math.hypot(x1 - x0, y1 - y0)
        if edge_len == 0.0:
            return x0, y0
        frac = min((t - seg_t) * speed / edge_len, 1.0)
        return x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac
