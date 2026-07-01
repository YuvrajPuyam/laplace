"""check_render_clear - ground-truth obstacle check for the twin render.

Tests EVERY AMR track frame against BOTH (a) the real USD rack point cloud and
(b) the station-prop footprints we intend to place. A prop that is only visual
will be PHASED THROUGH the moment a bot track crosses it - so a prop is only
safe to place if no bot-frame (other than a bot docked AT it) lands inside its
footprint. Run before any render.

  python -m scripts.check_render_clear [tracks.json] [provenance.json]
"""
from __future__ import annotations

import json
import sys

import numpy as np

TRACKS = sys.argv[1] if len(sys.argv) > 1 else "renderer/scenes/real_warehouse_demo_tracks.json"
PROV = sys.argv[2] if len(sys.argv) > 2 else "eval/dev_scenarios/real_full_warehouse.provenance.json"
CLOUD = "renderer/scenes/full_warehouse_cloud.json"

# candidate prop placement: each station's prop sits at its dock pose + an offset
# AWAY from where the bot stops (so the docked bot is beside/in-front, not inside),
# and OUT of the travel lanes. footprint (fx,fy) metres. dock_clear = radius around
# the dock pose where the station's OWN docked bot is expected (excluded from conflict).
PROPS = {   # kind -> (prop offset from dock pose dx,dy ; footprint fx,fy)
    "pack":   ((0.0, -1.0), (1.6, 1.0)),   # table 1 m forward of the dock, into the open front
    "charge": ((0.0, 0.9), (1.0, 0.8)),    # cabinet just behind the dock (toward back wall)
    "dock":   ((0.0, 0.9), (1.6, 1.0)),    # pallet behind the dock
    # pick: no prop (the warehouse USD racks ARE the pick faces)
}


def main() -> int:
    cloud = json.load(open(CLOUD, encoding="utf-8"))
    RX, RY = np.array(cloud["pitch"]), np.array(cloud["length"])
    p = json.load(open(PROV, encoding="utf-8"))["coordinate_map"]
    aw, y0, ysc, pit = p["aisle_world"], p["length_origin_m"], p["length_scale"], p["sim_pitch_m"]
    n = len(aw)

    def world_of(xs, ys):
        f = xs / pit
        i = max(0, min(int(f), n - 2))
        return aw[i] + (aw[i + 1] - aw[i]) * (f - i), y0 + ys * ysc

    def in_rack(X, Y, dx=0.6, dy=0.6):
        return bool(np.any((np.abs(RX - X) < dx) & (np.abs(RY - Y) < dy)))

    tr = json.load(open(TRACKS, encoding="utf-8"))
    ids = sorted(tr["agents"])

    # (1) rack clips
    rack_clips = sum(1 for aid in ids for sx, sy in tr["agents"][aid]["xy"]
                     if in_rack(*world_of(sx, sy)))
    print(f"frames={sum(len(tr['agents'][a]['xy']) for a in ids)}  rack_clips={rack_clips}")

    # (2) per-prop conflicts: bot-frames inside a prop footprint that are NOT the bot
    #     docked at that station (dock_clear radius). Reported in SIM metres.
    print("\nprop conflicts (bot-frames inside a prop footprint that aren't docking there):")
    safe, drop = [], []
    for s in tr.get("stations", []):
        spec = PROPS.get(s["kind"])
        if spec is None:
            continue
        (ox, oy), (fx, fy) = spec
        px, py = s["x"] + ox, s["y"] + oy          # prop center (sim)
        hits = 0
        for aid in ids:
            for sx, sy in tr["agents"][aid]["xy"]:
                if abs(sx - px) <= fx / 2 and abs(sy - py) <= fy / 2:
                    # inside footprint - is this just a bot docked at the station?
                    if (sx - s["x"]) ** 2 + (sy - s["y"]) ** 2 > 1.1 ** 2:
                        hits += 1
        tag = "SAFE" if hits == 0 else f"CONFLICT x{hits}"
        (safe if hits == 0 else drop).append(s["id"])
        wx, wy = world_of(px, py)
        print(f"  {s['id']:3} {s['kind']:6} prop@sim({px:5.1f},{py:5.1f}) "
              f"world({wx:6.2f},{wy:6.2f})  {tag}")
    print(f"\nSAFE to place: {safe}")
    print(f"DROP (would be phased): {drop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
