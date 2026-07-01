"""Recon for USD-grounded stations: find the depot/dock and other semantic prims
in a warehouse USD, and confirm the aisle clustering, in one Isaac session.

Standalone (pxr + stdlib only — does NOT import sim/facility, which the Isaac
venv may lack deps for). Reports: rack rows -> aisles + pitch; the rack Y span;
and an inventory of dock/door/vehicle/conveyor/worktable prim groups with their
counts, example paths, and group centroids (so we can locate the depot).

  D:\\iv\\Scripts\\python.exe -m renderer.probe_usd_markers \\
      /Isaac/Environments/Simple_Warehouse/full_warehouse.usd
"""

from __future__ import annotations

import sys
from collections import defaultdict

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom  # noqa: E402

from .build_stage import _resolve_asset_path  # noqa: E402

RACK_KEYS = ("rack", "shelf", "pile")
GROUPS = {
    "dock_door": ("dock", "door", "gate", "garage", "loading", "shutter", "roll"),
    "vehicle": ("forklift", "truck", "agv", "carter", "transporter", "jack", "mover"),
    "conveyor": ("conveyor", "belt", "sorter"),
    "worktable": ("table", "workstation", "desk", "bench", "pack", "station"),
    "human": ("human", "worker", "person", "character", "nurse"),
    "pallet": ("pallet",),
}


def _cluster(vals, tol):
    xs = sorted(vals)
    cl = [[xs[0]]]
    for v in xs[1:]:
        (cl.append([v]) if v - cl[-1][-1] > tol else cl[-1].append(v))
    return cl


target = sys.argv[1] if len(sys.argv) > 1 else \
    "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
stage = Usd.Stage.Open(_resolve_asset_path(target))
mpu = UsdGeom.GetStageMetersPerUnit(stage)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                         [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

dp = stage.GetDefaultPrim()
kids = (dp.GetChildren() if dp else stage.GetPseudoRoot().GetChildren())
print(f"\n[top] default-prim children: {[c.GetName() for c in kids][:40]}", flush=True)

rack_x, rack_y = [], []
groups = defaultdict(lambda: {"n": 0, "paths": [], "cx": 0.0, "cy": 0.0, "cz": 0.0})
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Xformable):
        continue
    nm = prim.GetName().lower()
    try:
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        c = (rng.GetMin() + rng.GetMax()) * 0.5
    except Exception:  # noqa: BLE001
        c = None
    if any(k in nm for k in RACK_KEYS):
        if c is not None:
            rack_x.append(c[0] * mpu)
            rack_y.append(c[1] * mpu)
        continue
    for g, keys in GROUPS.items():
        if any(k in nm for k in keys):
            d = groups[g]
            d["n"] += 1
            if len(d["paths"]) < 4:
                d["paths"].append(str(prim.GetPath()))
            if c is not None:
                d["cx"] += c[0] * mpu
                d["cy"] += c[1] * mpu
                d["cz"] += c[2] * mpu
            break

if rack_x:
    rows = [sum(c) / len(c) for c in _cluster(rack_x, 2.0)]
    aisles = [(rows[i] + rows[i + 1]) / 2 for i in range(len(rows) - 1)]
    pitch = [aisles[i + 1] - aisles[i] for i in range(len(aisles) - 1)]
    print(f"[aisle] {len(rows)} rack rows X={[round(r, 1) for r in rows]}", flush=True)
    print(f"[aisle] -> {len(aisles)} aisles; mean pitch "
          f"{round(sum(pitch) / len(pitch), 2) if pitch else 'NA'} m; "
          f"rack Y span {round(min(rack_y), 1)}..{round(max(rack_y), 1)} "
          f"({round(max(rack_y) - min(rack_y), 1)} m)", flush=True)

for g in GROUPS:
    d = groups[g]
    n = d["n"]
    ctr = (f"centroid=({d['cx'] / n:.1f},{d['cy'] / n:.1f},{d['cz'] / n:.1f})"
           if n else "")
    print(f"[marker] {g}: {n} prims {ctr} e.g. {d['paths']}", flush=True)

print("[done]", flush=True)
app.close()
