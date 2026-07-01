"""Introspect a warehouse USD (or a directory of them) for scan-to-sim:
can we EXTRACT a navgraph (aisle lines, rack rows, dock) from its geometry?

Pass a .usd path to introspect it, or a DIRECTORY to list its .usd files and
introspect each. Reports units/up-axis/extent and rack/shelf centroids
histogrammed by X and Y (clean clusters => aisles are the gaps between rows).

  D:\\iv\\Scripts\\python.exe -m renderer.probe_usd_layout /Isaac/Environments/Digital_Twin_Warehouse
"""

from __future__ import annotations

import sys
from collections import Counter

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.client  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from .build_stage import _resolve_asset_path  # noqa: E402

KEYS = ("rack", "shelf", "pile", "pallet")


def introspect(rel: str) -> None:
    url = _resolve_asset_path(rel)
    print(f"\n[usd] === {rel} ===", flush=True)
    stage = Usd.Stage.Open(url)
    if stage is None:
        print("[usd] FAILED to open", flush=True)
        return
    print(f"[usd] metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)} "
          f"upAxis={UsdGeom.GetStageUpAxis(stage)}", flush=True)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    allr = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    print(f"[usd] extent min={tuple(round(v,1) for v in allr.GetMin())} "
          f"max={tuple(round(v,1) for v in allr.GetMax())}", flush=True)
    rack_xy, n = [], 0
    for prim in stage.Traverse():
        n += 1
        nm = prim.GetName().lower()
        if any(k in nm for k in KEYS) and prim.IsA(UsdGeom.Xformable):
            try:
                rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                c = (rng.GetMin() + rng.GetMax()) * 0.5
                rack_xy.append((round(c[0], 1), round(c[1], 1)))
            except Exception:  # noqa: BLE001
                pass
    print(f"[usd] prims={n}  rack/shelf/pallet prims={len(rack_xy)}", flush=True)
    if rack_xy:
        xs = Counter(round(x) for x, y in rack_xy)
        ys = Counter(round(y) for x, y in rack_xy)
        print(f"[usd] centroid X histogram: {dict(sorted(xs.items()))}", flush=True)
        print(f"[usd] centroid Y histogram: {dict(sorted(ys.items()))}", flush=True)


target = sys.argv[1] if len(sys.argv) > 1 else \
    "/Isaac/Environments/Digital_Twin_Warehouse"
if target.lower().endswith(".usd"):
    introspect(target)
else:
    res, entries = omni.client.list(_resolve_asset_path(target))
    usds = sorted(e.relative_path for e in entries
                  if e.relative_path.lower().endswith(".usd"))
    print(f"[usd] {target} contains .usd: {usds}", flush=True)
    for u in usds[:4]:
        introspect(target.rstrip("/") + "/" + u)

print("\n[usd] done", flush=True)
app.close()
