"""dump_rack_cloud — ISAAC VENV: warehouse USD -> rack-cloud JSON.

The Isaac half of the scan-to-sim bridge. Runs in the Isaac venv (pxr only, NO
sim/facility imports, which the Isaac venv lacks deps for). Traverses the USD,
collects rack/shelf centroids in METRES on the Z-up floor frame, and writes a
JSON the MAIN env consumes via facility.usd_extractor.load_cloud.

  D:\\iv\\Scripts\\python.exe -m renderer.dump_rack_cloud \\
      /Isaac/Environments/Simple_Warehouse/full_warehouse.usd \\
      --out renderer/scenes/full_warehouse_cloud.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaacsim import SimulationApp  # noqa: E402

RACK_KEYS = ("rack", "shelf", "pile", "pallet")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dump_rack_cloud")
    ap.add_argument("usd_path", help="USD path or /Isaac asset path")
    ap.add_argument("--out", required=True, help="rack-cloud JSON output path")
    ap.add_argument("--keys", nargs="*", default=list(RACK_KEYS),
                    help="prim-name substrings that mark a storage rack")
    args = ap.parse_args(argv)

    app = SimulationApp({"headless": True})
    try:
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        from .build_stage import _resolve_asset_path  # noqa: PLC0415

        url = _resolve_asset_path(args.usd_path)
        stage = Usd.Stage.Open(url)
        if stage is None:
            print(f"[dump] FAILED to open {args.usd_path}", flush=True)
            return 1
        mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
        up = str(UsdGeom.GetStageUpAxis(stage))
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        keys = tuple(k.lower() for k in args.keys)
        pitch: list[float] = []
        length: list[float] = []
        z_mins: list[float] = []
        for prim in stage.Traverse():
            nm = prim.GetName().lower()
            if not (any(k in nm for k in keys) and prim.IsA(UsdGeom.Xformable)):
                continue
            try:
                rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            except Exception:  # noqa: BLE001 — skip un-boundable prims
                continue
            mn, mx = rng.GetMin(), rng.GetMax()
            c = (mn + mx) * 0.5
            pitch.append(c[0] * mpu)
            length.append(c[1] * mpu)
            z_mins.append(mn[2] * mpu)
        # floor = LOWEST rack bottom (~ floor level). Median would pick a
        # mid-shelf height on multi-level racking and float robots up in Step B.
        floor_z = min(z_mins) if z_mins else 0.0

        cloud = {
            "source": args.usd_path,
            "meters_per_unit": mpu,
            "up_axis": up,
            "floor_z": round(floor_z, 4),
            "pitch": [round(v, 4) for v in pitch],
            "length": [round(v, 4) for v in length],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(cloud, f)
        print(f"[dump] {len(pitch)} rack prims, mpu={mpu} up={up} "
              f"floor_z={floor_z:.2f} -> {args.out}", flush=True)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
