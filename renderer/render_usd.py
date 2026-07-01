"""render_usd — ISAAC VENV: open an arbitrary USD and RTX-capture an
establishing still. For LOOKING at a real warehouse scan before integrating it.

Reuses build_stage's proven asset-resolution + capture path. References the USD
under /World/env, auto-frames a 3/4 camera to its world bounds (override with
--eye/--target), adds fill lighting, and captures one PNG.

  D:\\iv\\Scripts\\python.exe -m renderer.render_usd \\
      /Isaac/Environments/Simple_Warehouse/full_warehouse.usd \\
      --out renderer/out/full_warehouse.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaacsim import SimulationApp  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="render_usd")
    ap.add_argument("usd_path")
    ap.add_argument("--out", default="renderer/out/usd_view.png")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--eye", type=float, nargs=3, default=None)
    ap.add_argument("--target", type=float, nargs=3, default=None)
    ap.add_argument("--settle", type=int, default=150)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})
    print(f"[render_usd] Isaac booted in {time.perf_counter() - t0:.1f}s", flush=True)

    import carb
    import omni.kit.viewport.utility as vp_util
    import omni.usd
    from omni.kit.viewport.utility import capture_viewport_to_file
    from pxr import Gf, Usd, UsdGeom, UsdLux

    from .build_stage import _look_at, _resolve_asset_path

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    env = UsdGeom.Xform.Define(stage, "/World/env")
    env.GetPrim().GetReferences().AddReference(_resolve_asset_path(args.usd_path))

    # let the reference (and its textures) load before measuring / capturing
    for _ in range(args.settle):
        app.update()

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/env")).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    cx, cy, cz = [(mn[i] + mx[i]) / 2 for i in range(3)]
    span = max(mx[0] - mn[0], mx[1] - mn[1])
    print(f"[render_usd] env bounds min={tuple(round(v, 1) for v in mn)} "
          f"max={tuple(round(v, 1) for v in mx)}", flush=True)

    eye = args.eye or (cx + 0.40 * span, mn[1] - 0.40 * span, mx[2] + 0.50 * span)
    tgt = args.target or (cx, cy, mn[2] + 0.2 * (mx[2] - mn[2]))

    # The warehouse USD brings its own authored lighting; add only a gentle dome
    # for ambient fill so we don't double-light it into a white blowout.
    dome = UsdLux.DomeLight.Define(stage, "/render_dome")
    dome.CreateIntensityAttr(50.0)

    cam = UsdGeom.Camera.Define(stage, "/render_cam")
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(36.0 * args.height / args.width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100000.0))
    _look_at(UsdGeom.Xformable(cam), Gf.Vec3d(*eye), Gf.Vec3d(*tgt))
    print(f"[render_usd] eye={tuple(round(v, 1) for v in eye)} "
          f"target={tuple(round(v, 1) for v in tgt)}", flush=True)

    vp = vp_util.get_active_viewport()
    vp.camera_path = "/render_cam"
    _s = carb.settings.get_settings()
    _s.set("/app/viewport/grid/enabled", False)
    _s.set("/rtx/post/histogram/enabled", False)   # no auto-exposure wash-out
    _s.set("/rtx/post/tonemap/filmIso", 100.0)     # clamp exposure (build_stage uses this)
    _s.set("/rtx/pathtracing/spp", 1)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for _ in range(args.settle):
        app.update()
    capture_viewport_to_file(vp, file_path=out)
    deadline = time.perf_counter() + 300
    ok = False
    while time.perf_counter() < deadline:
        app.update()
        if os.path.exists(out) and os.path.getsize(out) > 0:
            ok = True
            break
    print(f"[render_usd] {'rendered -> ' + out if ok else 'CAPTURE TIMED OUT'} "
          f"({time.perf_counter() - t0:.1f}s total)", flush=True)
    app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
