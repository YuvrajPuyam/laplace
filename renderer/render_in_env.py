"""render_in_env — ISAAC VENV: animate our AMRs INSIDE a real warehouse USD.

Step B of scan-to-sim. References the warehouse USD as the environment (its
racks, floor, walls, lighting — NO synthetic geometry) and places ONLY our
robots (iw_hub) on the extracted aisles, mapping each sim (x,y) to real-world
coordinates via the provenance coordinate_map. Motion comes from an
export_tracks rollout on the extracted config. One USD = source of truth for
both the simulated dynamics and the rendered twin.

Two modes:
  --at-frame N --out still.png   : one still with robots frozen at track frame N
  --frames-dir DIR               : full PNG sequence (encode with encode_mp4)

  D:\\iv\\Scripts\\python.exe -m renderer.render_in_env ^
      --env-usd /Isaac/Environments/Simple_Warehouse/full_warehouse.usd ^
      --tracks renderer/scenes/real_full_warehouse_tracks.json ^
      --coord-map eval/dev_scenarios/real_full_warehouse.provenance.json ^
      --at-frame 120 --out renderer/out/real_env_still.png ^
      --eye -30 -28 12 --target -8 4 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from isaacsim import SimulationApp  # noqa: E402


def _make_world_of(cmap):
    """sim (x,y) -> real-world (X,Y,Z) using the extracted coordinate_map.
    X: piecewise-linear over the real per-aisle positions (handles non-uniform
    pitch); Y: affine via the length origin/scale; Z: the real floor height."""
    aw = cmap["aisle_world"]
    y0 = cmap["length_origin_m"]
    ysc = cmap["length_scale"]
    fz = cmap["frame"]["floor_z"]
    psim = cmap.get("sim_pitch_m") or cmap.get("real_pitch_m") or 3.0
    n = len(aw)

    def world_of(xs, ys):
        f = xs / psim if psim else 0.0
        if n == 1:
            x = aw[0]
        else:
            i = max(0, min(int(f), n - 2))
            x = aw[i] + (aw[i + 1] - aw[i]) * (f - i)
        return x, y0 + ys * ysc, fz

    return world_of


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="render_in_env")
    ap.add_argument("--env-usd", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--coord-map", required=True, help="provenance JSON with coordinate_map")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--out", default="renderer/out/real_env_still.png")
    ap.add_argument("--at-frame", type=int, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--eye", type=float, nargs=3, default=None)
    ap.add_argument("--target", type=float, nargs=3, default=None)
    ap.add_argument("--orbit-deg", type=float, default=0.0,
                    help="total degrees the camera orbits around --target over the sequence")
    ap.add_argument("--follow", default=None,
                    help="amr id to chase-cam (e.g. amr_03); overrides --eye/--orbit")
    ap.add_argument("--follow-dist", type=float, default=6.0, help="chase distance behind the AMR (m)")
    ap.add_argument("--follow-height", type=float, default=7.5, help="chase camera height (m)")
    ap.add_argument("--settle", type=int, default=120)
    ap.add_argument("--settle-frame", type=int, default=4)
    ap.add_argument("--dome", type=float, default=60.0, help="fill dome intensity")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})
    print(f"[env] Isaac booted in {time.perf_counter() - t0:.1f}s", flush=True)

    import carb
    import omni.kit.viewport.utility as vp_util
    import omni.usd
    from omni.kit.viewport.utility import capture_viewport_to_file
    from pxr import Gf, UsdGeom, UsdLux

    from .build_stage import _look_at, _resolve_asset_path
    from .catalog import IWHUB, KLT_TOTE

    # Station props placed at each station's dock pose + an offset, matching
    # scripts/check_render_clear.py (which VERIFIED no bot-frame phases these footprints).
    # asset, (offset dx,dy in sim m), rotateZ deg. Pick = no prop (the warehouse racks ARE the
    # pick faces). All /Isaac/Props assets are metres -> scale 1.0.
    _PACK = "/Isaac/Props/PackingTable/packing_table.usd"
    _CABINET = "/Isaac/Props/Sektion_Cabinet/sektion_cabinet_visuals.usd"
    _PALLET = "/Isaac/Props/Pallet/pallet.usd"
    PROP_ASSET = {
        "pack":   (_PACK, (0.0, -1.0), 90.0),
        "charge": (_CABINET, (0.0, 0.9), 0.0),
        "dock":   (_PALLET, (0.0, 0.9), 0.0),
    }

    tracks = json.loads(open(args.tracks, encoding="utf-8").read())
    cm = json.loads(open(args.coord_map, encoding="utf-8").read())
    cmap = cm.get("coordinate_map", cm)
    world_of = _make_world_of(cmap)
    agent_ids = sorted(tracks["agents"])
    n_frames = tracks["n_frames"]

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    env = UsdGeom.Xform.Define(stage, "/World/env")
    env.GetPrim().GetReferences().AddReference(_resolve_asset_path(args.env_usd))
    dome = UsdLux.DomeLight.Define(stage, "/render_dome")
    dome.CreateIntensityAttr(args.dome)

    # ---- our robots: iw_hub + a (hidden until carrying) KLT tote -----------
    ops = {}
    for i, aid in enumerate(agent_ids):
        base = f"/World/agents/a_{i}"
        parent = UsdGeom.Xform.Define(stage, base)
        pxf = UsdGeom.Xformable(parent)
        t_op = pxf.AddTranslateOp()
        r_op = pxf.AddRotateZOp()
        robot = UsdGeom.Xform.Define(stage, f"{base}/robot")
        robot.GetPrim().GetReferences().AddReference(_resolve_asset_path(IWHUB))
        payload = UsdGeom.Xform.Define(stage, f"{base}/payload")
        payload.GetPrim().GetReferences().AddReference(_resolve_asset_path(KLT_TOTE))
        plxf = UsdGeom.Xformable(payload)
        plxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.55))
        UsdGeom.Imageable(payload).MakeInvisible()
        x0, y0 = tracks["agents"][aid]["xy"][0]
        X, Y, Z = world_of(x0, y0)
        t_op.Set(Gf.Vec3d(X, Y, Z))
        ops[aid] = {"t": t_op, "r": r_op, "payload": payload, "carry": None}

    # ---- station props: real Isaac assets at the verified dock-pose offsets ----
    props = tracks.get("stations", [])
    placed = []
    for j, s in enumerate(props):
        spec = PROP_ASSET.get(s["kind"])
        if spec is None:        # pick stations: the warehouse USD's own racks are the pick faces
            continue
        asset, (ox, oy), rz = spec
        X, Y, Z = world_of(s["x"] + ox, s["y"] + oy)
        prim = UsdGeom.Xform.Define(stage, f"/World/stations/s_{j}_{s['id']}")
        prim.GetPrim().GetReferences().AddReference(_resolve_asset_path(asset))
        pxf = UsdGeom.Xformable(prim)
        pxf.AddTranslateOp().Set(Gf.Vec3d(X, Y, Z))
        if rz:
            pxf.AddRotateZOp().Set(float(rz))
        placed.append(f"{s['id']}({s['kind']})")
    print(f"[env] placed {len(placed)} station props: {', '.join(placed)}", flush=True)

    # ---- camera ------------------------------------------------------------
    cam = UsdGeom.Camera.Define(stage, "/render_cam")
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(36.0 * args.height / args.width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100000.0))
    eye = args.eye or (-30, -28, 12)
    tgt = args.target or (-8, 4, 1.5)
    _look_at(UsdGeom.Xformable(cam), Gf.Vec3d(*eye), Gf.Vec3d(*tgt))

    vp = vp_util.get_active_viewport()
    vp.camera_path = "/render_cam"
    _s = carb.settings.get_settings()
    _s.set("/app/viewport/grid/enabled", False)
    _s.set("/rtx/post/histogram/enabled", False)
    _s.set("/rtx/post/tonemap/filmIso", 100.0)
    _s.set("/rtx/pathtracing/spp", 1)

    cam_xf = UsdGeom.Xformable(cam)
    import math as _math
    def cam_eye(i):
        if not args.orbit_deg:
            return eye
        cx, cy = tgt[0], tgt[1]
        ex, ey, ez = eye
        r = _math.hypot(ex - cx, ey - cy)
        a = _math.atan2(ey - cy, ex - cx) + _math.radians(args.orbit_deg) * (i / max(1, n_frames - 1))
        return (cx + r * _math.cos(a), cy + r * _math.sin(a), ez)

    # ---- chase-cam: precompute a smooth eye/target that glides with one AMR ----
    follow_path = None
    if args.follow:
        fa = tracks["agents"][args.follow]
        wp = [world_of(*fa["xy"][i])[:2] for i in range(n_frames)]   # AMR (X,Y) per frame
        def _smooth(seq, w=7):
            out = []
            for i in range(len(seq)):
                lo, hi = max(0, i - w), min(len(seq), i + w + 1)
                out.append((sum(p[0] for p in seq[lo:hi]) / (hi - lo),
                            sum(p[1] for p in seq[lo:hi]) / (hi - lo)))
            return out
        sp = _smooth(wp)                                   # smoothed target path (no jitter)
        d = args.follow_dist
        h = args.follow_height
        last = (0.0, -1.0)                                 # chase direction; default = view from front
        follow_path = []
        for i in range(n_frames):
            tx, ty = sp[i]
            j = min(n_frames - 1, i + 6)
            vx, vy = sp[j][0] - tx, sp[j][1] - ty
            vl = _math.hypot(vx, vy)
            if vl > 0.15:                                  # moving: chase from behind travel dir
                last = (vx / vl, vy / vl)
            ex, ey = tx - last[0] * d, ty - last[1] * d
            follow_path.append(((ex, ey, h), (tx, ty, 0.7)))
        # smooth the eye too, so direction changes ease in
        eyes = _smooth([(e[0], e[1]) for e, _ in follow_path], w=6)
        follow_path = [((eyes[i][0], eyes[i][1], h), follow_path[i][1]) for i in range(n_frames)]

    def apply(i):
        if follow_path is not None:
            e, tg = follow_path[i]
            _look_at(cam_xf, Gf.Vec3d(*e), Gf.Vec3d(*tg))            # chase-cam glides with the AMR
        elif args.orbit_deg:
            _look_at(cam_xf, Gf.Vec3d(*cam_eye(i)), Gf.Vec3d(*tgt))   # orbit the camera per frame
        for aid, o in ops.items():
            tr = tracks["agents"][aid]
            x, y = tr["xy"][i]
            X, Y, Z = world_of(x, y)
            o["t"].Set(Gf.Vec3d(X, Y, Z))
            o["r"].Set(float(tr["heading_deg"][i]))
            carry = bool(tr.get("carrying", [False] * n_frames)[i])
            if carry != o["carry"]:
                img = UsdGeom.Imageable(o["payload"])
                img.MakeVisible() if carry else img.MakeInvisible()
                o["carry"] = carry

    def capture(path, first=False):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)   # else we'd return a STALE file from a prior run
        capture_viewport_to_file(vp, file_path=path)
        deadline = time.perf_counter() + (300 if first else 25)
        while time.perf_counter() < deadline:
            app.update()
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
        return False

    for _ in range(args.settle):
        app.update()

    if args.frames_dir is None:
        i = args.at_frame if args.at_frame is not None else n_frames // 2
        apply(i)
        for _ in range(args.settle):
            app.update()
        ok = capture(os.path.abspath(args.out), first=True)
        print(f"[env] still frame {i} {'-> ' + args.out if ok else 'TIMED OUT'} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
    else:
        fdir = os.path.abspath(args.frames_dir)
        os.makedirs(fdir, exist_ok=True)
        for i in range(n_frames):
            apply(i)
            for _ in range(args.settle_frame):
                app.update()
            capture(os.path.join(fdir, f"frame_{i:04d}.png"), first=(i == 0))
            if i % 30 == 0 or i == n_frames - 1:
                print(f"[env] frame {i + 1}/{n_frames} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
        print(f"[env] {n_frames} frames -> {fdir} "
              f"({time.perf_counter() - t0:.1f}s total)", flush=True)
        ok = True

    app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
