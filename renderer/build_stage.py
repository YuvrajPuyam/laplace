"""build_stage — ISAAC VENV: TwinScene JSON -> USD stage -> headless RTX still.

Runs in the Isaac Sim python (D:\\iv). Domain-agnostic: it reads a TwinScene
and the asset catalog, builds the stage from primitives (floor, lane markings,
props, agents), and captures a frame. It never imports the sim or any domain
code — give it a hospital scene and it renders a hospital.

  D:\\iv\\Scripts\\python.exe -m renderer.build_stage ^
      renderer/scenes/braess_shortcut.json ^
      --out renderer/out/braess_shortcut.png --camera congestion_closeup

USD axis convention here: scene (x, y) ground plane -> USD (x, y); z is up.
Props/agents are lifted so they rest on the floor (z = height/2).
"""

from __future__ import annotations

import argparse
import sys
import time

# Boot Isaac BEFORE importing pxr/omni (SimulationApp must come first).
from isaacsim import SimulationApp  # noqa: E402


def _color_to_usd(rgb):
    from pxr import Gf
    return Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))


_ASSETS_ROOT_CACHE = {}


def _assets_root():
    if "v" in _ASSETS_ROOT_CACHE:
        return _ASSETS_ROOT_CACHE["v"]
    root = None
    for mod in ("isaacsim.storage.native", "omni.isaac.nucleus"):
        try:
            m = __import__(mod, fromlist=["get_assets_root_path"])
            root = m.get_assets_root_path()
            if root:
                break
        except Exception:  # noqa: BLE001
            continue
    _ASSETS_ROOT_CACHE["v"] = root
    return root


def _resolve_asset_path(usd_path: str) -> str:
    """Catalog usd_path -> a referenceable URL. Absolute URLs pass through;
    paths starting with '/Isaac' (or any '/') are resolved against the Isaac
    asset root (local cache or NVIDIA content server); anything else is a
    local file path used as-is."""
    if "://" in usd_path:
        return usd_path
    if usd_path.startswith("/"):
        root = _assets_root()
        if root:
            return root.rstrip("/") + usd_path
    return usd_path


def build(scene, stage, out_png: str, camera_name: str,
          width: int, height: int, settle_frames: int,
          eye_override=None, target_override=None, dress=False):
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    from .catalog import catalog_from_dict, resolve

    # the catalog rides inside the scene — no domain code needed here
    domain_catalog = catalog_from_dict(scene.catalog)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")

    # ---- materials: cache one UsdPreviewSurface per distinct color ---------
    mat_root = "/World/Looks"
    UsdGeom.Scope.Define(stage, mat_root)
    _mats: dict = {}

    def material_for(spec):
        key = (round(spec.color[0], 3), round(spec.color[1], 3),
               round(spec.color[2], 3), round(spec.metallic, 2),
               round(spec.roughness, 2))
        if key in _mats:
            return _mats[key]
        idx = len(_mats)
        mat = UsdShade.Material.Define(stage, f"{mat_root}/m_{idx}")
        shader = UsdShade.Shader.Define(stage, f"{mat_root}/m_{idx}/surf")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            _color_to_usd(spec.color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            float(spec.metallic))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            float(spec.roughness))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        _mats[key] = mat
        return mat

    def box(path, cx, cy, cz, sx, sy, sz, spec, rot_deg=0.0):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
        if rot_deg:
            xf.AddRotateZOp().Set(rot_deg)
        xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        UsdShade.MaterialBindingAPI(cube).Bind(material_for(spec))
        return cube

    def emissive_material(path, rgb, diffuse=(0.05, 0.05, 0.06)):
        """A material whose emissiveColor we mutate per frame (status bands,
        congestion heatmap). Returns (material, shader) — keep the shader to
        Set emissiveColor each frame."""
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/s")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            _color_to_usd(diffuse))
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            _color_to_usd(rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat, sh

    def place(path, cx, cy, spec, rot_deg=0.0):
        """Spec-driven object as either a primitive box or a referenced USD
        asset. Returns (translate_op, rotate_op, z_rest) — z_rest is where the
        object sits on the floor, for the animation loop. This is the seam
        that lets the catalog point a type at a real NVIDIA asset (kind='usd')
        without changing any caller."""
        if spec.kind == "usd":
            asset = _resolve_asset_path(spec.usd_path)
            xform = UsdGeom.Xform.Define(stage, path)
            xform.GetPrim().GetReferences().AddReference(asset)
            xf = UsdGeom.Xformable(xform)
            t_op = xf.AddTranslateOp(); t_op.Set(Gf.Vec3d(cx, cy, 0.0))
            r_op = xf.AddRotateZOp(); r_op.Set(rot_deg)
            s = float(spec.usd_scale)
            xf.AddScaleOp().Set(Gf.Vec3f(s, s, s))
            return t_op, r_op, 0.0  # assets are authored with base at z=0
        sx, sy, sz = spec.size
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube)
        t_op = xf.AddTranslateOp(); t_op.Set(Gf.Vec3d(cx, cy, sz / 2))
        r_op = xf.AddRotateZOp(); r_op.Set(rot_deg)
        xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        UsdShade.MaterialBindingAPI(cube).Bind(material_for(spec))
        return t_op, r_op, sz / 2

    # ---- ground + floor ----------------------------------------------------
    f = scene.floor
    m = f.get("margin", 3.0)
    fx0, fy0 = f["min_x"] - m, f["min_y"] - m
    fx1, fy1 = f["max_x"] + m, f["max_y"] + m
    cx0, cy0 = (fx0 + fx1) / 2, (fy0 + fy1) / 2
    # large neutral ground so the warehouse never floats on Isaac's default grid
    ground_spec = type("S", (), {"color": (0.09, 0.09, 0.10), "metallic": 0.0,
                                 "roughness": 0.96})()
    box("/World/ground", cx0, cy0, -0.20, 400.0, 400.0, 0.2, ground_spec)
    floor_spec = type("S", (), {"color": (0.17, 0.17, 0.19), "metallic": 0.0,
                                "roughness": 0.85})()
    box("/World/floor", cx0, cy0, -0.05, (fx1 - fx0), (fy1 - fy0), 0.1, floor_spec)

    # ---- lanes: thin floor strips; the shortcut is an emissive heatmap -----
    import math
    lane_colors = {"aisle": (0.30, 0.30, 0.34), "cross_aisle": (0.34, 0.34, 0.40),
                   "shortcut": (0.30, 0.30, 0.34), "corridor": (0.32, 0.32, 0.36)}
    lanes_scope = "/World/lanes"
    UsdGeom.Scope.Define(stage, lanes_scope)
    shortcut_shader = None
    for i, lane in enumerate(scene.lanes):
        (ax, ay), (bx, by) = lane.polyline[0], lane.polyline[-1]
        cx, cy = (ax + bx) / 2, (ay + by) / 2
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 or 0.1
        ang = math.degrees(math.atan2(by - ay, bx - ax))
        if lane.kind == "shortcut":
            # flush decal, wider, with a mutable emissive (green->amber->red)
            cube = UsdGeom.Cube.Define(stage, f"{lanes_scope}/l_{i}")
            cube.GetSizeAttr().Set(1.0)
            xf = UsdGeom.Xformable(cube)
            xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.03))
            xf.AddRotateZOp().Set(ang)
            xf.AddScaleOp().Set(Gf.Vec3f(length, max(lane.width, 1.0), 0.06))
            mat, shortcut_shader = emissive_material(
                f"{lanes_scope}/sc_mat", (0.12, 0.45, 0.16))
            UsdShade.MaterialBindingAPI(cube).Bind(mat)
            continue
        col = lane_colors.get(lane.kind, lane_colors["aisle"])
        spec = type("S", (), {"color": col, "metallic": 0.0, "roughness": 0.8})()
        box(f"{lanes_scope}/l_{i}", cx, cy, 0.02, length, lane.width, 0.04, spec,
            rot_deg=ang)

    # ---- props (stations) --------------------------------------------------
    props_scope = "/World/props"
    UsdGeom.Scope.Define(stage, props_scope)
    for i, p in enumerate(scene.props):
        spec = resolve(p.type, domain_catalog, is_agent=False)
        place(f"{props_scope}/p_{i}", p.x, p.y, spec, rot_deg=p.rot_deg)

    # ---- agents: parent Xform (animated) > robot child + payload child -----
    # the parent carries the per-frame translate/rotate; the robot sits under
    # it, and a payload box (hidden by default) toggles on while carrying so
    # the pick->carry->pack work is visible.
    agents_scope = "/World/agents"
    UsdGeom.Scope.Define(stage, agents_scope)
    cardboard = type("S", (), {"color": (0.62, 0.45, 0.24), "metallic": 0.0,
                               "roughness": 0.9})()
    agent_ops: dict = {}
    for i, a in enumerate(scene.agents):
        spec = resolve(a.type, domain_catalog, is_agent=True)
        ax, ay = scene.node_xy(a.start_node)
        ax += (i % 3) * 1.0  # fan the fleet near the dock so they don't z-fight
        ay += (i // 3) * 1.0
        base = f"{agents_scope}/a_{i}"
        parent = UsdGeom.Xform.Define(stage, base)
        pxf = UsdGeom.Xformable(parent)
        t_op = pxf.AddTranslateOp(); t_op.Set(Gf.Vec3d(ax, ay, 0.0))
        r_op = pxf.AddRotateZOp(); r_op.Set(0.0)

        robot_top = 0.4
        if spec.kind == "usd":
            robot = UsdGeom.Xform.Define(stage, f"{base}/robot")
            robot.GetPrim().GetReferences().AddReference(
                _resolve_asset_path(spec.usd_path))
            s = float(spec.usd_scale)
            UsdGeom.Xformable(robot).AddScaleOp().Set(Gf.Vec3f(s, s, s))
        else:
            sx, sy, sz = spec.size
            robot = UsdGeom.Cube.Define(stage, f"{base}/robot")
            robot.GetSizeAttr().Set(1.0)
            rxf = UsdGeom.Xformable(robot)
            rxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, sz / 2))
            rxf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
            UsdShade.MaterialBindingAPI(robot).Bind(material_for(spec))
            robot_top = sz

        payload_spec = domain_catalog.get("payload")
        if payload_spec is not None and payload_spec.kind == "usd":
            payload = UsdGeom.Xform.Define(stage, f"{base}/payload")
            payload.GetPrim().GetReferences().AddReference(
                _resolve_asset_path(payload_spec.usd_path))
            plxf = UsdGeom.Xformable(payload)
            plxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, robot_top + 0.05))
            s = float(payload_spec.usd_scale)
            plxf.AddScaleOp().Set(Gf.Vec3f(s, s, s))
        else:
            payload = UsdGeom.Cube.Define(stage, f"{base}/payload")
            payload.GetSizeAttr().Set(1.0)
            plxf = UsdGeom.Xformable(payload)
            plxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, robot_top + 0.22))
            plxf.AddScaleOp().Set(Gf.Vec3f(0.42, 0.42, 0.42))
            UsdShade.MaterialBindingAPI(payload).Bind(material_for(cardboard))
        UsdGeom.Imageable(payload).MakeInvisible()

        # status band: a thin emissive pad under the robot, recolored per frame
        # by its state (green=driving, cyan=working, blue=charging, red=blocked)
        band = UsdGeom.Cylinder.Define(stage, f"{base}/band")
        band.CreateRadiusAttr(0.62)
        band.CreateHeightAttr(0.05)
        band.CreateAxisAttr("Z")
        UsdGeom.Xformable(band).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.04))
        band_mat, band_sh = emissive_material(f"{base}/band_mat", (0.18, 0.20, 0.24))
        UsdShade.MaterialBindingAPI(band).Bind(band_mat)

        agent_ops[a.id] = {"t": t_op, "r": r_op, "z": 0.0,
                           "payload": payload, "band": band_sh}

    # ---- warehouse dressing: instanced racks between aisles + walls --------
    if dress:
        rack_spec = domain_catalog.get("_rack")
        wall_spec = domain_catalog.get("_wall")
        dress_scope = "/World/dressing"
        UsdGeom.Scope.Define(stage, dress_scope)

        bulk_spec = domain_catalog.get("_rack")          # tall pallet pile
        shelf_spec = domain_catalog.get("_rack4") or bulk_spec  # low pick shelf

        def ref_instance(path, asset_spec, x, y, z, rotz=0.0, sz=None):
            xf_prim = UsdGeom.Xform.Define(stage, path)
            xf_prim.GetPrim().GetReferences().AddReference(
                _resolve_asset_path(asset_spec.usd_path))
            xf = UsdGeom.Xformable(xf_prim)
            xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            if rotz:
                xf.AddRotateZOp().Set(rotz)
            s = float(asset_spec.usd_scale)
            xf.AddScaleOp().Set(Gf.Vec3f(s, s, (sz if sz is not None else s)))
            xf_prim.GetPrim().SetInstanceable(True)   # share geometry on GPU
            return xf_prim

        aisle_xs = sorted({round(n.x, 3) for n in scene.nodes})
        cross_ys = sorted({round((l.polyline[0][1] + l.polyline[-1][1]) / 2, 3)
                           for l in scene.lanes if l.kind == "cross_aisle"})
        pick_ys = [p.y for p in scene.props if "pick" in p.type]
        depot = [(p.x, p.y) for p in scene.props
                 if p.type in ("pack_station", "charger", "dock")]
        y_lo, y_hi = f["min_y"], f["max_y"]

        def _near(v, vs, d):
            return any(abs(v - u) < d for u in vs)

        # Zone-aware racking: uniform WITHIN a zone (as real racking is
        # installed), but the zones differ. Pick-face rows get low shelf
        # racking; bulk-storage rows get tall pallet racking; the front depot
        # (pack/charge/dock) is kept clear; cross-aisles stay open.
        if bulk_spec is not None and len(aisle_xs) >= 2:
            ri = 0
            for xa, xb in zip(aisle_xs[:-1], aisle_xs[1:]):
                xc = (xa + xb) / 2.0
                yy = y_lo + 1.2
                while yy < y_hi - 1.0:
                    open_here = (_near(yy, cross_ys, 1.6) or
                                 any(abs(xc - dx) < 2.0 and abs(yy - dy) < 2.6
                                     for dx, dy in depot))
                    if not open_here:
                        # NATURAL proportions (no z-stretch) so racks aren't
                        # warped; height variety comes from rack TYPE + the
                        # mezzanine, not distortion.
                        if _near(yy, pick_ys, 2.2):       # pick face: shelf rack
                            ref_instance(f"{dress_scope}/shelf_{ri}", shelf_spec,
                                         xc, yy, 0.0, rotz=90.0)
                        else:                              # bulk storage: pallet pile
                            ref_instance(f"{dress_scope}/bulk_{ri}", bulk_spec,
                                         xc, yy, 0.0, rotz=90.0)
                        ri += 1
                    yy += 3.8   # >= a rack bay's footprint, so no overlap
        if wall_spec is not None:
            wi = 0
            for x in (fx0, fx1):
                yy = fy0
                while yy < fy1:
                    ref_instance(f"{dress_scope}/wall_{wi}", wall_spec, x, yy, 0.0,
                                 rotz=90.0)
                    wi += 1
                    yy += 6.0
            for y in (fy0, fy1):
                xx = fx0
                while xx < fx1:
                    ref_instance(f"{dress_scope}/wall_{wi}", wall_spec, xx, y, 0.0)
                    wi += 1
                    xx += 6.0

        # tall AS/RS-style block along one wall — the asymmetric skyline accent
        # that a real facility has (automation tower / mezzanine), distinct
        # height and structure from the racking.
        mezz = type("S", (), {"color": (0.20, 0.22, 0.28), "metallic": 0.35,
                              "roughness": 0.45})()
        box(f"{dress_scope}/mezzanine", fx1 - 2.0, y_lo + (y_hi - y_lo) * 0.62,
            3.2, 3.0, (y_hi - y_lo) * 0.34, 6.4, mezz)

    # ---- lights: warehouse ceiling fixtures + low fill ---------------------
    # rect "fixtures" down the length give believable pools of light and real
    # contact shadows; the dome is dropped to fill only so the scene isn't flat
    key = UsdLux.DistantLight.Define(stage, "/World/key")
    key.CreateIntensityAttr(1500.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-65, 12, 0))
    dome = UsdLux.DomeLight.Define(stage, "/World/dome")
    dome.CreateIntensityAttr(160.0)
    fy_lo, fy_hi = f["min_y"], f["max_y"]
    for j in range(4):
        fr = (j + 0.5) / 4.0
        rl = UsdLux.RectLight.Define(stage, f"/World/ceil_{j}")
        rl.CreateIntensityAttr(4200.0)
        rl.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
        rl.CreateWidthAttr(5.0)
        rl.CreateHeightAttr(3.0)
        rxf = UsdGeom.Xformable(rl)
        rxf.AddTranslateOp().Set(Gf.Vec3d((f["min_x"] + f["max_x"]) / 2,
                                          fy_lo + fr * (fy_hi - fy_lo), 8.0))
        rxf.AddRotateXYZOp().Set(Gf.Vec3f(180, 0, 0))  # face down

    # ---- camera ------------------------------------------------------------
    cam = next((c for c in scene.cameras if c.name == camera_name),
               scene.cameras[0] if scene.cameras else None)
    cam_path = "/World/cam"
    cam_prim = UsdGeom.Camera.Define(stage, cam_path)
    if eye_override is not None:
        eye = Gf.Vec3d(*eye_override)
        tgt = Gf.Vec3d(*(target_override or (0, 0, 0)))
    elif cam:
        eye = Gf.Vec3d(*cam.eye)
        tgt = Gf.Vec3d(*cam.target)
    else:
        eye, tgt = Gf.Vec3d(0, -20, 18), Gf.Vec3d(0, 0, 0)
    # wide establishing lens (USD default 50mm is too telephoto for a facility)
    cam_prim.CreateFocalLengthAttr(18.0)
    cam_prim.CreateHorizontalApertureAttr(36.0)
    cam_prim.CreateVerticalApertureAttr(36.0 * height / width)
    cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.1, 10000.0))
    _look_at(UsdGeom.Xformable(cam_prim), eye, tgt)
    cam_meta = {"eye": [eye[0], eye[1], eye[2]],
                "target": [tgt[0], tgt[1], tgt[2]],
                "focal_length": 18.0, "aperture_h": 36.0,
                "aperture_v": 36.0 * height / width,
                "width": width, "height": height}
    return cam_path, agent_ops, cam_meta, shortcut_shader


def _look_at(xform, eye, tgt):
    from pxr import Gf
    fwd = (tgt - eye)
    fwd = fwd.GetNormalized()
    up = Gf.Vec3d(0, 0, 1)
    # guard the straight-down/up degeneracy: fwd parallel to up -> zero cross
    if abs(Gf.Dot(fwd, up)) > 0.999:
        up = Gf.Vec3d(0, 1, 0)
    right = Gf.Cross(fwd, up).GetNormalized()
    true_up = Gf.Cross(right, fwd)
    # USD camera looks down -Z; build a basis (right, up, -fwd)
    m = Gf.Matrix4d(
        right[0], right[1], right[2], 0,
        true_up[0], true_up[1], true_up[2], 0,
        -fwd[0], -fwd[1], -fwd[2], 0,
        eye[0], eye[1], eye[2], 1)
    xform.MakeMatrixXform().Set(m)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="build_stage")
    ap.add_argument("scene_json")
    ap.add_argument("--out", default="renderer/out/stage.png")
    ap.add_argument("--camera", default="overview")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--usd-out", default=None, help="also write the .usd stage")
    ap.add_argument("--settle", type=int, default=60)
    ap.add_argument("--eye", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="override camera eye (meters)")
    ap.add_argument("--target", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="override camera target (meters)")
    ap.add_argument("--dress", action="store_true",
                    help="dress the scene: instanced shelf racks between aisles + walls")
    ap.add_argument("--shotlist", default=None,
                    help="JSON list of {name,eye,target}; renders a still per shot "
                         "in ONE boot (a gallery)")
    ap.add_argument("--at-frame", type=int, default=None,
                    help="with --shotlist + --animate: freeze robots at this frame")
    ap.add_argument("--animate", default=None,
                    help="tracks.json from export_tracks; renders a PNG sequence")
    ap.add_argument("--frames-dir", default="renderer/out/frames",
                    help="output dir for the animated PNG sequence")
    ap.add_argument("--settle-frame", type=int, default=4,
                    help="viewport updates between animation frames")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})
    print(f"[build] Isaac booted in {time.perf_counter() - t0:.1f}s", flush=True)

    import os

    import omni.usd

    from .twin_scene import TwinScene

    scene = TwinScene.from_json(args.scene_json)

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    cam_path, agent_ops, cam_meta, shortcut_shader = build(
        scene, stage, args.out, args.camera, args.width, args.height, args.settle,
        eye_override=args.eye, target_override=args.target, dress=args.dress)
    print(f"[build] stage built: {len(scene.props)} props, {len(scene.agents)} "
          f"agents, {len(scene.lanes)} lanes (domain={scene.domain})", flush=True)

    if args.usd_out:
        stage.GetRootLayer().Export(args.usd_out)
        print(f"[build] wrote {args.usd_out}", flush=True)

    import omni.kit.viewport.utility as vp_util
    from omni.kit.viewport.utility import capture_viewport_to_file
    from pxr import Gf
    vp = vp_util.get_active_viewport()
    vp.camera_path = cam_path

    # kill the two things that make captures look like a dev preview:
    import carb
    _s = carb.settings.get_settings()
    _s.set("/app/viewport/grid/enabled", False)      # no overlay grid in frame
    _s.set("/app/viewport/show/camera", False)
    _s.set("/app/viewport/show/lights", False)
    _s.set("/rtx/post/histogram/enabled", False)     # disable auto-exposure wash-out
    _s.set("/rtx/post/tonemap/filmIso", 100.0)
    _s.set("/rtx/pathtracing/spp", 1)

    def capture(path, first=False):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        capture_viewport_to_file(vp, file_path=path)
        deadline = time.perf_counter() + (300 if first else 20)
        while time.perf_counter() < deadline:
            app.update()
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
        return False

    def apply_frame(tracks, i):
        """Freeze robots/heatmap at animation frame i (for still galleries)."""
        from pxr import Gf as _G
        from pxr import UsdGeom as _UG
        for aid, ops in agent_ops.items():
            tr = tracks["agents"].get(aid)
            if tr is None:
                continue
            x, y = tr["xy"][i]
            ops["t"].Set(_G.Vec3d(float(x), float(y), ops["z"]))
            ops["r"].Set(float(tr["heading_deg"][i]))
            if ops.get("payload") is not None and tr.get("carrying", [False])[i]:
                _UG.Imageable(ops["payload"]).MakeVisible()
            if ops.get("band") is not None:
                c = tr.get("band", [[0.3, 0.3, 0.3]] * (i + 1))[i]
                ops["band"].GetInput("emissiveColor").Set(
                    _G.Vec3f(float(c[0]), float(c[1]), float(c[2])))
        if shortcut_shader is not None:
            occ = tracks.get("shortcut_occ", [0])[i] if i < len(
                tracks.get("shortcut_occ", [])) else 0
            fr = min(occ / 2.0, 1.0)
            shortcut_shader.GetInput("emissiveColor").Set(
                _G.Vec3f(0.12 + 0.85 * fr, 0.45 * (1 - fr) + 0.25 * fr, 0.16 * (1 - fr)))

    if args.shotlist:
        import json as _json

        from pxr import Gf as _G
        from pxr import UsdGeom as _UG
        shots = _json.loads(open(args.shotlist, encoding="utf-8").read())
        if args.animate and args.at_frame is not None:
            tracks = _json.loads(open(args.animate, encoding="utf-8").read())
            for _ in range(args.settle):
                app.update()
            apply_frame(tracks, args.at_frame)
        cam_xf = _UG.Xformable(stage.GetPrimAtPath(cam_path))
        out_dir = os.path.abspath(args.frames_dir)
        os.makedirs(out_dir, exist_ok=True)
        for j, s in enumerate(shots):
            _look_at(cam_xf, _G.Vec3d(*s["eye"]), _G.Vec3d(*s["target"]))
            for _ in range(args.settle if j == 0 else args.settle_frame):
                app.update()
            fp = os.path.join(out_dir, f"{s['name']}.png")
            capture(fp, first=(j == 0))
            print(f"[build] shot '{s['name']}' -> {fp}", flush=True)
        print(f"[build] {len(shots)} shots -> {out_dir} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
        app.close()
        return 0

    if args.animate:
        import json as _json

        from pxr import UsdGeom
        tracks = _json.loads(open(args.animate, encoding="utf-8").read())
        n = tracks["n_frames"]
        frames_dir = os.path.abspath(args.frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        with open(os.path.join(frames_dir, "render_manifest.json"), "w",
                  encoding="utf-8") as mf:
            _json.dump(cam_meta, mf)
        from pxr import Gf as _Gf
        carry_state = {aid: None for aid in agent_ops}  # avoid redundant toggles
        occ_series = tracks.get("shortcut_occ", [0] * n)
        for _ in range(args.settle):  # warm up before the first capture
            app.update()
        for i in range(n):
            for aid, ops in agent_ops.items():
                tr = tracks["agents"].get(aid)
                if tr is None:
                    continue
                x, y = tr["xy"][i]
                ops["t"].Set(Gf.Vec3d(float(x), float(y), ops["z"]))
                ops["r"].Set(float(tr["heading_deg"][i]))
                carrying = bool(tr.get("carrying", [False] * n)[i])
                if carrying != carry_state[aid] and ops.get("payload") is not None:
                    img = UsdGeom.Imageable(ops["payload"])
                    img.MakeVisible() if carrying else img.MakeInvisible()
                    carry_state[aid] = carrying
                band = tr.get("band")
                if band and ops.get("band") is not None:
                    c = band[i]
                    ops["band"].GetInput("emissiveColor").Set(
                        _Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])))
            if shortcut_shader is not None:
                occ = occ_series[i] if i < len(occ_series) else 0
                # green (clear) -> amber (1) -> red (2+ over a capacity-1 edge)
                f = min(occ / 2.0, 1.0)
                heat = (0.12 + 0.85 * f, 0.45 * (1 - f) + 0.25 * f, 0.16 * (1 - f))
                shortcut_shader.GetInput("emissiveColor").Set(
                    _Gf.Vec3f(*[float(v) for v in heat]))
            for _ in range(args.settle_frame):
                app.update()
            fp = os.path.join(frames_dir, f"frame_{i:04d}.png")
            capture(fp, first=(i == 0))
            if i % 30 == 0 or i == n - 1:
                print(f"[build] frame {i + 1}/{n} ({time.perf_counter() - t0:.0f}s)",
                      flush=True)
        print(f"[build] {n} frames -> {frames_dir} "
              f"({time.perf_counter() - t0:.1f}s total)", flush=True)
        app.close()
        return 0

    out = os.path.abspath(args.out)
    for _ in range(args.settle):
        app.update()
    ok = capture(out, first=True)
    print(f"[build] {'rendered -> ' + out if ok else 'CAPTURE TIMED OUT'} "
          f"({time.perf_counter() - t0:.1f}s total)", flush=True)
    app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
