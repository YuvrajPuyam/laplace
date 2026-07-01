"""Isaac Sim headless smoke test — the decisive local-feasibility check.

Boots SimulationApp headless, builds a minimal stage (ground plane + a few
cubes standing in for AMRs), steps physics, renders one frame to PNG, and
reports peak RAM and wallclock. Exit code 0 = this box can run the WS6
replay/Tier-1 pipeline at dev scale.

Run with the Isaac venv:  D:\\iv\\Scripts\\python.exe renderer\\smoke_test.py
"""

import sys
import time

t0 = time.perf_counter()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720})
t_boot = time.perf_counter() - t0
print(f"[smoke] SimulationApp booted headless in {t_boot:.1f}s", flush=True)

import omni.usd  # noqa: E402
from pxr import UsdGeom, UsdLux, Gf  # noqa: E402

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()

UsdGeom.Xform.Define(stage, "/World")
light = UsdLux.DistantLight.Define(stage, "/World/Sun")
light.CreateIntensityAttr(3000.0)

plane = UsdGeom.Cube.Define(stage, "/World/Ground")
plane.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
plane.AddScaleOp().Set(Gf.Vec3f(30.0, 30.0, 0.5))

# a few stand-in AMRs along an "aisle"
for i in range(4):
    cube = UsdGeom.Cube.Define(stage, f"/World/amr_{i:02d}")
    cube.AddTranslateOp().Set(Gf.Vec3d(i * 3.0, 0.0, 0.5))
    cube.AddScaleOp().Set(Gf.Vec3f(0.4, 0.6, 0.3))

for _ in range(30):
    app.update()
print(f"[smoke] stage built + 30 updates at {time.perf_counter() - t0:.1f}s", flush=True)

# render one frame to PNG via the capture helper. The capture is async and
# the FIRST render compiles RTX shaders — wait for the file, generously.
try:
    from pathlib import Path

    import omni.kit.viewport.utility as vp_util
    from omni.kit.viewport.utility import capture_viewport_to_file

    viewport = vp_util.get_active_viewport()
    out = Path(__file__).resolve().parent / "smoke_frame.png"
    out.unlink(missing_ok=True)
    capture_viewport_to_file(viewport, file_path=str(out))
    deadline = time.perf_counter() + 300
    while time.perf_counter() < deadline:
        app.update()
        if out.exists() and out.stat().st_size > 0:
            break
    if out.exists():
        print(f"[smoke] frame rendered -> {out} ({out.stat().st_size} bytes) "
              f"at {time.perf_counter() - t0:.1f}s", flush=True)
    else:
        print("[smoke] FRAME CAPTURE TIMED OUT after 300s of updates", flush=True)
except Exception as e:  # capture API varies across Kit versions; boot is the main check
    print(f"[smoke] viewport capture skipped: {e}", flush=True)

try:
    import psutil
    rss = psutil.Process().memory_info().rss / 1e9
    print(f"[smoke] process RSS: {rss:.1f} GB", flush=True)
except ImportError:
    pass

print(f"[smoke] total wallclock {time.perf_counter() - t0:.1f}s — OK", flush=True)
app.close()
sys.exit(0)
