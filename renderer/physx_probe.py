"""physx_probe — ISAAC VENV, GPU: does contention-scale PhysX fit in VRAM?

The FIRST command of the "Option 3" GPU spike (docs/critique-response.md §5;
docs/PHYSX_VALIDATION.md). Before building the full DES-vs-PhysX comparison, this
answers the one gating question: can we step N iw_hub robots under PhysX dynamics,
headless, without OOM-ing — at enough robots to actually queue at a choke edge
(contention scale, NOT free-flow 1-3 robots).

It spawns N iw_hubs as physics articulations on a ground plane, steps physics for
a few seconds, and reports wall-time per step + peak VRAM. Run it with increasing
N until it OOMs; the largest N that holds is the contention budget for the run.

  D:\\iv\\Scripts\\python.exe -m renderer.physx_probe --robots 12 --steps 600
  D:\\iv\\Scripts\\python.exe -m renderer.physx_probe --robots 24 --env-usd /Isaac/Environments/Simple_Warehouse/full_warehouse.usd

⚠ GPU-UNVERIFIED: written against the Isaac Sim 5.x API + render_in_env patterns,
but NOT run (dev box is 8 GB, below Isaac min). The boot/stage half mirrors the
working renderer; the physics half (World, Articulation, drive) is the part to
sanity-check first on the rented GPU — see the NOTE markers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time


def _vram_mib() -> float | None:
    """Peak/used VRAM via nvidia-smi (MiB). None if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=10).decode().strip().splitlines()
        return float(out[0])
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="physx_probe")
    ap.add_argument("--robots", type=int, default=12, help="fleet size to stress")
    ap.add_argument("--steps", type=int, default=600, help="physics steps to run")
    ap.add_argument("--env-usd", default=None,
                    help="optional warehouse USD env; default = bare ground plane")
    ap.add_argument("--spacing", type=float, default=1.5, help="robot start spacing (m)")
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    print(f"[probe] Isaac booted in {time.perf_counter() - t0:.1f}s", flush=True)
    vram_boot = _vram_mib()

    # NOTE: Isaac Sim 5.x core API. If the import path differs on the rented image,
    # this is the first thing to fix (older builds: omni.isaac.core.*).
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, UsdGeom

    import omni.usd

    from .build_stage import _resolve_asset_path
    from .catalog import IWHUB

    world = World(physics_dt=args.dt, rendering_dt=args.dt, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    if args.env_usd:
        add_reference_to_stage(_resolve_asset_path(args.env_usd), "/World/env")
    else:
        GroundPlane(prim_path="/World/ground", size=200.0)

    # spawn N iw_hubs in a grid; each becomes a PhysX articulation
    cols = max(1, int(args.robots ** 0.5))
    for i in range(args.robots):
        path = f"/World/robot_{i}"
        add_reference_to_stage(_resolve_asset_path(IWHUB), path)
        x = (i % cols) * args.spacing
        y = (i // cols) * args.spacing
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).AddTranslateOp().Set(
            Gf.Vec3d(x, y, 0.1))
    print(f"[probe] spawned {args.robots} iw_hubs", flush=True)

    # NOTE: world.reset() initializes physics + articulation handles. If a robot's
    # articulation root isn't found here, the iw_hub ref needs an explicit
    # ArticulationRoot — verify on GPU (see PHYSX_VALIDATION.md).
    world.reset()
    vram_ready = _vram_mib()

    step_t0 = time.perf_counter()
    done = 0
    try:
        for _ in range(args.steps):
            world.step(render=False)          # physics only, no RTX (probe is about PhysX fit)
            done += 1
    except Exception as e:  # noqa: BLE001 — an OOM/crash IS the answer
        print(f"[probe] FAILED after {done} steps: {type(e).__name__}: {e}", flush=True)
    wall = time.perf_counter() - step_t0
    vram_peak = _vram_mib()

    report = {
        "robots": args.robots, "steps_requested": args.steps, "steps_done": done,
        "fits": done == args.steps,
        "ms_per_step": round(1000 * wall / done, 2) if done else None,
        "realtime_x": round((done * args.dt) / wall, 1) if wall > 0 else None,
        "vram_mib": {"boot": vram_boot, "ready": vram_ready, "peak": vram_peak},
    }
    print("[probe] RESULT " + json.dumps(report), flush=True)
    app.close()
    return 0 if report["fits"] else 2


if __name__ == "__main__":
    sys.exit(main())
