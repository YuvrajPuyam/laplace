"""renderer/physx_spike_a.py - Spike A: N rigid AMRs path-following + ORCA-avoiding,
measured for REAL-TIME under PhysX on Gilbreth (v2 Phase A, the load-bearing unknown).

Question this answers: can ~12 rigid AMRs drive to waypoints AND locally avoid each
other at real-time (realtime_x >= 1.0) in Isaac headless? The handoff's blocker was that
`physx_probe` loaded full ARTICULATIONS (heavy joint solver) and booted the RTX stack
though only physics is needed. This harness uses the cheap path: rigid cuboids +
physics-only stepping (render=False) + the verified CPU ORCA layer for avoidance.

  # smoke (boot + fit), then the real measurement, via the existing SLURM wrapper:
  sbatch --export=ALL,REPO=$HOI/laplace,SIF=$HOI/laplace/containers/isaac-sim-5.1.0.sif,\
PHYSX_MODULE=renderer.physx_spike_a,PHYSX_ARGS="--robots 12 --seconds 30 \
--out runs/physx/spikeA_12.json" scripts/gilbreth/physx.slurm

Pattern: robots on a circle, each ping-ponging to its antipode -> everyone crosses the
centre -> a worst-case continuous avoidance + neighbour-density stress for the timing.
The ORCA velocity math is verified on CPU (tests/test_avoidance.py); only the Isaac
calls below are GPU-unverified - they carry NOTE markers, like renderer/physx_run.drive.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from renderer.avoidance import collision_free_velocities


# -- pure layout / control helpers (no Isaac; unit-testable) ---------------------
def circle_layout(n: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """N robots evenly on a circle of `radius`; goal of each is its antipode.
    A tiny deterministic jitter breaks perfect symmetry (avoids a degenerate pinch)."""
    base = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    ang = base + 0.05 * np.cos(3 * base)
    pos = np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])
    goals = -pos.copy()
    return pos, goals


def preferred_velocities(pos: np.ndarray, goals: np.ndarray, speed: float,
                         goal_tol: float) -> np.ndarray:
    """Velocity each robot WANTS this tick: toward its goal at `speed`, zero if arrived."""
    to_goal = goals - pos
    d = np.linalg.norm(to_goal, axis=1, keepdims=True)
    return np.where(d > goal_tol, to_goal / np.clip(d, 1e-9, None) * speed, 0.0)


def min_separation(pos: np.ndarray) -> float:
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


# -- the GPU harness (Isaac; NOTE-marked, GPU-unverified) ------------------------
def run_spike(n_robots: int = 12, seconds: float = 30.0, dt: float = 1.0 / 60.0,
              control_hz: float = 20.0, radius: float = 0.5, max_speed: float = 1.5,
              time_horizon: float = 2.0, circle_radius: float | None = None,
              settle_steps: int = 30, out_path: str | None = None) -> dict:
    """Drive `n_robots` rigid cuboids under PhysX with ORCA avoidance; measure realtime_x.

    Control (ORCA + set velocity) runs at ~control_hz, physics at 1/dt - decoupling the
    cheap CPU control from the physics rate, as the Isaac perf handbook recommends.
    """
    from isaacsim import SimulationApp                       # NOTE: Isaac entry point
    app = SimulationApp({"headless": True})                  # NOTE: physics-only; never render

    # NOTE: Isaac Sim 5.x core API (older builds: omni.isaac.core.*). Fix here first.
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, GroundPlane
    from pxr import UsdGeom
    import omni.usd

    world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)          # navgraph plane is x,y; z up
    GroundPlane(prim_path="/World/ground", size=400.0)

    if circle_radius is None:
        # size the circle so the ring isn't pre-overlapped: circumference > n * 3 * radius
        circle_radius = max(3.0, n_robots * 3.0 * radius / (2 * math.pi))
    pos, goals = circle_layout(n_robots, circle_radius)

    HALF = radius / math.sqrt(2.0)                           # cube half-extent ~ inscribed radius
    bodies = []
    for i in range(n_robots):
        bodies.append(DynamicCuboid(                         # NOTE: rigid body, NOT articulation
            prim_path=f"/World/r_{i:02d}", name=f"r_{i:02d}",
            position=np.array([pos[i, 0], pos[i, 1], HALF + 0.02]),
            size=2 * HALF, mass=120.0))
    world.reset()                                            # NOTE: inits PhysX handles
    for _ in range(settle_steps):
        world.step(render=False)                             # NOTE: physics only

    vel = np.zeros((n_robots, 2), dtype=float)
    control_every = max(1, int(round((1.0 / control_hz) / dt)))
    n_steps = int(round(seconds / dt))
    step_ms: list[float] = []
    min_sep = np.inf
    pings = 0

    wall0 = time.perf_counter()
    for step in range(n_steps):
        # read current planar positions from PhysX
        for i, b in enumerate(bodies):
            p = b.get_world_pose()[0]                        # NOTE: returns (pos, quat)
            pos[i, 0], pos[i, 1] = float(p[0]), float(p[1])

        # ping-pong: a robot that reached its goal swaps to the antipode (keeps crossings)
        reached = np.linalg.norm(goals - pos, axis=1) < (2 * radius)
        if reached.any():
            goals[reached] = -goals[reached]
            pings += int(reached.sum())

        if step % control_every == 0:
            pref = preferred_velocities(pos, goals, max_speed, goal_tol=radius)
            vel = collision_free_velocities(
                pos, vel, pref, radius=radius, max_speed=max_speed,
                time_horizon=time_horizon, dt=control_every * dt)
            for i, b in enumerate(bodies):
                b.set_linear_velocity(np.array([vel[i, 0], vel[i, 1], 0.0]))  # NOTE

        t_step = time.perf_counter()
        world.step(render=False)                             # NOTE: physics only
        step_ms.append((time.perf_counter() - t_step) * 1000.0)
        min_sep = min(min_sep, min_separation(pos))

    wall = time.perf_counter() - wall0
    sim_seconds = n_steps * dt
    arr = np.array(step_ms)
    report = {
        "n_robots": n_robots, "dt": dt, "control_hz": control_hz,
        "sim_seconds": round(sim_seconds, 3), "wall_seconds": round(wall, 3),
        "realtime_x": round(sim_seconds / wall, 3) if wall > 0 else None,
        "mean_step_ms": round(float(arr.mean()), 3),
        "p95_step_ms": round(float(np.percentile(arr, 95)), 3),
        "min_separation_m": round(min_sep, 3),
        "collision": bool(min_sep < 2 * radius - 0.10),     # >10cm interpenetration
        "pings": pings, "circle_radius_m": round(circle_radius, 3),
        "verdict": ("REAL-TIME OK" if wall > 0 and sim_seconds / wall >= 1.0
                    else "TOO SLOW - tune dt/solver or disable render extensions"),
    }
    # Persist + print BEFORE Isaac teardown: app.close() can kill the process mid-shutdown.
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[spike-a] " + json.dumps(report), flush=True)
    app.close()
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="physx_spike_a", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robots", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=30.0, help="sim seconds to measure")
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--control-hz", type=float, default=20.0)
    ap.add_argument("--radius", type=float, default=0.5, help="AMR avoidance radius (m)")
    ap.add_argument("--max-speed", type=float, default=1.5)
    ap.add_argument("--time-horizon", type=float, default=2.0)
    ap.add_argument("--circle-radius", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    report = run_spike(
        n_robots=args.robots, seconds=args.seconds, dt=args.dt,
        control_hz=args.control_hz, radius=args.radius, max_speed=args.max_speed,
        time_horizon=args.time_horizon, circle_radius=args.circle_radius,
        out_path=args.out)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n[spike-a] {args.robots} robots: {report['verdict']} "
          f"(realtime_x={report['realtime_x']}, mean_step={report['mean_step_ms']}ms, "
          f"min_sep={report['min_separation_m']}m, collision={report['collision']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
