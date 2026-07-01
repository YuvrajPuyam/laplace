"""physx_run - the missing piece of the Option-3 PhysX validation.

Turns a fast-DES rollout into a per-robot leg plan, drives that EXACT plan under full
PhysX dynamics on a GPU, and reports how well the kinematic DES agrees with physics
(docs/PHYSX_VALIDATION.md). PhysX is a higher-fidelity MODEL, not ground truth - this is
sim-vs-sim verification, never sim-to-real.

Three sub-commands, split so the two CPU halves run and test on any box, and the one
GPU half is isolated for the cluster:

  # 1. CPU (main env): DES rollout -> per-robot waypoints + constant-speed DES leg times
  python -m renderer.physx_run extract --scenario braess_dev --seed 0 \
      --window-min 8 --out runs/physx/braess_plan.json

  # 2. GPU (Isaac venv, A100 / L40S / A6000): drive the plan under PhysX, measure leg times
  <iv-python> -m renderer.physx_run drive --plan runs/physx/braess_plan.json \
      --out runs/physx/braess_phys.json

  # 3. CPU (main env): the "agree within X%" headline from the two
  python -m renderer.physx_run compare --plan runs/physx/braess_plan.json \
      --phys runs/physx/braess_phys.json --out runs/physx/braess_agreement.json

Times are in SECONDS on both sides, so the relative error is unitless. Leg keys are
(amr_id, leg_index); the plan defines the indices and the drive runner reuses them, so
`extract` and `drive` cannot disagree about which leg is which.

A100/H100 note (PHYSX_VALIDATION.md): PhysX runs fine on them - the AGREEMENT NUMBER needs
only physics, not ray tracing. Drive defaults to a BARE GROUND PLANE in navgraph
coordinates so the PhysX leg distances are byte-identical to the DES geometry; the
photoreal warehouse env (which wants RT cores) is a separate concern, not this number.

Isaac imports are lazy (inside drive()), so extract/compare import and run in the main
env with no Isaac installed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ── shared helpers ────────────────────────────────────────────────────────────
def _dump(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_config(scenario: str | None, config_path: str | None) -> tuple[dict, str]:
    from sim.config import load_config
    if config_path:
        return load_config(config_path), Path(config_path).stem.replace(".config", "")
    for base in ("eval/dev_scenarios", "examples"):
        p = Path(base) / f"{scenario}.config.json"
        if p.exists():
            return load_config(str(p)), scenario
    raise SystemExit(f"scenario '{scenario}' not found in eval/dev_scenarios/ or examples/")


# ── 1. EXTRACT (CPU, main env) - DES rollout -> per-robot leg plan ─────────────
def extract(config: dict, scenario: str, seed: int, *, window_min: float | None,
            include_warmup: bool) -> dict:
    """Run one DES rollout and pull every post-warmup edge traversal as a leg:
    its waypoints (navgraph node coords) and the DES constant-speed travel time."""
    from sim import events as ev
    from sim.navgraph import NavGraph
    from sim.runner import run_rollout

    result, rows = run_rollout(config, seed, write_log=False)
    g = NavGraph(config)
    warmup = float(config["horizon"]["warmup_minutes"])
    t_hi = warmup + window_min if window_min else float("inf")

    per: dict[str, list[dict]] = {}
    for t, etype, amr_id, event, _loc, payload in rows:
        if etype != "amr" or event != ev.AMR_DEPART_EDGE:
            continue
        if not include_warmup and t < warmup:
            continue
        if t > t_hi:
            continue
        p = json.loads(payload)
        u_name, v_name = p["edge"].split("->")
        ax, ay = g.node_xy[g.node_index(u_name)]
        bx, by = g.node_xy[g.node_index(v_name)]
        dist = math.hypot(bx - ax, by - ay)
        speed = float(p["speed_mps"])
        per.setdefault(amr_id, []).append({
            "edge": p["edge"], "a": [round(ax, 4), round(ay, 4)],
            "b": [round(bx, 4), round(by, 4)], "speed_mps": speed,
            "dist_m": round(dist, 4), "des_depart_min": round(float(t), 4),
            "des_time_s": round(dist / speed, 4) if speed > 0 else 0.0})

    legs, des_legs = [], []
    for amr_id in sorted(per):                       # leg_index = order within this amr
        for i, leg in enumerate(per[amr_id]):
            legs.append({"amr": amr_id, "leg": i, **leg})
            des_legs.append([amr_id, i, leg["des_time_s"]])

    m = result.get("metrics", {})
    des_metrics = {k: m[k] for k in ("throughput_orders_per_hr", "p95_order_latency_min")
                   if k in m}
    return {
        "scenario": scenario, "seed": seed, "config_hash": result["config_hash"],
        "warmup_min": warmup, "window_min": window_min,
        "n_robots": config["fleet"]["amr_count"], "n_legs": len(legs),
        "des_metrics": des_metrics, "legs": legs, "des_legs": des_legs,
    }


# ── 2. DRIVE (GPU, Isaac venv) - replay the plan under PhysX ───────────────────
# ⚠ GPU-UNVERIFIED, like renderer/physx_probe.py: written against the Isaac Sim 5.x API,
# not executed (dev box is below Isaac minimums). The control logic is plain math; the
# lines that touch the Isaac API carry NOTE markers - sanity-check those first on the GPU.
def drive(plan: dict, *, dt: float = 1.0 / 60.0, max_accel: float = 1.2,
          goal_tol: float = 0.25, max_sim_min: float | None = None,
          settle_steps: int = 30, out_path: str | Path | None = None) -> dict:
    """Drive every robot along its plan legs under PhysX with accel-limited velocity
    control on a bare ground plane (navgraph coords). Each robot waits until its leg's
    DES departure time, then drives a->b; the per-leg travel time is measured from start
    of motion to arrival. Robots are solid bodies, so contention at a choke edge emerges
    physically. Returns {"phys_legs": [[amr, leg, phys_time_s], ...], "phys_metrics": {}}."""
    import numpy as np
    from isaacsim import SimulationApp                       # NOTE: Isaac entry point
    app = SimulationApp({"headless": True})

    # NOTE: Isaac Sim 5.x core API. Older builds use omni.isaac.core.* - fix here first.
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, GroundPlane
    from pxr import UsdGeom
    import omni.usd

    world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)          # navgraph is x,y; z is up
    GroundPlane(prim_path="/World/ground", size=400.0)

    # group plan legs per robot, in leg order
    by_robot: dict[str, list[dict]] = {}
    for leg in plan["legs"]:
        by_robot.setdefault(leg["amr"], []).append(leg)
    for legs in by_robot.values():
        legs.sort(key=lambda L: L["leg"])

    HALF = 0.35                                              # robot half-extent (m)
    bodies: dict[str, object] = {}
    for amr_id, legs in by_robot.items():
        ax, ay = legs[0]["a"]
        # NOTE: a solid dynamic body per robot. Swap for an iw_hub articulation later for
        # the photoreal render; for the NUMBER a calibrated rigid body is the honest model
        # (it adds accel dynamics + physical collision the kinematic DES lacks).
        bodies[amr_id] = DynamicCuboid(
            prim_path=f"/World/r_{amr_id}", name=amr_id,
            position=np.array([ax, ay, HALF + 0.02]),
            size=2 * HALF, mass=120.0)
        # Keep bodies awake: the accel ramp starts sub-threshold, and a sleeping body
        # ignores set_linear_velocity (the 0-legs bug). Defensive: API name varies by build.
        try:
            bodies[amr_id].set_sleep_threshold(0.0)        # NOTE: Isaac RigidPrim API
        except Exception as e:  # noqa: BLE001
            if __import__("os").environ.get("LAPLACE_DRIVE_DBG"):
                print(f"[dbg] set_sleep_threshold unavailable: {e}", flush=True)
    world.reset()                                           # NOTE: inits PhysX handles
    for _ in range(settle_steps):                          # let bodies settle on the floor
        world.step(render=False)

    _DBG = bool(__import__("os").environ.get("LAPLACE_DRIVE_DBG"))
    if _DBG:                                                # settle-displacement: detect spawn explosion
        for amr_id, legs in by_robot.items():
            ax, ay = legs[0]["a"]
            p = bodies[amr_id].get_world_pose()[0]
            d = math.hypot(float(p[0]) - ax, float(p[1]) - ay)
            print(f"[dbg settle] {amr_id} target_a=({ax},{ay}) actual=({float(p[0]):.2f},"
                  f"{float(p[1]):.2f},{float(p[2]):.2f}) disp={d:.3f}m", flush=True)
    _dbg_amr = "amr_06"; _dbg_next = 0.0

    # DES departure times are ABSOLUTE sim minutes (warmup-offset, e.g. warmup=30 min). Shift so
    # the captured window starts at sim_t=0 — else every robot waits past max_sim_min and nothing
    # departs (0 legs measured). The within-window leg DURATIONS are unaffected (they're relative).
    t_origin = min(L["des_depart_min"] for L in plan["legs"]) * 60.0

    state = {a: {"i": 0, "moving": False, "t_start": None, "speed": 0.0} for a in by_robot}
    phys_legs: list[list] = []
    sim_t, t0_offset = 0.0, settle_steps * dt
    t_cap = (max_sim_min * 60.0) if max_sim_min else float("inf")

    while sim_t < t_cap:
        active = False
        for amr_id, legs in by_robot.items():
            st = state[amr_id]
            if st["i"] >= len(legs):
                continue
            active = True
            leg = legs[st["i"]]
            ax, ay = leg["a"]; bx, by = leg["b"]
            body = bodies[amr_id]
            pos = body.get_world_pose()[0]                  # NOTE: returns (pos, quat)
            depart_s = leg["des_depart_min"] * 60.0 - t_origin
            if not st["moving"]:
                # hold at 'a' until the DES says this robot departs this edge
                body.set_linear_velocity(np.zeros(3))      # NOTE
                if sim_t >= depart_s:
                    body.set_world_pose(np.array([ax, ay, HALF + 0.02]))   # NOTE: snap to a
                    st["moving"] = True
                    st["t_start"] = sim_t
                    st["speed"] = 0.0
                    if _DBG and amr_id == _dbg_amr:
                        print(f"[dbg depart] {amr_id} leg{st['i']} t={sim_t:.2f} "
                              f"a=({ax},{ay}) b=({bx},{by})", flush=True)
                continue
            # accel-limited velocity toward 'b', capped at the DES leg speed
            to = np.array([bx - pos[0], by - pos[1]]); dist = float(np.linalg.norm(to))
            if _DBG and amr_id == _dbg_amr and sim_t >= _dbg_next:
                vv = body.get_linear_velocity()
                print(f"[dbg move] {amr_id} leg{st['i']} t={sim_t:.2f} "
                      f"pos=({float(pos[0]):.2f},{float(pos[1]):.2f}) dist={dist:.3f} "
                      f"|v|={float(np.linalg.norm(vv[:2])):.3f}", flush=True)
                _dbg_next = sim_t + 2.0
            if dist <= goal_tol:                           # crossed waypoint b: record the leg
                if _DBG and amr_id == _dbg_amr:
                    print(f"[dbg arrive] {amr_id} leg{st['i']} t={sim_t:.2f} "
                          f"dt={sim_t - st['t_start']:.3f} |v|="
                          f"{float(np.linalg.norm(body.get_linear_velocity()[:2])):.2f}", flush=True)
                phys_legs.append([amr_id, leg["leg"],
                                  round(sim_t - st["t_start"], 4)])
                st["i"] += 1
                # A robot does not stop at every navgraph node: the DES emits consecutive edges
                # of one trip a leg-time apart (continuous constant-speed motion). Re-accelerating
                # from rest at each 1 m node would inflate every leg ~2x — a stop-start artifact,
                # not physics. So CARRY velocity through contiguous waypoints; only park + ramp
                # again at a genuine inter-trip GAP (next edge departs later than now).
                if st["i"] < len(legs):
                    nxt = legs[st["i"]]
                    nxt_depart = nxt["des_depart_min"] * 60.0 - t_origin
                    if nxt_depart <= sim_t:                # contiguous trip → flow through
                        st["t_start"] = sim_t              # next leg's clock starts at the crossing
                    else:                                  # parked between trips → stop and wait
                        body.set_linear_velocity(np.zeros(3))   # NOTE
                        st["moving"] = False; st["t_start"] = None; st["speed"] = 0.0
                else:                                      # last leg done → stop
                    body.set_linear_velocity(np.zeros(3))  # NOTE
                    st["moving"] = False; st["t_start"] = None; st["speed"] = 0.0
                continue
            # Accel-limited speed ramp integrated in SOFTWARE, then command the ABSOLUTE
            # target velocity. Do NOT derive the command from get_linear_velocity(): a body
            # resting on the floor sheds horizontal velocity to the contact solver each step,
            # and a sub-threshold velocity puts the body to SLEEP (PhysX stops integrating it).
            # Either makes the physics velocity a useless accumulator. The drive IS the AMR's
            # motor — it asserts the velocity it wants every step.
            st["speed"] = min(leg["speed_mps"], st["speed"] + max_accel * dt)
            v = to / dist * st["speed"]
            body.set_linear_velocity(np.array([v[0], v[1], 0.0]))   # NOTE
        world.step(render=False)
        sim_t += dt
        if not active:
            break

    if _DBG:                                                # per-robot progress summary
        for amr_id, legs in by_robot.items():
            st = state[amr_id]; p = bodies[amr_id].get_world_pose()[0]
            print(f"[dbg final] {amr_id} legs_done={st['i']}/{len(legs)} "
                  f"pos=({float(p[0]):.2f},{float(p[1]):.2f}) moving={st['moving']}", flush=True)

    # robots that never reached their last waypoint within t_cap are reported as unfinished
    n_planned = sum(len(v) for v in by_robot.values())
    out = {
        "phys_legs": phys_legs,
        "n_planned_legs": n_planned, "n_measured_legs": len(phys_legs),
        "sim_seconds": round(sim_t, 2),
        "params": {"dt": dt, "max_accel": max_accel, "goal_tol": goal_tol,
                   "max_sim_min": max_sim_min},
        "phys_metrics": {},   # fill if you also aggregate throughput/p95 from the run
    }
    # Persist + print BEFORE app.close(): Isaac teardown can kill the process mid-shutdown and
    # lose a post-close write (the Spike A failure mode). Write while Isaac is still alive.
    if out_path is not None:
        _dump(out, out_path)
        print(f"[drive] measured {out['n_measured_legs']}/{out['n_planned_legs']} legs "
              f"in {out['sim_seconds']}s sim -> {out_path}", flush=True)
    app.close()
    return out


# ── 3. COMPARE (CPU, main env) - the "agree within X%" headline ────────────────
def compare(plan: dict, phys: dict) -> dict:
    from experiments.physx_compare import summarize
    des_legs = {(a, leg): t for a, leg, t in plan["des_legs"]}
    phys_legs = {(a, leg): t for a, leg, t in phys["phys_legs"]}
    out = summarize(des_legs, phys_legs,
                    des_metrics=plan.get("des_metrics") or None,
                    phys_metrics=phys.get("phys_metrics") or None)
    out["coverage"] = {"des_legs": len(des_legs), "phys_legs": len(phys_legs),
                       "compared": out["legs"]["n_legs"]}
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="physx_run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="CPU: DES rollout -> leg plan + DES times")
    e.add_argument("--scenario", default=None)
    e.add_argument("--config", default=None)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--window-min", type=float, default=None,
                   help="cap legs to the first N minutes after warmup (bounds the GPU run)")
    e.add_argument("--include-warmup", action="store_true")
    e.add_argument("--out", required=True)

    d = sub.add_parser("drive", help="GPU: replay the plan under PhysX")
    d.add_argument("--plan", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--max-accel", type=float, default=1.2, help="m/s^2 (calibrate + freeze)")
    d.add_argument("--goal-tol", type=float, default=0.25)
    d.add_argument("--dt", type=float, default=1.0 / 60.0)
    d.add_argument("--max-sim-min", type=float, default=None)

    c = sub.add_parser("compare", help="CPU: agreement number from plan + phys")
    c.add_argument("--plan", required=True)
    c.add_argument("--phys", required=True)
    c.add_argument("--out", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "extract":
        if not (args.scenario or args.config):
            raise SystemExit("extract needs --scenario or --config")
        config, scenario = _load_config(args.scenario, args.config)
        plan = extract(config, scenario, args.seed,
                       window_min=args.window_min, include_warmup=args.include_warmup)
        _dump(plan, args.out)
        print(f"[extract] {scenario} seed={args.seed}: {plan['n_legs']} legs "
              f"across {plan['n_robots']} robots -> {args.out}")
        return 0

    if args.cmd == "drive":
        plan = _load(args.plan)
        drive(plan, dt=args.dt, max_accel=args.max_accel, goal_tol=args.goal_tol,
              max_sim_min=args.max_sim_min, out_path=args.out)   # writes before app.close()
        return 0

    if args.cmd == "compare":
        plan, phys = _load(args.plan), _load(args.phys)
        out = compare(plan, phys)
        if args.out:
            _dump(out, args.out)
        print(json.dumps(out, indent=2))
        if out.get("headline"):
            print("\n" + out["headline"])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
