"""renderer/physx_stream.py - LIVE pose+telemetry stream for the twin (v2 Phase 2).

Runs up to MAX AMRs on a scenario's navgraph as a small live OPERATIONAL model: each
robot runs a task cycle (fetch an order -> service at a pick -> carry to pack -> service
-> repeat, charging when its battery runs low, idling when demand is slack), path-follows
aisle waypoints (shortest_path) and locally avoids the others with the verified ORCA layer.
Every frame it emits NDJSON over a TCP socket:

  {t_sim, agents:{id:[x,y,theta]}, kpis:{intent:{...}, carrying, throughput, p95_latency,
                                         chargers:{busy,cap}, fleet, robots, source}}

so the viewer renders BOTH the motion AND an honest live Fleet-Status / Facility-Readout
panel (every count is derived from the live motion, never from a stale baked run). Two
integrators behind one identical wire format:

  --mock   kinematic integration (pos += v*dt), CPU, runs anywhere. For wiring/tests.
  (default) PhysX rigid-body integration in Isaac (set velocity -> step -> read pose) -
           real contact dynamics. Reuses the Isaac API proven by renderer/physx_spike_a.
           GPU; run in the Isaac container on Gilbreth.

The socket is BIDIRECTIONAL: the engine (/twin/live) connects as a TCP client, feeds frames
through the delayed-feed jitter buffer (engine/playout.py) to the viewer, AND forwards live
control messages back the other way as NDJSON lines:

  {"fleet": N}                          resize the active fleet (pre-spawned bodies park/wake)
  {"demand": d}                         change order pressure (idle gap between tasks)
  {"patch": {"extra_edges":[{"from","to"}], ...}}   change layout: rebuild navgraph + re-route

so a change the operator makes in Three.js is applied to the running physics twin WITHOUT a
reboot. Coordinates are the scenario's navgraph frame, so poses line up with the viewer twin.

  python -m renderer.physx_stream --scenario braess_dev --robots 9 --port 8765 --mock
  <iv-python> -m renderer.physx_stream --scenario real_full_warehouse --robots 12 --port 8765
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import select
import socket
import time
from collections import deque
from pathlib import Path

import numpy as np

from renderer.avoidance import collision_free_velocities
from sim.navgraph import NavGraph

# Same taxonomy/order the baked DES export uses (renderer/export_tracks.py), so the viewer's
# Fleet-Status rows light up identically whether the feed is replay or live.
INTENT_ORDER = ["carrying", "to_pick", "servicing", "to_charge", "charging", "queued", "idle"]
DEMAND_MAX = 3.0   # matches the viewer demand slider's top end


def _load_config(scenario: str) -> dict:
    """Load a scenario config WITHOUT sim.config/engine.store, so this runs in the bare
    Isaac container (no jsonschema). Fills only the layout defaults NavGraph needs."""
    for base in ("eval/dev_scenarios", "examples"):
        p = Path(base) / f"{scenario}.config.json"
        if p.exists():
            cfg = json.loads(p.read_text(encoding="utf-8"))
            cfg.setdefault("layout", {}).setdefault("extra_edges", [])
            cfg["layout"].setdefault("edge_overrides", [])
            return cfg
    raise SystemExit(f"scenario '{scenario}' not found in eval/dev_scenarios/ or examples/")


class Fleet:
    """A live operational model on a scenario's navgraph (integrator-agnostic).

    Holds MAX rigid-body slots; `active` of them are working, the rest are parked off-floor
    so the fleet can grow/shrink live without spawning/destroying bodies. Per-robot task
    cycle drives both the motion (preferred velocity -> ORCA) and the honest fleet telemetry.
    """

    def __init__(self, config: dict, n_robots: int, *, max_robots: int = 12,
                 speed: float = 1.2, radius: float = 0.5, seed: int = 0):
        self.config = copy.deepcopy(config)
        self.speed = speed
        self.radius = radius
        self.rng = np.random.default_rng(seed)
        self.max = max(int(max_robots), int(n_robots))
        self.active = max(1, min(int(n_robots), self.max))
        self.demand = self._cfg_demand(self.config)
        self.t = 0.0

        self._build_graph(self.config)
        N = self.max
        # park column off to the side of the floor (negative x = outside the aisles)
        ymax = float(self.nodes_y.max()) if len(self.nodes_y) else 8.0
        self.park = np.column_stack([np.full(N, -4.0),
                                     np.linspace(0.0, max(ymax, 1.0), N)])
        start = self.rng.integers(0, len(self.nodes), N)
        self.pos = self.nodes[start] + self.rng.normal(0, 0.05, (N, 2))
        # inactive robots begin parked
        for i in range(self.active, N):
            self.pos[i] = self.park[i]
        self.vel = np.zeros((N, 2))
        self.ids = [f"amr_{i:02d}" for i in range(N)]

        # per-robot task state
        self.task = ["idle"] * N        # fetch | carry | charge | idle
        self.phase = ["moving"] * N     # moving | dwell
        self.carry = [False] * N
        self.batt = [1.0] * N
        self.goal_kind = ["pick"] * N
        self.path: list[list[int]] = [[] for _ in range(N)]
        self.wp = [0] * N
        self.dwell_until = [0.0] * N
        self.order_start = [0.0] * N
        self.stall = [0.0] * N

        # telemetry
        self.completions: deque = deque(maxlen=600)   # completion timestamps (s)
        self.latency: deque = deque(maxlen=200)       # order durations (s)

        for i in range(self.active):
            self._assign_task(i)

    # ---- graph / config ---------------------------------------------------------
    @staticmethod
    def _cfg_demand(cfg: dict) -> float:
        d = cfg.get("demand")
        if isinstance(d, dict):
            d = d.get("orders_per_min") or d.get("rate")
        try:
            return float(d)
        except (TypeError, ValueError):
            return 1.0

    def _build_graph(self, cfg: dict) -> None:
        self.g = NavGraph(cfg)
        self.nodes = np.asarray(self.g.node_xy, dtype=float)
        self.nodes_y = self.nodes[:, 1]
        self.stn = {"pick": [], "pack": [], "charge": [], "dock": []}
        for kind in self.stn:
            for s in cfg.get("stations", {}).get(kind, []) or []:
                try:
                    self.stn[kind].append(self.g.node_index(s["node"]))
                except Exception:
                    pass
        if not self.stn["pick"]:
            self.stn["pick"] = list(range(len(self.nodes)))
        if not self.stn["pack"]:
            self.stn["pack"] = self.stn["pick"]
        self.charge_cap = len(self.stn["charge"])

    def _nearest(self, p) -> int:
        return int(np.argmin(((self.nodes - p) ** 2).sum(axis=1)))

    def _route(self, i: int, dst: int) -> None:
        src = self._nearest(self.pos[i])
        try:
            edges = self.g.shortest_path(src, dst)
        except Exception:
            edges = []
        seq = [edges[0][0]] + [e[1] for e in edges] if edges else [src]
        self.path[i] = seq
        self.wp[i] = 1 if len(seq) > 1 else 0

    def _pick_goal(self, kind: str) -> int:
        pool = self.stn.get(kind) or self.stn["pick"]
        return int(self.rng.choice(pool))

    def _svc(self) -> float:
        return 2.5

    def _idle_gap(self) -> float:
        # slack demand => robots idle between orders; full demand => no idle
        return float(max(0.0, 5.0 * (1.0 - self.demand / DEMAND_MAX)))

    # ---- task cycle -------------------------------------------------------------
    def _assign_task(self, i: int) -> None:
        if self.batt[i] < 0.25 and self.stn["charge"]:
            self.task[i] = "charge"
            self.goal_kind[i] = "charge"
            self.carry[i] = False
            self._route(i, self._pick_goal("charge"))
        else:
            self.task[i] = "fetch"
            self.goal_kind[i] = "pick"
            self.carry[i] = False
            self.order_start[i] = self.t
            self._route(i, self._pick_goal("pick"))
        self.phase[i] = "moving"

    def _on_arrive(self, i: int) -> None:
        self.phase[i] = "dwell"
        self.dwell_until[i] = self.t + (6.0 if self.task[i] == "charge" else self._svc())

    def _transition(self, i: int) -> None:
        t = self.task[i]
        if t == "fetch":                      # picked up -> carry to a pack station
            self.task[i] = "carry"
            self.goal_kind[i] = "pack"
            self.carry[i] = True
            self._route(i, self._pick_goal("pack"))
            self.phase[i] = "moving"
        elif t == "carry":                    # order completed at pack
            self.carry[i] = False
            self.completions.append(self.t)
            self.latency.append(self.t - self.order_start[i])
            self.batt[i] -= 0.12
            gap = self._idle_gap()
            if gap > 0.05:
                self.task[i] = "idle"
                self.phase[i] = "dwell"
                self.dwell_until[i] = self.t + gap
            else:
                self._assign_task(i)
        else:                                 # charge done, or idle gap elapsed
            if t == "charge":
                self.batt[i] = 1.0
            self._assign_task(i)

    # ---- one logic step: returns ORCA-corrected velocities for ALL slots --------
    def plan(self, dt: float) -> np.ndarray:
        self.t += dt
        arrive = self.radius * 1.2
        pref = np.zeros((self.max, 2))
        for i in range(self.max):
            if i >= self.active:              # parked: steer to the park column, then rest
                d = self.park[i] - self.pos[i]
                n = np.linalg.norm(d)
                if n > self.radius:
                    pref[i] = d / n * self.speed
                continue
            if self.phase[i] == "dwell":
                if self.t >= self.dwell_until[i]:
                    self._transition(i)
                if self.phase[i] == "dwell":  # still dwelling -> hold position
                    continue
            # moving
            seq = self.path[i]
            while self.wp[i] < len(seq) and \
                    np.linalg.norm(self.nodes[seq[self.wp[i]]] - self.pos[i]) < arrive:
                self.wp[i] += 1
            if self.wp[i] >= len(seq):
                self._on_arrive(i)
                continue
            tgt = self.nodes[seq[min(self.wp[i], len(seq) - 1)]]
            d = tgt - self.pos[i]
            n = np.linalg.norm(d)
            if n > 1e-6:
                pref[i] = d / n * self.speed
        v = collision_free_velocities(self.pos, self.vel, pref, radius=self.radius,
                                      max_speed=self.speed, time_horizon=2.0, dt=dt)
        # stall accounting: a robot that *should* move but is ~stopped is "queued" (blocked)
        for i in range(self.active):
            if self.phase[i] == "moving" and float(np.hypot(v[i, 0], v[i, 1])) < 0.12:
                self.stall[i] += dt
            else:
                self.stall[i] = 0.0
        return v

    # ---- live control -----------------------------------------------------------
    def set_active(self, n: int) -> None:
        n = max(1, min(self.max, int(n)))
        if n > self.active:
            for i in range(self.active, n):
                self.batt[i] = 1.0
                self._assign_task(i)
        self.active = n

    @staticmethod
    def _dig(src: dict, *keys):
        for k in keys:
            v = src.get(k)
            if v is not None:
                return v
        return None

    def apply_control(self, msg: dict) -> None:
        """Accept operator control in any shape the rest of the stack uses: bare
        ({"fleet":N,"demand":d}), the dot-path config patch apply_patch() emits
        ({"layout.extra_edges":[...], "fleet.amr_count":N}), or a nested layout dict."""
        if not isinstance(msg, dict):
            return
        patch = msg.get("patch") if isinstance(msg.get("patch"), dict) else {}
        src = dict(patch)
        for k, v in msg.items():                 # top-level keys also accepted (viewer sends bare)
            if k != "patch":
                src[k] = v
        fleet = self._dig(src, "fleet.amr_count", "fleet")
        if fleet is not None and not isinstance(fleet, dict):
            try:
                self.set_active(int(fleet))
            except (TypeError, ValueError):
                pass
        dem = self._dig(src, "demand.arrival_rate_per_min", "demand.orders_per_min", "demand")
        if dem is not None and not isinstance(dem, dict):
            try:
                self.demand = max(0.0, float(dem))
            except (TypeError, ValueError):
                pass
        ee = self._dig(src, "layout.extra_edges", "extra_edges")
        eo = self._dig(src, "layout.edge_overrides", "edge_overrides")
        layout = src.get("layout") if isinstance(src.get("layout"), dict) else None
        if layout is not None:
            ee = layout.get("extra_edges") if ee is None else ee
            eo = layout.get("edge_overrides") if eo is None else eo
        if ee is not None or eo is not None:
            self._relayout(extra_edges=ee, edge_overrides=eo)

    def _relayout(self, *, extra_edges=None, edge_overrides=None) -> None:
        cfg = copy.deepcopy(self.config)
        cfg.setdefault("layout", {})
        if extra_edges is not None:
            cfg["layout"]["extra_edges"] = extra_edges
        if edge_overrides is not None:
            cfg["layout"]["edge_overrides"] = edge_overrides
        try:
            probe = NavGraph(cfg)            # validate before committing
        except Exception:
            return
        self.config = cfg
        self.g = probe
        self.nodes = np.asarray(self.g.node_xy, dtype=float)
        self.nodes_y = self.nodes[:, 1]
        self._build_graph(cfg)              # re-index stations on the new graph
        for i in range(self.active):        # re-route everyone on the new topology
            if self.phase[i] == "moving":
                self._route(i, self._pick_goal(self.goal_kind[i]))

    # ---- telemetry --------------------------------------------------------------
    def _intent(self, i: int) -> str:
        if self.task[i] == "charge":
            return "charging" if self.phase[i] == "dwell" else "to_charge"
        if self.task[i] == "idle":
            return "idle"
        if self.phase[i] == "dwell":
            return "servicing"
        if self.stall[i] > 1.0:
            return "queued"
        return "carrying" if self.task[i] == "carry" else "to_pick"

    def _throughput(self) -> float:
        window = 120.0
        recent = sum(1 for c in self.completions if c > self.t - window)
        if self.t < window:                 # ramp: annualize what we have so far
            return recent / max(self.t, 1.0) * 3600.0
        return recent * (3600.0 / window)

    def _p95_min(self) -> float:
        if not self.latency:
            return 0.0
        return float(np.percentile(np.asarray(self.latency), 95)) / 60.0

    def kpis(self) -> dict:
        counts = {k: 0 for k in INTENT_ORDER}
        carrying = 0
        busy_chg = 0
        for i in range(self.active):
            counts[self._intent(i)] += 1
            if self.carry[i]:
                carrying += 1
            if self.task[i] == "charge" and self.phase[i] == "dwell":
                busy_chg += 1
        return {
            "robots": self.active,
            "fleet": self.active,
            "intent": counts,
            "carrying": carrying,
            "throughput": round(self._throughput(), 1),
            "p95_latency": round(self._p95_min(), 1),
            "chargers": {"busy": busy_chg, "cap": self.charge_cap},
            "source": "physx",
        }

    def frame(self) -> dict:
        agents = {}
        for i in range(self.active):
            vx, vy = self.vel[i]
            th = math.atan2(vy, vx) if (vx * vx + vy * vy) > 1e-6 else 0.0
            agents[self.ids[i]] = [round(float(self.pos[i, 0]), 3),
                                   round(float(self.pos[i, 1]), 3), round(th, 3)]
        return {"t_sim": round(self.t, 3), "agents": agents, "kpis": self.kpis()}


# ── integrators ────────────────────────────────────────────────────────────────
def step_mock(fleet: Fleet, dt: float) -> None:
    v = fleet.plan(dt)
    fleet.vel = v
    fleet.pos = fleet.pos + v * dt


def make_isaac_integrator(fleet: Fleet, dt: float):
    """Real PhysX rigid bodies on a ground plane (navgraph coords). Mirrors the Isaac
    API proven by renderer/physx_spike_a. Pre-spawns ALL fleet.max bodies so the active
    fleet can grow/shrink live. Returns a step() closure. NOTE-marked lines touch Isaac."""
    from isaacsim import SimulationApp                        # NOTE: Isaac entry point
    app = SimulationApp({"headless": True})
    from isaacsim.core.api import World                       # NOTE
    from isaacsim.core.api.objects import DynamicCuboid, GroundPlane  # NOTE
    from pxr import UsdGeom
    import omni.usd

    world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    GroundPlane(prim_path="/World/ground", size=600.0)
    HALF = fleet.radius / math.sqrt(2.0)
    bodies = []
    for i in range(fleet.max):
        bodies.append(DynamicCuboid(
            prim_path=f"/World/r_{i:02d}", name=f"r_{i:02d}",
            position=np.array([fleet.pos[i, 0], fleet.pos[i, 1], HALF + 0.02]),
            size=2 * HALF, mass=120.0))
    world.reset()
    for _ in range(20):
        world.step(render=False)

    def step():
        v = fleet.plan(dt)
        for i, b in enumerate(bodies):
            b.set_linear_velocity(np.array([v[i, 0], v[i, 1], 0.0]))  # NOTE
        world.step(render=False)                              # NOTE: physics only
        for i, b in enumerate(bodies):
            p = b.get_world_pose()[0]                         # NOTE: (pos, quat)
            fleet.pos[i, 0], fleet.pos[i, 1] = float(p[0]), float(p[1])
        fleet.vel = v
    return step, app


def serve(args) -> None:
    cfg = _load_config(args.scenario)
    n = args.robots or cfg["fleet"]["amr_count"]
    fleet = Fleet(cfg, n, max_robots=max(args.max_robots, n),
                  speed=args.speed, radius=args.radius)
    dt = 1.0 / args.hz

    if args.mock:
        step = lambda: step_mock(fleet, dt)
        app = None
    else:
        step, app = make_isaac_integrator(fleet, dt)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(8)
    print(f"[physx_stream] READY host={socket.gethostname()} port={args.port} "
          f"robots={fleet.active}/{fleet.max} scenario={args.scenario} mock={args.mock}",
          flush=True)

    t0 = time.monotonic()
    deadline = t0 + args.seconds if args.seconds else None
    try:
        while deadline is None or time.monotonic() < deadline:
            srv.settimeout(5.0)
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            print(f"[physx_stream] client {addr} connected", flush=True)
            conn.settimeout(5.0)   # a stalled/half-open client must not wedge the loop forever
            ctrl_buf = b""
            try:
                while deadline is None or time.monotonic() < deadline:
                    tick = time.monotonic()
                    # drain any inbound control (non-blocking) before stepping
                    try:
                        r, _, _ = select.select([conn], [], [], 0)
                    except (OSError, ValueError):
                        break
                    if r:
                        data = conn.recv(65536)
                        if not data:
                            raise ConnectionResetError
                        ctrl_buf += data
                        while b"\n" in ctrl_buf:
                            line, ctrl_buf = ctrl_buf.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                fleet.apply_control(json.loads(line))
                                print(f"[physx_stream] control {line[:160].decode('utf-8','replace')}",
                                      flush=True)
                            except ValueError:
                                pass
                    step()
                    out = (json.dumps(fleet.frame()) + "\n").encode("utf-8")
                    conn.sendall(out)
                    sleep = dt - (time.monotonic() - tick)
                    if sleep > 0:
                        time.sleep(sleep)
            except (BrokenPipeError, ConnectionResetError, OSError):
                print("[physx_stream] client disconnected", flush=True)
            finally:
                conn.close()
    finally:
        srv.close()
        if app is not None:
            app.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="physx_stream", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="braess_dev")
    ap.add_argument("--robots", type=int, default=0)
    ap.add_argument("--max-robots", dest="max_robots", type=int, default=12,
                    help="pre-spawned body count = ceiling for live fleet resize")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=1.2)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until killed")
    ap.add_argument("--mock", action="store_true", help="kinematic (no Isaac/GPU)")
    serve(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
