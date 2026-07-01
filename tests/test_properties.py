"""Property tests required by CLAUDE.md: conservation and replay sufficiency
(no teleporting AMRs)."""

import json

import pytest

from sim.config import apply_patch, fill_defaults
from sim.engine import Engine
from sim.replay import ReplayState


def _run(config, seed):
    cfg = fill_defaults(config)
    return Engine(cfg, seed).run(), cfg


def test_conservation(baseline_config):
    """arrived = completed + abandoned at sim_end (in-flight counts as
    abandoned per Contract B.1 'still incomplete at sim end')."""
    rows, _ = _run(baseline_config, 3)
    arrived = sum(1 for r in rows if r[3] == "order_arrived")
    completed = sum(1 for r in rows if r[3] == "order_complete")
    sim_end = [r for r in rows if r[3] == "sim_end"]
    assert len(sim_end) == 1
    assert rows[-1][3] == "sim_end"  # last row, exactly once
    abandoned = json.loads(sim_end[0][5])["orders_abandoned"]
    assert arrived == completed + abandoned
    # every completed order arrived and was assigned exactly once
    assigned = sum(1 for r in rows if r[3] == "task_assigned")
    assert completed <= assigned <= arrived


def test_warmup_marker(baseline_config):
    rows, cfg = _run(baseline_config, 3)
    marks = [r for r in rows if r[3] == "sim_warmup_end"]
    assert len(marks) == 1
    assert marks[0][0] == pytest.approx(cfg["horizon"]["warmup_minutes"])


def test_rows_time_ordered(baseline_config):
    rows, _ = _run(baseline_config, 5)
    ts = [r[0] for r in rows]
    assert all(a <= b for a, b in zip(ts, ts[1:]))


def test_service_slots_never_exceeded(baseline_config):
    rows, cfg = _run(baseline_config, 9)
    slots = {s["id"]: s["slots"] for kind in ("pick", "pack")
             for s in cfg["stations"][kind]}
    busy = {sid: 0 for sid in slots}
    for r in rows:
        if r[3] == "service_start":
            st = json.loads(r[5])["station"]
            busy[st] += 1
            assert busy[st] <= slots[st], f"{st} exceeded slots at t={r[0]}"
        elif r[3] == "service_end":
            busy[json.loads(r[5])["station"]] -= 1


def test_edge_capacity_never_exceeded(baseline_config, braess_patch):
    cfg = fill_defaults(apply_patch(baseline_config, braess_patch["patch"]))
    rows = Engine(cfg, 13).run()
    # reconstruct pool occupancy from depart events (duration = len/speed)
    from sim.navgraph import NavGraph
    g = NavGraph(cfg)
    intervals = []  # (t_start, t_end, pool)
    for t, etype, eid, event, loc, payload in rows:
        if event == "amr_depart_edge":
            p = json.loads(payload)
            u, v = (g.node_index(n) for n in p["edge"].split("->"))
            pi = g.pool_index(u, v)
            dur = g.pools[pi]["length"] / (p["speed_mps"] * 60.0)
            intervals.append((t, t + dur, pi))
    by_pool = {}
    for t0, t1, pi in intervals:
        by_pool.setdefault(pi, []).append((t0, t1))
    eps = 1e-9
    for pi, ivs in by_pool.items():
        cap = g.pools[pi]["capacity"]
        points = sorted({t for iv in ivs for t in iv})
        for pt in points:
            occ = sum(1 for t0, t1 in ivs if t0 <= pt + eps and pt + eps < t1 - eps)
            assert occ <= cap, f"pool {pi} occ {occ} > cap {cap} at t={pt}"


def test_no_teleporting_amrs(baseline_config):
    """events.schema.md replay-sufficiency rule: sampling positions through
    the replay reconstruction, no AMR moves more than speed * dt between
    consecutive samples (plus the queue-offset allowance)."""
    from sim.replay import _QUEUE_SPACING_M, _SLOT_SPACING_M
    rows, cfg = _run(baseline_config, 17)
    rs = ReplayState(rows, cfg)
    speed = cfg["fleet"]["speed_mps"] * 60.0  # m/min
    dt = 2.0 / 60.0  # 2-second samples
    # A parked AMR is anchored at a queue / service-bay / dock OFFSET from its node, because bays
    # are physical (a robot is ~0.8 m wide, so slots sit >= 1 robot-width apart). Snapping to or
    # from such an anchor is a legitimate per-sample discontinuity bounded by the largest offset —
    # this is the "queue-offset allowance" the replay rule intends, NOT a teleport. A real teleport
    # (a wrong node, tens of metres) still exceeds this and is caught.
    n = cfg["fleet"]["amr_count"]
    max_slots = max((st["slots"] for k in ("pick", "pack", "charge")
                     for st in cfg["stations"][k]), default=1)
    anchor_allow = max(_QUEUE_SPACING_M * (n - 1), _SLOT_SPACING_M * max(n - 1, max_slots - 1))
    tol = speed * dt + anchor_allow + 1e-6
    horizon = cfg["horizon"]["sim_minutes"]
    for i in range(cfg["fleet"]["amr_count"]):
        amr_id = f"amr_{i:02d}"
        prev = rs.position(amr_id, 0.0)
        t = dt
        while t <= horizon:
            cur = rs.position(amr_id, t)
            jump = ((cur[0] - prev[0]) ** 2 + (cur[1] - prev[1]) ** 2) ** 0.5
            assert jump <= tol, f"{amr_id} teleported {jump:.2f} m at t={t:.3f}"
            prev = cur
            t += dt
