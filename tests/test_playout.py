"""Tests for engine/playout.py - the delayed-feed jitter buffer + relay, driven entirely
by a mock producer + manual clock (no cluster, deterministic)."""

from __future__ import annotations

import math

from engine.playout import (Frame, JitterBuffer, ManualClock, MockProducer,
                            PlayoutController, Relay)

DT_SIM = 1.0 / 15.0


def _frames(n: int):
    fr = []
    for i in range(n):
        t = i * DT_SIM
        fr.append(Frame(t, {"agents": {"a": [t, 0.0, 0.0]}, "kpis": {"done": i}}))
    return fr


def _run(frames, *, delay=3.0, tick_dt=1 / 30, total_wall=None, **prod_kw):
    clock = ManualClock(0.0)
    buf = JitterBuffer(prune_margin=delay)
    ctrl = PlayoutController(clock, delay=delay, max_hold=0.5)
    relay = Relay(buf, ctrl)
    received: list = []
    relay.subscribe(received.append)
    sched = MockProducer(frames, **prod_kw).schedule()
    total_wall = total_wall if total_wall is not None else sched[-1][0] + delay + 2.0
    si, w, samples = 0, 0.0, []
    while w <= total_wall:
        clock.t = w
        while si < len(sched) and sched[si][0] <= w:
            relay.push(sched[si][1]); si += 1
        samples.append(relay.tick())
        w += tick_dt
    return samples, received


def test_monotone_playout_under_jitter():
    samples, _ = _run(_frames(40), jitter=0.03, seed=1)
    pts = [s.playout_t for s in samples if s.playout_t is not None]
    assert all(b >= a - 1e-9 for a, b in zip(pts, pts[1:])), "playout_t went backward"


def test_no_teleport_bound():
    tick_dt = 1 / 30
    samples, _ = _run(_frames(40), tick_dt=tick_dt, jitter=0.02, seed=2)
    xs = [s.state["agents"]["a"][0] for s in samples
          if s.state and "a" in s.state["agents"]]
    bound = 1.0 * tick_dt * 1.25 + 1e-6        # v=1 m/s, catch-up 1.25x
    assert all(abs(b - a) <= bound for a, b in zip(xs, xs[1:])), "agent teleported"


def test_interpolation_midpoint_and_shortest_arc_theta():
    buf = JitterBuffer()
    buf.push(Frame(0.0, {"agents": {"a": [0.0, 0.0, math.radians(350)]}, "kpis": {"k": 1}}))
    buf.push(Frame(1.0, {"agents": {"a": [10.0, 4.0, math.radians(10)]}, "kpis": {"k": 2}}))
    s = buf.sample(0.5)
    x, y, th = s.state["agents"]["a"]
    assert x == 5.0 and y == 2.0
    # 350deg -> 10deg via the short way passes through 0/360, not 180
    d0 = min(th % (2 * math.pi), 2 * math.pi - th % (2 * math.pi))
    assert d0 < 1e-6, f"theta took the long arc: {math.degrees(th)}"
    assert s.state["kpis"] == {"k": 1}          # KPIs carried from the earlier frame
    assert s.source == "interp"


def test_exact_knot_returns_raw_frame():
    buf = JitterBuffer()
    buf.push(Frame(0.0, {"agents": {"a": [1.0, 2.0, 0.3]}, "kpis": {"k": 9}}))
    buf.push(Frame(1.0, {"agents": {"a": [5.0, 6.0, 0.7]}, "kpis": {"k": 10}}))
    s = buf.sample(1.0)                          # exactly on the newest -> hold raw
    assert s.state["agents"]["a"] == [5.0, 6.0, 0.7]
    assert s.state["kpis"] == {"k": 10}


def test_fanout_consumers_identical():
    clock = ManualClock(0.0)
    buf = JitterBuffer()
    relay = Relay(buf, PlayoutController(clock, delay=1.0))
    viewer, operator = [], []
    relay.subscribe(viewer.append)
    relay.subscribe(operator.append)
    sched = MockProducer(_frames(30), jitter=0.02, seed=3).schedule()
    si, w = 0, 0.0
    while w <= sched[-1][0] + 3.0:
        clock.t = w
        while si < len(sched) and sched[si][0] <= w:
            relay.push(sched[si][1]); si += 1
        relay.tick(); w += 1 / 30
    assert viewer == operator and len(viewer) > 0


def test_fanout_skewed_independent_sampling_would_differ():
    # Teeth for the consistency test: two samplers at slightly different playout_t diverge.
    buf = JitterBuffer()
    buf.push(Frame(0.0, {"agents": {"a": [0.0, 0.0, 0.0]}, "kpis": {}}))
    buf.push(Frame(1.0, {"agents": {"a": [10.0, 0.0, 0.0]}, "kpis": {}}))
    a = buf.sample(0.5).state["agents"]["a"]
    b = buf.sample(0.5 + 1e-3).state["agents"]["a"]
    assert a != b                               # independent skewed sampling => divergence


def test_stall_then_stale_then_resume():
    frames = _frames(30)
    stall_t = frames[12].t_sim
    samples, _ = _run(frames, delay=1.0, jitter=0.0, stall_at=(stall_t, 3.0), seed=4)
    statuses = [s.status for s in samples]
    assert "stale" in statuses, "a 3s stall past a 1s cushion should surface STALE"
    # recovers to live AFTER the stall resumes (it correctly goes stale again only once
    # the whole stream ends and no further frames ever arrive)
    first_stale = statuses.index("stale")
    assert "live" in statuses[first_stale + 1:], "feed never recovered after the stall"
    # no teleport across the whole run (incl. the resume seam)
    xs = [s.state["agents"]["a"][0] for s in samples
          if s.state and "a" in s.state["agents"]]
    assert all(abs(b - a) <= 1.25 * (1 / 30) + 1e-6 for a, b in zip(xs, xs[1:]))


def test_determinism_same_inputs():
    s1, _ = _run(_frames(35), jitter=0.03, seed=7)
    s2, _ = _run(_frames(35), jitter=0.03, seed=7)
    assert s1 == s2


def test_dedup_last_writer_and_drop_too_late():
    buf = JitterBuffer(prune_margin=0.5)
    buf.push(Frame(1.0, {"agents": {}, "kpis": {"v": 1}}))
    buf.push(Frame(1.0, {"agents": {}, "kpis": {"v": 2}}))     # same t_sim -> overwrite
    assert buf.sample(1.0).state["kpis"]["v"] == 2
    buf.push(Frame(2.0, {"agents": {}, "kpis": {}}))
    buf.push(Frame(5.0, {"agents": {}, "kpis": {}}))
    # cut = 5.0 - 0.5 = 4.5; frames 1.0 and 2.0 are below it -> the 1.0 is dropped,
    # advancing last_pruned_t so an even-later straggler is rejected.
    assert buf.prune(5.0) >= 1
    assert buf.push(Frame(0.5, {"agents": {}, "kpis": {}})) is False
