"""engine/playout.py - laptop-side delayed-feed playout for the live twin (Track L).

Turns a jittery, possibly-out-of-order pose stream from the GPU cluster into a smooth
state played back a fixed delay D behind live, and fans the SAME sampled state to both
consumers (Three.js viewer + LLM operator) so they observe byte-identical state. Pure,
deterministic (the only impurity, wall time, is an injected Clock), and presentational /
control only - it touches no sim contract and never synthesizes a metric (KPIs are carried
from a real frame, never interpolated).

Design (see the steering spec): anchor on the newest frame; playout_t advances at real
rate but stays >= D behind newest; interpolate agent x,y and shortest-arc theta between
bracketing frames; hold + mark STALE on a producer stall; ease back (bounded catch-up) on
resume; the Relay samples ONCE per tick and publishes one serialized state to all sinks
(closing the timing-skew SPOF).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Frame:
    t_sim: float
    data: dict                      # {"agents": {id: [x, y, theta]}, "kpis": {...}}


@dataclass(frozen=True)
class Sampled:
    playout_t: float | None
    state: dict | None
    status: str                     # priming | live | stale | disconnected
    stale_for: float
    source: str                     # none | priming | exact | interp | hold


class Clock:
    def now(self) -> float:
        raise NotImplementedError


class ManualClock(Clock):
    def __init__(self, t: float = 0.0):
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + alpha * (b - a)


def _interp_agents(lo: dict, hi: dict, alpha: float) -> dict:
    out = {}
    for k, v in lo.items():
        if k in hi:
            w = hi[k]
            out[k] = [_lerp(v[0], w[0], alpha), _lerp(v[1], w[1], alpha),
                      v[2] + alpha * _wrap_to_pi(w[2] - v[2])]
        else:
            out[k] = list(v)        # vanished downstream: carry last-known
    for k, w in hi.items():
        if k not in lo:
            out[k] = list(w)        # appeared: carry
    return out


class JitterBuffer:
    """Sorted-by-t_sim frame store with a PURE interpolating sample()."""

    def __init__(self, prune_margin: float = 3.0):
        self.prune_margin = prune_margin
        self._frames: list[Frame] = []         # sorted by t_sim
        self._last_pruned_t = -math.inf

    def push(self, frame: Frame) -> bool:
        if frame.t_sim < self._last_pruned_t:
            return False                        # too late: already rendered past it
        # dedup last-writer-wins; insert sorted (small buffers -> linear is fine)
        for i, f in enumerate(self._frames):
            if f.t_sim == frame.t_sim:
                self._frames[i] = frame
                return True
            if f.t_sim > frame.t_sim:
                self._frames.insert(i, frame)
                return True
        self._frames.append(frame)
        return True

    @property
    def t_newest(self) -> float | None:
        return self._frames[-1].t_sim if self._frames else None

    @property
    def t_oldest(self) -> float | None:
        return self._frames[0].t_sim if self._frames else None

    def sample(self, playout_t: float | None) -> Sampled:
        if not self._frames or playout_t is None:
            return Sampled(playout_t, None, "priming", 0.0, "none")
        if playout_t <= self._frames[0].t_sim:
            f = self._frames[0]
            return Sampled(playout_t, dict(f.data), "priming", 0.0, "priming")
        if playout_t >= self._frames[-1].t_sim:
            f = self._frames[-1]
            return Sampled(playout_t, dict(f.data), "live", 0.0, "hold")
        # bracket
        lo = self._frames[0]
        hi = self._frames[-1]
        for i in range(len(self._frames) - 1):
            if self._frames[i].t_sim <= playout_t <= self._frames[i + 1].t_sim:
                lo, hi = self._frames[i], self._frames[i + 1]
                break
        if playout_t == lo.t_sim:
            return Sampled(playout_t, dict(lo.data), "live", 0.0, "exact")
        alpha = (playout_t - lo.t_sim) / (hi.t_sim - lo.t_sim)
        state = {"agents": _interp_agents(lo.data.get("agents", {}),
                                          hi.data.get("agents", {}), alpha),
                 "kpis": dict(lo.data.get("kpis", {}))}     # carry earlier KPIs (no lerp)
        return Sampled(playout_t, state, "live", 0.0, "interp")

    def prune(self, playout_t: float) -> int:
        cut = playout_t - self.prune_margin
        keep_from = 0
        for i, f in enumerate(self._frames):
            if f.t_sim < cut:
                keep_from = i               # keep the last one below cut as a left bracket
            else:
                break
        dropped = max(0, keep_from)
        if dropped:
            self._last_pruned_t = self._frames[dropped - 1].t_sim
            self._frames = self._frames[dropped:]
        return dropped


class PlayoutController:
    """wall -> playout_t with a fixed delay, bounded catch-up, and stall/disconnect."""

    def __init__(self, clock: Clock, delay: float = 3.0, max_hold: float = 0.5,
                 h_max: float = 10.0, catchup_slack: float = 1.0,
                 catchup_rate: float = 1.25):
        self.clock = clock
        self.delay = delay
        self.max_hold = max_hold
        self.h_max = h_max
        self.catchup_slack = catchup_slack
        self.catchup_rate = catchup_rate
        self._playout: float | None = None
        self._last_wall: float | None = None
        self._t_newest: float | None = None
        self._newest_wall: float | None = None

    def on_push(self, frame: Frame) -> None:
        now = self.clock.now()
        if self._t_newest is None or frame.t_sim > self._t_newest:
            self._t_newest = frame.t_sim
            self._newest_wall = now

    def playout_t(self, buffer: JitterBuffer) -> float | None:
        if self._t_newest is None or buffer.t_oldest is None:
            return None
        now = self.clock.now()
        target = self._t_newest - self.delay
        if self._playout is None:
            self._playout = max(target, buffer.t_oldest)
            self._last_wall = now
            return self._playout
        dt = max(0.0, now - self._last_wall)
        self._last_wall = now
        depth = self._t_newest - self._playout
        rate = self.catchup_rate if depth > self.delay + self.catchup_slack else 1.0
        nxt = self._playout + rate * dt
        nxt = min(nxt, target)                 # stay >= D behind newest (hold the cushion)
        nxt = min(nxt, self._t_newest)         # never extrapolate past live
        nxt = max(nxt, self._playout)          # monotone non-decreasing
        nxt = max(nxt, buffer.t_oldest)
        self._playout = nxt
        return nxt

    def status(self, buffer: JitterBuffer) -> tuple[str, float]:
        if self._playout is None or buffer.t_oldest is None:
            return ("priming", 0.0)
        stale_for = self.clock.now() - (self._newest_wall or self.clock.now())
        if stale_for > self.h_max:
            return ("disconnected", stale_for)
        if stale_for > self.max_hold:
            return ("stale", stale_for)
        return ("live", stale_for)


class Relay:
    """The single sampler: one playout_t, one sample, one serialized state to all sinks
    (the fan-out purity that kills the timing-skew SPOF)."""

    def __init__(self, buffer: JitterBuffer, controller: PlayoutController):
        self.buffer = buffer
        self.controller = controller
        self._sinks: list = []

    def subscribe(self, sink) -> None:
        self._sinks.append(sink)

    def push(self, frame: Frame) -> bool:
        self.controller.on_push(frame)
        return self.buffer.push(frame)

    def tick(self) -> Sampled:
        pt = self.controller.playout_t(self.buffer)
        sampled = self.buffer.sample(pt)
        status, stale_for = self.controller.status(self.buffer)
        out = Sampled(sampled.playout_t, sampled.state, status, stale_for, sampled.source)
        if pt is not None:
            self.buffer.prune(pt)
        for sink in self._sinks:                # same object handed to every sink
            sink(out)
        return out


@dataclass
class MockProducer:
    """Deterministic frame source with injectable jitter / reorder / drop / stall.
    schedule() returns [(arrival_wall, frame)] - pure given seed (honors CRN: the same
    seed gives the same arrival script regardless of the buffer's delay D)."""
    frames: list[Frame]
    seed: int = 0
    period: float = 1.0 / 15.0          # nominal inter-frame wall spacing
    jitter: float = 0.0                 # uniform +/- jitter on arrival wall
    reorder_prob: float = 0.0
    drop_prob: float = 0.0
    stall_at: tuple[float, float] | None = None   # (t_sim, extra_wall_delay)
    _rng: random.Random = field(default=None, repr=False)

    def schedule(self) -> list[tuple[float, Frame]]:
        rng = random.Random(self.seed)
        out: list[tuple[float, Frame]] = []
        wall = 0.0
        for f in self.frames:
            wall += self.period
            if self.drop_prob and rng.random() < self.drop_prob:
                continue
            arrive = wall + (rng.uniform(-self.jitter, self.jitter) if self.jitter else 0.0)
            if self.reorder_prob and rng.random() < self.reorder_prob:
                arrive += self.period * 1.5     # arrives after the next frame
            if self.stall_at and f.t_sim >= self.stall_at[0]:
                arrive += self.stall_at[1]
            out.append((max(0.0, arrive), f))
        out.sort(key=lambda x: x[0])
        return out
