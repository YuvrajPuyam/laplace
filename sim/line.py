"""sim/line.py - serial production-line simulator (the manufacturing domain).

A NEW domain alongside the warehouse DES (it touches no warehouse contract). Models a
tandem line of K machines M0->M1->...->M(K-1) with finite inter-machine buffers under
BLOCKING-AFTER-SERVICE (BAS): a machine that finishes a part while its downstream buffer
is full HOLDS that part (status BLOCKED) until a slot frees. Exponential service. For the
validation case the first machine is never starved (saturated raw-part supply) and the
last is never blocked (unlimited output).

CONVENTION (a dynamics choice - flagged Yuv-gated; documented + validated, never silent):
`buffers[i]` = number of intermediate slots between machine i and i+1, NOT counting the
parts held *inside* the machines. With BAS + saturated source this makes the two-machine
steady-state throughput  TH = mu2 * (1 - (1-r)/(1 - r**(N+3))),  r = mu1/mu2  - the
birth-death chain on states 0..N+2 that tests/test_line_validation.py checks against an
independently-built CTMC.

DETERMINISM: each machine draws its exponential service times from its OWN RNG substream
(numpy SeedSequence.spawn keyed by machine index), drawn lazily at service start. Two
configs differing only in buffering therefore see the same per-machine service streams
(common random numbers), and identical (rates, buffers, seed) reproduce byte-identically.
"""

from __future__ import annotations

import heapq

import numpy as np

FREE, BUSY, BLOCKED = 0, 1, 2


def run_line(service_rates, buffers, *, sim_time: float, warmup: float = 0.0,
             seed: int = 0) -> dict:
    """Simulate a saturated tandem line under BAS and return steady-state throughput.

    service_rates: list of mu_i (parts per unit time), length K >= 1.
    buffers:       list of intermediate buffer capacities, length K-1.
    Throughput = departures from the last machine within [warmup, sim_time] divided by
    (sim_time - warmup). Pure function; deterministic in (service_rates, buffers, seed).
    """
    mu = [float(m) for m in service_rates]
    K = len(mu)
    if K < 1:
        raise ValueError("need at least one machine")
    if any(m <= 0 for m in mu):
        raise ValueError("service rates must be positive")
    cap = [int(b) for b in buffers]
    if len(cap) != K - 1:
        raise ValueError(f"buffers must have length K-1={K - 1}, got {len(cap)}")
    if any(c < 0 for c in cap):
        raise ValueError("buffer capacities must be >= 0")

    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(K)]
    status = [FREE] * K
    buf = [0] * (K - 1)
    heap: list[tuple[float, int, int]] = []
    seq = 0
    now = 0.0

    def start(i: int) -> None:
        nonlocal seq
        status[i] = BUSY
        dt = float(rngs[i].exponential(1.0 / mu[i]))
        heapq.heappush(heap, (now + dt, seq, i))
        seq += 1

    def sweep() -> None:
        # Deterministic fixed-order repeat-until-stable: start every FREE machine that
        # has input; pulling from buf[i-1] frees a slot that unblocks a BLOCKED M(i-1).
        changed = True
        while changed:
            changed = False
            for i in range(K):
                if status[i] != FREE:
                    continue
                if i == 0:                       # saturated source: always has input
                    start(0)
                    changed = True
                elif buf[i - 1] > 0:
                    buf[i - 1] -= 1
                    if status[i - 1] == BLOCKED:  # deposit the held part into the freed slot
                        buf[i - 1] += 1
                        status[i - 1] = FREE
                    start(i)
                    changed = True
                elif status[i - 1] == BLOCKED:    # zero-buffer / full-then-drained: take the
                    status[i - 1] = FREE          # blocked machine's held part directly
                    start(i)
                    changed = True

    sweep()  # prime the line

    post_dep = 0
    while heap and heap[0][0] <= sim_time:
        now, _, i = heapq.heappop(heap)
        if i == K - 1:                           # completion at the sink == a departure
            status[i] = FREE
            if now >= warmup:
                post_dep += 1
        elif buf[i] < cap[i]:                    # push downstream, machine free
            buf[i] += 1
            status[i] = FREE
        else:                                    # downstream full -> hold the part (BAS)
            status[i] = BLOCKED
        sweep()

    window = sim_time - warmup
    throughput = post_dep / window if window > 0 else 0.0
    return {
        "throughput": throughput,
        "departures_post_warmup": post_dep,
        "window": window,
        "K": K,
        "service_rates": mu,
        "buffers": cap,
        "seed": seed,
    }
