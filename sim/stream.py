"""Common-random-numbers order stream (the determinism non-negotiable).

The order-arrival stream depends ONLY on (seed, demand params, horizon) —
never on layout or fleet. Paired comparisons across configs rely on this:
the same seed yields the same orders (arrival times, pick-station draw, and
per-stage service-time normals) under any layout/fleet variant.

Per-order pre-drawn randomness:
- t_arrival: Poisson process at arrival_rate_per_min over [0, sim_minutes)
- u_pick in [0,1): mapped to a pick station as floor(u * n_pick) at runtime,
  so the stream itself is independent of the station count
- z_pick, z_pack: standard normals; service time at station S is
  exp(mu_S + sigma_S * z) minutes regardless of WHEN service happens —
  service randomness is paired across configs too.

Pack-station choice is a deterministic policy (round_robin / shortest_queue),
not part of the random stream.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

_ARRIVALS_STREAM = 0xA221


class OrderDraw(NamedTuple):
    t_arrival: float
    u_pick: float
    z_pick: float
    z_pack: float


def generate_order_stream(seed: int, arrival_rate_per_min: float,
                          sim_minutes: float) -> list[OrderDraw]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, _ARRIVALS_STREAM])))
    times: list[float] = []
    t = 0.0
    mean_gap = 1.0 / arrival_rate_per_min
    while True:
        t += rng.exponential(mean_gap)
        if t >= sim_minutes:
            break
        times.append(t)
    n = len(times)
    u_pick = rng.random(n)
    z_pick = rng.standard_normal(n)
    z_pack = rng.standard_normal(n)
    return [OrderDraw(times[i], u_pick[i], z_pick[i], z_pack[i]) for i in range(n)]
