"""Week-1 gate: >= 500x realtime per rollout on one core (spec §8)."""

import time

import pytest

from sim.config import fill_defaults
from sim.engine import Engine
from sim.metrics import compute_metrics


@pytest.mark.slow
def test_500x_realtime(baseline_config):
    cfg = fill_defaults(baseline_config)
    engine = Engine(cfg, 42)  # build outside the timed section: setup isn't the hot loop
    t0 = time.perf_counter()
    rows = engine.run()
    compute_metrics(rows, cfg, engine.graph)  # metrics count as rollout work
    elapsed = time.perf_counter() - t0
    sim_seconds = cfg["horizon"]["sim_minutes"] * 60
    speedup = sim_seconds / elapsed
    print(f"\n{speedup:.0f}x realtime ({elapsed:.2f}s wall for {sim_seconds}s sim, "
          f"{len(rows)} events)")
    assert speedup >= 500, f"only {speedup:.0f}x realtime"
