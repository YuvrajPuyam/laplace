import json

import pytest

from sim.config import fill_defaults, validate_result
from sim.runner import run_rollout


@pytest.fixture(scope="module")
def rollout(baseline_config, tmp_path_factory):
    log_dir = tmp_path_factory.mktemp("logs")
    return run_rollout(baseline_config, 17, log_dir)


def test_result_validates_against_contract(rollout):
    result, _ = rollout
    validate_result(result)


def test_throughput_consistent_with_counts(rollout, baseline_config):
    result, _ = rollout
    cfg = fill_defaults(baseline_config)
    window_hr = (cfg["horizon"]["sim_minutes"] - cfg["horizon"]["warmup_minutes"]) / 60.0
    m = result["metrics"]
    assert m["throughput_orders_per_hr"] == pytest.approx(
        m["orders_completed"] / window_hr, abs=1e-3)


def test_throughput_bounded_by_arrival_rate(rollout, baseline_config):
    # baseline_small is heavily overloaded (lambda far above capacity), so
    # throughput must be positive but well below the arrival rate; the
    # rate-matching check for a STABLE system lives in test_mmc_validation.
    result, _ = rollout
    rate_hr = fill_defaults(baseline_config)["demand"]["arrival_rate_per_min"] * 60
    assert 0 < result["metrics"]["throughput_orders_per_hr"] <= rate_hr
    assert result["metrics"]["orders_abandoned"] > 0


def test_latency_percentiles_ordered(rollout):
    m = rollout[0]["metrics"]
    assert 0 < m["p50_order_latency_min"] <= m["p95_order_latency_min"]


def test_percentages_in_range(rollout):
    m = rollout[0]["metrics"]
    for key in ("amr_utilization_pct", "deadhead_pct", "charge_downtime_pct"):
        assert 0.0 <= m[key] <= 100.0, key
    for e in m["edge_congestion_top5"]:
        assert 0.0 <= e["occupancy_pct"] <= 100.0


def test_metrics_recomputable_from_parquet(rollout, baseline_config):
    """Metrics are a pure function of the event log + config (consumer #1
    of the one-format rule)."""
    from sim.events import read_events
    from sim.metrics import compute_metrics
    result, rows = rollout
    rows_rt = read_events(result["event_log_uri"])
    m = compute_metrics(rows_rt, fill_defaults(baseline_config))
    assert m == result["metrics"]


def test_station_waits_cover_all_service_starts(rollout):
    from sim.metrics import station_wait_samples
    result, rows = rollout
    n_starts = sum(1 for r in rows if r[3] == "service_start")
    waits = station_wait_samples(rows)
    assert sum(len(v) for v in waits.values()) == n_starts
    assert all(w >= 0 for v in waits.values() for w in v)


def test_abandoned_counts(rollout):
    result, rows = rollout
    m = result["metrics"]
    warmup = 30.0
    arrived_measured = sum(1 for r in rows if r[3] == "order_arrived" and r[0] >= warmup)
    assert m["orders_completed"] + m["orders_abandoned"] == arrived_measured


def test_run_many_paired_and_parallel(baseline_config, tmp_path):
    from sim.runner import run_many
    results = run_many(baseline_config, [1, 2], log_dir=tmp_path,
                       write_log=False, max_workers=2)
    assert [r["seed"] for r in results] == [1, 2]
    assert results[0]["config_hash"] == results[1]["config_hash"]
    assert results[0]["metrics"] != results[1]["metrics"]
