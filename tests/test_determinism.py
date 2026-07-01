"""Non-negotiable #2: identical (config, seed) => identical event logs,
and the CRN pairing property: the order-arrival stream depends only on
(seed, demand params), never on layout or fleet."""

import json

from sim.config import apply_patch, fill_defaults
from sim.engine import Engine
from sim.stream import generate_order_stream


def _events(config, seed):
    return Engine(fill_defaults(config), seed).run()


def test_identical_config_seed_identical_log(baseline_config):
    rows_a = _events(baseline_config, 17)
    rows_b = _events(baseline_config, 17)
    assert rows_a == rows_b  # exact tuple equality, byte-identical payloads


def test_different_seed_different_log(baseline_config):
    assert _events(baseline_config, 17) != _events(baseline_config, 18)


def test_parquet_round_trip_identical(baseline_config, tmp_path):
    from sim import events as ev
    rows = _events(baseline_config, 7)
    path = tmp_path / "x.parquet"
    ev.write_events(rows, path)
    assert ev.read_events(path) == rows


def _arrival_stream_from_log(rows):
    return [(t, json.loads(p)["pick"], eid)
            for t, etype, eid, event, loc, p in rows if event == "order_arrived"]


def test_crn_pairing_across_fleet_and_layout(baseline_config, braess_patch):
    """Same seed must yield the identical arrival stream under fleet and
    layout changes (this is what makes paired comparisons valid)."""
    seed = 23
    base = _arrival_stream_from_log(_events(baseline_config, seed))

    more_amrs = apply_patch(baseline_config, {"fleet.amr_count": 7,
                                              "fleet.speed_mps": 2.5})
    assert _arrival_stream_from_log(_events(more_amrs, seed)) == base

    braess = apply_patch(baseline_config, braess_patch["patch"])
    assert _arrival_stream_from_log(_events(braess, seed)) == base

    routing = apply_patch(baseline_config, {"fleet.routing": "congestion_aware"})
    assert _arrival_stream_from_log(_events(routing, seed)) == base


def test_stream_independent_of_layout_params():
    a = generate_order_stream(5, 3.0, 480)
    b = generate_order_stream(5, 3.0, 480)
    assert a == b
    assert generate_order_stream(6, 3.0, 480) != a


def test_service_draws_paired_across_configs(baseline_config):
    """z draws ride with the order: same seed => same service randomness."""
    s1 = generate_order_stream(11, 2.0, 240)
    s2 = generate_order_stream(11, 2.0, 240)
    assert [(d.z_pick, d.z_pack) for d in s1] == [(d.z_pick, d.z_pack) for d in s2]
