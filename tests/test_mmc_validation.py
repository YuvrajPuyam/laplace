"""Non-negotiable #3: degenerate configs must match queueing closed forms.

Service times are lognormal (Contract A), so the exact closed form for a
single-slot station is M/G/1 Pollaczek-Khinchine (not M/M/1 — exponential
service is not expressible in the schema):

    Wq = lambda * E[S^2] / (2 * (1 - rho)),   rho = lambda * E[S]

For multi-slot stations we check the Allen-Cunneen M/G/c approximation with a
looser tolerance.

Degenerate config: one pick station fed (effectively) directly by the Poisson
order stream — a large, fast fleet and a fast multi-slot pack station make
travel/fleet/pack delays non-binding, so arrivals at the pick station are the
Poisson arrivals shifted by (near-constant) travel.
"""

import math

import pytest

from sim.config import fill_defaults, validate_config
from sim.engine import Engine
from sim.metrics import station_wait_samples


def _degenerate_config(lam: float, mu: float, sigma: float, slots: int) -> dict:
    cfg = {
        "schema_version": "1.0",
        "scenario_id": "mg1_degenerate",
        "layout": {"grid": {"aisles": 2, "aisle_length_m": 10, "cross_aisles": [0, 10]}},
        "stations": {
            "pick": [{"id": "P1", "node": "A1_05", "slots": slots,
                      "service_lognorm": [mu, sigma]}],
            # pack: 4 fast slots so it never queues
            "pack": [{"id": "K1", "node": "A2_05", "slots": 4,
                      "service_lognorm": [-3.0, 0.1]}],
            "charge": [{"id": "C1", "node": "A1_00", "slots": 4}],
            "dock": [{"id": "D1", "node": "A2_10"}],
        },
        "fleet": {"amr_count": 12, "speed_mps": 3.0, "battery_capacity_m": 20000,
                  "charge_minutes": 1, "routing": "shortest_path"},
        "demand": {"arrival_rate_per_min": lam, "pack_assignment": "round_robin"},
        "horizon": {"sim_minutes": 1440, "warmup_minutes": 120},
    }
    validate_config(cfg)
    return fill_defaults(cfg)


def _mean_wait(cfg: dict, seeds: range) -> float:
    waits = []
    warmup = cfg["horizon"]["warmup_minutes"]
    for seed in seeds:
        rows = Engine(cfg, seed).run()
        waits.extend(station_wait_samples(rows, warmup=warmup).get("P1", []))
    assert len(waits) > 5000, "not enough samples for a stable mean"
    return sum(waits) / len(waits)


@pytest.mark.slow
def test_mg1_pollaczek_khinchine():
    lam, mu, sigma = 1.5, math.log(0.5) - 0.125, 0.5  # E[S]=0.5 min, rho=0.75
    es = math.exp(mu + sigma**2 / 2)
    es2 = math.exp(2 * mu + 2 * sigma**2)
    rho = lam * es
    wq_theory = lam * es2 / (2 * (1 - rho))

    cfg = _degenerate_config(lam, mu, sigma, slots=1)
    wq_sim = _mean_wait(cfg, range(20))
    assert wq_sim == pytest.approx(wq_theory, rel=0.10), (
        f"M/G/1 P-K: sim {wq_sim:.4f} vs theory {wq_theory:.4f} (rho={rho})")


@pytest.mark.slow
def test_mgc_allen_cunneen():
    # 2 slots, rho = lam*E[S]/c = 0.75
    lam, mu, sigma = 3.0, math.log(0.5) - 0.125, 0.5
    c = 2
    es = math.exp(mu + sigma**2 / 2)
    es2 = math.exp(2 * mu + 2 * sigma**2)
    cv2 = es2 / es**2 - 1.0
    rho = lam * es / c

    # Erlang C for M/M/c
    a = lam * es  # offered load
    inv_erlang_b = sum((a**k / math.factorial(k)) for k in range(c + 1)) / (a**c / math.factorial(c))
    erlang_b = 1.0 / inv_erlang_b
    erlang_c = erlang_b / (1.0 - rho + rho * erlang_b)
    wq_mmc = erlang_c * es / (c * (1.0 - rho))
    wq_theory = wq_mmc * (1.0 + cv2) / 2.0  # Allen-Cunneen correction

    cfg = _degenerate_config(lam, mu, sigma, slots=c)
    wq_sim = _mean_wait(cfg, range(20))
    assert wq_sim == pytest.approx(wq_theory, rel=0.20), (
        f"M/G/c Allen-Cunneen: sim {wq_sim:.4f} vs theory {wq_theory:.4f}")


@pytest.mark.slow
def test_throughput_matches_arrival_rate_when_stable():
    cfg = _degenerate_config(1.5, math.log(0.5) - 0.125, 0.5, slots=1)
    from sim.metrics import compute_metrics
    rows = Engine(cfg, 0).run()
    m = compute_metrics(rows, cfg)
    assert m["throughput_orders_per_hr"] == pytest.approx(1.5 * 60, rel=0.05)
    assert m["orders_abandoned"] <= 0.02 * (m["orders_completed"] + m["orders_abandoned"])
