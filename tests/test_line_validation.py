"""Validation of sim/line.py (manufacturing domain) against closed-form theory.

The M/M/c analogue for the new domain: a serial line under blocking-after-service is a
tandem queue with finite buffers (Buzacott 1967; Buzacott & Shanthikumar, *Stochastic
Models of Manufacturing Systems*). The oracle here is an independently-built continuous-
time Markov chain (CTMC) for the two-machine line, plus the balanced-line rational
fingerprints, the no-buffer closed form, Muth's reversibility property, monotonicity in
buffer size, and the bottleneck limit. This is the line domain's equivalent of
tests/test_mmc_validation.py for the warehouse.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.line import run_line

# Long horizon + a tolerance that absorbs Monte-Carlo noise at this scale.
SIM_TIME = 30000.0
WARMUP = 3000.0
SEED = 7
REL_TOL = 0.035


def _ctmc_two_machine_throughput(mu1: float, mu2: float, N: int) -> float:
    """Exact steady-state throughput of the 2-machine, buffer-N line under BAS with a
    saturated M0 and never-blocked M1 — solved from the birth-death generator directly
    (states 0..N+2; birth mu1 below the top, death mu2 above the bottom)."""
    size = (N + 2) + 1
    Q = np.zeros((size, size))
    for n in range(size):
        if n < size - 1:
            Q[n, n + 1] += mu1
        if n > 0:
            Q[n, n - 1] += mu2
    for n in range(size):
        Q[n, n] = -Q[n].sum()
    A = Q.T.copy()
    A[-1, :] = 1.0                      # replace last balance eq with normalization
    b = np.zeros(size)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    return mu2 * (1.0 - pi[0])          # departure rate = mu2 * P(sink busy)


@pytest.mark.parametrize("mu1,mu2,N", [
    (1.0, 1.0, 0), (1.0, 1.0, 2), (2.0, 1.0, 1),
    (1.0, 2.0, 3), (1.5, 0.8, 2), (0.9, 1.3, 4),
])
def test_matches_ctmc_two_machine(mu1, mu2, N):
    expected = _ctmc_two_machine_throughput(mu1, mu2, N)
    got = run_line([mu1, mu2], [N], sim_time=SIM_TIME, warmup=WARMUP, seed=SEED)["throughput"]
    assert got == pytest.approx(expected, rel=REL_TOL), (
        f"mu1={mu1} mu2={mu2} N={N}: sim {got:.4f} vs CTMC {expected:.4f}")


@pytest.mark.parametrize("N,expected", [
    (0, 2 / 3), (1, 0.75), (2, 0.80), (5, 0.875), (10, 12 / 13),
])
def test_balanced_fingerprints(N, expected):
    # mu=1 balanced line: TH = (N+2)/(N+3), a convention-robust rational fingerprint.
    got = run_line([1.0, 1.0], [N], sim_time=SIM_TIME, warmup=WARMUP, seed=SEED)["throughput"]
    assert got == pytest.approx(expected, rel=REL_TOL)


@pytest.mark.parametrize("mu1,mu2", [(1.0, 1.0), (2.0, 1.0), (0.7, 1.4)])
def test_no_buffer_closed_form(mu1, mu2):
    # Classic two-stage zero-buffer result: mu1*mu2*(mu1+mu2)/(mu1^2+mu1*mu2+mu2^2).
    expected = mu1 * mu2 * (mu1 + mu2) / (mu1 ** 2 + mu1 * mu2 + mu2 ** 2)
    got = run_line([mu1, mu2], [0], sim_time=SIM_TIME, warmup=WARMUP, seed=SEED)["throughput"]
    assert got == pytest.approx(expected, rel=REL_TOL)


def test_single_machine_is_rate():
    got = run_line([1.5], [], sim_time=SIM_TIME, warmup=WARMUP, seed=SEED)["throughput"]
    assert got == pytest.approx(1.5, rel=REL_TOL)


def test_large_buffer_approaches_bottleneck():
    # N -> large: TH -> min(mu1, mu2).
    got = run_line([1.0, 1.6], [40], sim_time=SIM_TIME, warmup=WARMUP, seed=SEED)["throughput"]
    assert got == pytest.approx(1.0, rel=REL_TOL)


def test_reversibility_two_machine_exact():
    # Muth (1979): reversing the line preserves throughput. K=2 -> swap the rates.
    a = _ctmc_two_machine_throughput(1.0, 2.0, 3)
    b = _ctmc_two_machine_throughput(2.0, 1.0, 3)
    assert a == pytest.approx(b, rel=1e-9)


def test_reversibility_three_machine_sim():
    # Reverse machine order AND buffer order; throughput must match (within MC noise).
    fwd = run_line([1.3, 0.9, 1.7], [2, 4], sim_time=SIM_TIME, warmup=WARMUP, seed=SEED)["throughput"]
    rev = run_line([1.7, 0.9, 1.3], [4, 2], sim_time=SIM_TIME, warmup=WARMUP, seed=SEED)["throughput"]
    assert fwd == pytest.approx(rev, rel=0.045)


def test_monotonic_in_buffer_exact():
    # Throughput is non-decreasing in buffer size (exact, via the CTMC oracle).
    ths = [_ctmc_two_machine_throughput(1.0, 1.1, N) for N in range(0, 8)]
    assert all(b >= a - 1e-12 for a, b in zip(ths, ths[1:])), ths


def test_determinism_byte_identical():
    a = run_line([1.2, 0.8, 1.5], [2, 3], sim_time=5000.0, warmup=500.0, seed=42)
    b = run_line([1.2, 0.8, 1.5], [2, 3], sim_time=5000.0, warmup=500.0, seed=42)
    assert a == b
    # different seed -> (almost surely) different realization
    c = run_line([1.2, 0.8, 1.5], [2, 3], sim_time=5000.0, warmup=500.0, seed=43)
    assert c["departures_post_warmup"] != a["departures_post_warmup"]
