"""Statistical power: is the benchmark big enough to support its claims? Normal-approx
formulas for (a) detecting a coverage miscalibration and (b) a paired (McNemar) decision-
accuracy gap between two methods scored on the same scenarios. With a ~15-scenario suite the
honest message is usually 'underpowered for small effects' - these functions quantify that.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _z(p: float) -> float:
    return float(norm.ppf(p))


def n_for_coverage_test(margin: float, *, p0: float = 0.90, alpha: float = 0.05,
                        power: float = 0.80, two_sided: bool = True) -> int:
    """Smallest n to detect that true coverage = p0 - margin differs from nominal p0
    (one-proportion test, normal approx). PITFALL: p0=0.9 is near the boundary - for the
    headline claim prefer a simulation power calc consistent with the Wilson test used."""
    if not (0 < margin < min(p0, 1 - p0) + margin):
        raise ValueError("margin must be positive")
    p1 = min(max(p0 - margin, 1e-6), 1 - 1e-6)
    za = _z(1 - alpha / 2) if two_sided else _z(1 - alpha)
    zb = _z(power)
    n = ((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / margin) ** 2
    return math.ceil(n)


def n_for_paired_accuracy(p_b: float, p_c: float, *, alpha: float = 0.05,
                          power: float = 0.80) -> int:
    """McNemar power: smallest number of SCENARIOS (paired) to detect discordance p_b vs
    p_c, where p_b = P(agent right & grid wrong), p_c = P(agent wrong & grid right). n is
    driven by the discordance p_b+p_c, not just the marginal gap p_b-p_c. (Connor/Lachin.)"""
    if p_b == p_c:
        raise ValueError("p_b == p_c: no detectable effect (n -> infinity)")
    if not (0 <= p_b <= 1 and 0 <= p_c <= 1 and p_b + p_c <= 1):
        raise ValueError("need 0<=p_b,p_c and p_b+p_c<=1")
    disc = p_b + p_c
    diff = p_b - p_c
    za = _z(1 - alpha / 2)
    zb = _z(power)
    n = ((za * math.sqrt(disc) + zb * math.sqrt(disc - diff * diff)) / diff) ** 2
    return math.ceil(n)


def n_from_accuracy_gap(acc_a: float, acc_b: float, p_disc: float, *,
                        alpha: float = 0.05, power: float = 0.80) -> int:
    """Convenience: parameterize the paired test from a marginal accuracy gap (acc_a-acc_b)
    plus an assumed discordance p_disc. n shrinks as methods agree more often (smaller p_disc)."""
    diff = acc_a - acc_b
    p_b = (p_disc + diff) / 2
    p_c = (p_disc - diff) / 2
    return n_for_paired_accuracy(p_b, p_c, alpha=alpha, power=power)
