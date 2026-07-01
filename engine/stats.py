"""Paired statistics for compare_configs and power_check (tools.md §4, §5).

CRN pairing means every comparison is a paired design: differences are taken
per-seed, then tested. Paired t by default; Wilcoxon signed-rank when n < 15
or normality is clearly violated (Shapiro p < 0.01) — method always reported.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

ALPHA = 0.05


def paired_compare(a: list[float], b: list[float]) -> dict:
    """Compare paired samples. Returns the compare_configs stats block
    (everything except warnings/config identity, which the caller owns)."""
    if len(a) != len(b) or len(a) < 5:
        raise ValueError("need >= 5 paired samples")
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    d = xb - xa
    n = len(d)
    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1))

    method = "paired_t"
    if n < 15:
        method = "wilcoxon"
    elif sd_d > 0:
        # Shapiro on the differences; only override on clear violation.
        try:
            if sps.shapiro(d).pvalue < 0.01:
                method = "wilcoxon"
        except ValueError:
            pass

    if sd_d == 0.0:
        # Degenerate: identical differences across all seeds.
        p_value = 1.0 if mean_d == 0.0 else 0.0
        ci95 = ci90 = (mean_d, mean_d)
        effect = 0.0 if mean_d == 0.0 else math.inf * (1 if mean_d > 0 else -1)
    else:
        if method == "wilcoxon":
            try:
                p_value = float(sps.wilcoxon(d, zero_method="wilcox").pvalue)
            except ValueError:  # all differences zero after dropping
                p_value = 1.0
        else:
            p_value = float(sps.ttest_rel(xb, xa).pvalue)
        se = sd_d / math.sqrt(n)
        ci95 = tuple(mean_d + t * se for t in sps.t.ppf([0.025, 0.975], n - 1))
        ci90 = tuple(mean_d + t * se for t in sps.t.ppf([0.05, 0.95], n - 1))
        effect = mean_d / sd_d  # paired Cohen's d

    # Absolute per-config 90% CIs (full per-arm across-seed variance). An ABSOLUTE metric
    # claim (e.g. report primary_metric.recommended.ci90) MUST use these; ci90_diff is the
    # CRN-narrowed DIFFERENCE CI and is far too tight to bound a single config's metric.
    def _abs_ci90(x: np.ndarray) -> list[float]:
        m = float(x.mean()); s = float(x.std(ddof=1))
        if s == 0.0:
            return [m, m]
        se = s / math.sqrt(n)
        lo, hi = (m + tt * se for tt in sps.t.ppf([0.05, 0.95], n - 1))
        return [float(lo), float(hi)]

    return {
        "n_pairs": n,
        "mean_a": float(xa.mean()),
        "mean_b": float(xb.mean()),
        "ci90_a": _abs_ci90(xa),
        "ci90_b": _abs_ci90(xb),
        "diff_mean": mean_d,
        "ci95_diff": [float(ci95[0]), float(ci95[1])],
        "ci90_diff": [float(ci90[0]), float(ci90[1])],
        "p_value": p_value,
        "method": method,
        "effect_size_d": float(effect),
    }


def power_n_pairs(observed_effect: float, observed_sd_of_diff: float,
                  target_power: float = 0.8) -> int:
    """Pairs required for a paired t-test to detect observed_effect at
    alpha=0.05 (two-sided) with target_power. Normal approximation, +1 for
    the t-distribution penalty; floor of 5 (compare_configs minimum)."""
    if observed_sd_of_diff <= 0:
        return 5
    if observed_effect == 0:
        return 10**9  # cannot power a zero effect
    d = abs(observed_effect) / observed_sd_of_diff
    za = sps.norm.ppf(1 - ALPHA / 2)
    zb = sps.norm.ppf(target_power)
    n = math.ceil(((za + zb) / d) ** 2) + 1
    return max(n, 5)
