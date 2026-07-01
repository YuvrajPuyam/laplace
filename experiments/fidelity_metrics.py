"""experiments/fidelity_metrics.py - decision-relevant DES-vs-PhysX fidelity metrics.

The headline metrics the review demanded in place of per-leg travel-time accuracy
(experiments/physx_compare.py keeps travel-time as the mechanism layer):

  - treatment_effect_fidelity: for an intervention (baseline A -> variant B) with paired
    per-seed effects, does the effect agree in sign/size, and is the DES's RECOMMENDATION
    correct under physics? The null (no real effect) is equivalence-tested via a band
    `delta` so an INVENTED effect is also a failure.
  - decision_flip_rate: across decisions, how often (and how COSTLY) DES vs physics argmax
    disagree - plain rate with a Wilson CI plus a regret-weighted version.
  - srcc: Sim2Real Predictivity (Kadian et al., RA-L 2020) = Pearson of per-config DES vs
    physics scores; Spearman (ordering) reported as the decision-relevant companion.

CRN-correct: every bootstrap resamples the PAIRING UNIT (seed / decision / config) and
recomputes BOTH sides on it. Pure numpy; deterministic; no contract touched.
"""

from __future__ import annotations

import math

import numpy as np


def _bootstrap_ci(stat_fn, n: int, *, n_boot: int = 2000, seed: int = 12345,
                  lo: float = 5, hi: float = 95) -> tuple[float, float]:
    """Percentile CI of stat_fn(resampled_index) where the index has length n (the
    pairing unit). nan replicates (degenerate resamples) are dropped."""
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = stat_fn(rng.integers(0, n, n))
    return (float(np.nanpercentile(stats, lo)), float(np.nanpercentile(stats, hi)))


def _ranks(x: np.ndarray) -> np.ndarray:
    """Dense competition ranks (ties broken deterministically; exact for distinct x)."""
    return np.argsort(np.argsort(x)).astype(float)


def _wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (correct near 0/1; default 90%)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def treatment_effect_fidelity(d_des, d_phys, *, delta: float,
                              n_boot: int = 2000, seed: int = 12345) -> dict:
    """Per-seed effects d = metric(B) - metric(A) in each sim (paired on seeds).
    `delta` = the negligible-effect / decision band in METRIC units. Returns the two
    effects with CIs, the magnitude agreement, the decision under each model, and whether
    the DES's recommendation is correct under physics."""
    d_des = np.asarray(d_des, float)
    d_phys = np.asarray(d_phys, float)
    n = len(d_des)
    if n == 0 or len(d_phys) != n:
        raise ValueError("d_des and d_phys must be equal-length, non-empty")
    e_des, e_phys = float(d_des.mean()), float(d_phys.mean())
    des_lo, des_hi = _bootstrap_ci(lambda i: d_des[i].mean(), n, n_boot=n_boot, seed=seed)
    phys_lo, phys_hi = _bootstrap_ci(lambda i: d_phys[i].mean(), n, n_boot=n_boot, seed=seed + 1)

    def decide(lo: float, hi: float) -> int:
        if lo > delta:
            return 1                      # ship B
        if hi < -delta:
            return -1                     # keep A (B hurts)
        return 0                          # indifferent / no real effect

    dec_des, dec_phys = decide(des_lo, des_hi), decide(phys_lo, phys_hi)
    if dec_des != 0 and dec_phys != 0:
        label = "AGREE" if (e_des > 0) == (e_phys > 0) else "DISAGREE"
    else:
        label = "INCONCLUSIVE"

    abs_gap = e_des - e_phys
    abs_gap_ci = _bootstrap_ci(lambda i: d_des[i].mean() - d_phys[i].mean(), n,
                               n_boot=n_boot, seed=seed + 2)
    rel_eff_err = (e_des - e_phys) / abs(e_phys) if abs(e_phys) > delta else None
    rel_ci = None
    if rel_eff_err is not None:
        def _rel(i):
            ep = d_phys[i].mean()
            return (d_des[i].mean() - ep) / abs(ep) if abs(ep) > 1e-12 else np.nan
        rel_ci = _bootstrap_ci(_rel, n, n_boot=n_boot, seed=seed + 3)

    return {
        "e_des": e_des, "e_phys": e_phys,
        "e_des_ci": (des_lo, des_hi), "e_phys_ci": (phys_lo, phys_hi),
        "decide_des": dec_des, "decide_phys": dec_phys,
        "rec_correct": dec_des == dec_phys, "label": label,
        "abs_gap": abs_gap, "abs_gap_ci": abs_gap_ci,
        "rel_eff_err": rel_eff_err, "rel_eff_err_ci": rel_ci, "n_seeds": n,
    }


def decision_flip_rate(decisions, *, n_boot: int = 2000, seed: int = 12345,
                       z: float = 1.645) -> dict:
    """`decisions` = list of (des_means, phys_means) per decision, same config order.
    DES picks argmax des_means; physics picks argmax phys_means. Reports the plain flip
    rate (Wilson CI) and the regret-weighted version (a flip to a near-tie costs ~0)."""
    flips, regrets, norm_regrets = [], [], []
    for des, phys in decisions:
        des = np.asarray(des, float)
        phys = np.asarray(phys, float)
        bd, bp = int(np.argmax(des)), int(np.argmax(phys))
        regret = float(phys[bp] - phys[bd])           # >= 0 by construction
        spread = float(phys[bp] - phys.min())
        flips.append(bd != bp)
        regrets.append(regret)
        norm_regrets.append(regret / spread if spread > 1e-12 else 0.0)
    D = len(decisions)
    k = int(sum(flips))
    nr = np.asarray(norm_regrets, float)
    nr_ci = _bootstrap_ci(lambda i: nr[i].mean(), D, n_boot=n_boot, seed=seed)
    return {
        "n_decisions": D, "n_flips": k,
        "flip_rate": k / D if D else float("nan"),
        "flip_rate_ci": _wilson(k, D, z),
        "mean_regret": float(np.mean(regrets)) if D else float("nan"),
        "mean_norm_regret": float(nr.mean()) if D else float("nan"),
        "norm_regret_ci": nr_ci,
    }


def srcc(des_scores, phys_scores, *, n_boot: int = 2000, seed: int = 12345) -> dict:
    """Sim2Real Predictivity across configs. Pearson (Kadian's SRCC) + Spearman (the
    ordering companion, robust to a monotone DES<->physics scale mismatch)."""
    des = np.asarray(des_scores, float)
    phys = np.asarray(phys_scores, float)
    n = len(des)
    if n < 3 or len(phys) != n:
        raise ValueError("need >= 3 paired configs")
    pearson = float(np.corrcoef(des, phys)[0, 1])
    spearman = float(np.corrcoef(_ranks(des), _ranks(phys))[0, 1])

    def _pear(i):
        a, b = des[i], phys[i]
        return np.corrcoef(a, b)[0, 1] if a.std() > 1e-12 and b.std() > 1e-12 else np.nan

    def _spear(i):
        a, b = _ranks(des[i]), _ranks(phys[i])
        return np.corrcoef(a, b)[0, 1] if a.std() > 1e-12 and b.std() > 1e-12 else np.nan

    return {
        "pearson_srcc": pearson, "spearman": spearman, "n_configs": n,
        "pearson_ci": _bootstrap_ci(_pear, n, n_boot=n_boot, seed=seed),
        "spearman_ci": _bootstrap_ci(_spear, n, n_boot=n_boot, seed=seed + 1),
    }
