"""Calibration: CI coverage (does a stated 90% CI contain the truth 90% of the time?) and
confidence reliability / ECE (does a stated confidence match empirical accuracy?). The two
are DISTINCT objects on different agent outputs - reported separately, each with a CI.
"""

from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054          # Phi^-1(0.975), the two-sided 95% normal quantile


def ci_coverage(lo, hi, gt, *, nominal: float = 0.90, z: float = Z95) -> dict:
    """Empirical coverage of stated intervals over ground-truth values, with a Wilson CI
    (valid near the 0/1 boundary, where good calibration lives). Boundary-inclusive."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); gt = np.asarray(gt, float)
    if not (len(lo) == len(hi) == len(gt)) or len(lo) == 0:
        raise ValueError("lo, hi, gt must be equal-length, non-empty")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or not np.all(np.isfinite(gt)):
        raise ValueError("non-finite input")
    if np.any(lo > hi):
        raise ValueError("lo > hi for some interval")
    covered = (lo <= gt) & (gt <= hi)
    n = len(covered); k = int(covered.sum()); p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {"coverage": p, "gap": p - nominal, "n": n, "k": k,
            "wilson": (max(0.0, center - half), min(1.0, center + half)),
            "nominal": nominal}


def _edges(conf: np.ndarray, n_bins: int, mode: str) -> np.ndarray:
    if mode == "quantile":
        e = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        e = np.unique(e)                          # dedupe (ties) -> M may shrink
        if len(e) < 2:
            e = np.array([conf.min(), conf.max() + 1e-12])
        return e
    return np.linspace(0.0, 1.0, n_bins + 1)


def _ece_given_edges(conf: np.ndarray, y: np.ndarray, edges: np.ndarray):
    n = len(conf)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, len(edges) - 2)
    bins, ece, mce = [], 0.0, 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        cb = float(conf[m].mean()); acc = float(y[m].mean())
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "conf_bar": cb, "acc": acc, "count": cnt})
        ece += cnt / n * abs(cb - acc)
        mce = max(mce, abs(cb - acc))
    return ece, mce, bins


def _n_bins(n: int) -> int:
    return max(2, min(10, int(np.floor(np.sqrt(n)))))


def confidence_reliability(conf, y, *, n_bins: int | None = None,
                           mode: str = "quantile") -> dict:
    """Bin by stated confidence; per bin report mean confidence vs empirical accuracy.
    ECE = count-weighted mean |conf - acc|. Empty bins contribute 0 (excluded). Equal-mass
    (quantile) bins by default - robust to the confidence clustering LLMs show at small N."""
    conf = np.asarray(conf, float); y = np.asarray(y, float)
    if len(conf) != len(y) or len(conf) == 0:
        raise ValueError("conf, y must be equal-length, non-empty")
    nb = n_bins or _n_bins(len(conf))
    edges = _edges(conf, nb, mode)
    ece, mce, bins = _ece_given_edges(conf, y, edges)
    return {"ece": ece, "mce": mce, "bins": bins, "n_bins": len(edges) - 1, "n": len(conf)}


def bootstrap_ece_ci(conf, y, *, n_bins: int | None = None, mode: str = "quantile",
                     B: int = 2000, level: float = 0.95, seed: int = 0,
                     cluster_id=None) -> dict:
    """Bootstrap CI on ECE. Edges are fixed once on the full sample (so resamples are
    comparable). Resamples records, or whole CLUSTERS if cluster_id is given (correlated
    paraphrase/seed reps -> resample at the scenario level or the CI is too narrow)."""
    conf = np.asarray(conf, float); y = np.asarray(y, float)
    nb = n_bins or _n_bins(len(conf))
    edges = _edges(conf, nb, mode)
    rng = np.random.default_rng(seed)
    point = _ece_given_edges(conf, y, edges)[0]
    if cluster_id is not None:
        cluster_id = np.asarray(cluster_id)
        groups = [np.where(cluster_id == c)[0] for c in np.unique(cluster_id)]
    stats = np.empty(B)
    for b in range(B):
        if cluster_id is not None:
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[p] for p in pick])
        else:
            idx = rng.integers(0, len(conf), len(conf))
        stats[b] = _ece_given_edges(conf[idx], y[idx], edges)[0]
    lo = float(np.quantile(stats, (1 - level) / 2))
    hi = float(np.quantile(stats, 1 - (1 - level) / 2))
    return {"ece": point, "ci": (lo, hi), "B": B,
            "bootstrap_mean": float(stats.mean())}   # ~point; bootstrap does NOT debias ECE
