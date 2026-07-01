"""Feedback-responsiveness ablation (Gupta et al. 2025, 'LLMs are feedback-blind in
closed-loop optimization'). Confronts the threat head-on: re-run the agent with the fed-back
simulated outcomes PERMUTED and measure whether its decisions track the outcomes.

  - flip fraction T = mean(rec_perm != rec): 0 = blind, ~1-1/K = fully responsive (ceiling
    is 1-1/K, NOT 1 - always report K).
  - agreement S = mean(rec == best_fed): ~1/K under blindness, 1.0 under full responsiveness.
  - permutation-test p-value for H0 'decisions are independent of the fed outcomes' (blind),
    by shuffling the rec<->outcome pairing. Phipson-Smyth +1 correction (never reports 0).
"""

from __future__ import annotations

import numpy as np


def responsiveness_test(rec, rec_perm, *, best_fed=None, n_perm: int = 10000,
                        seed: int = 0, k_options: int | None = None,
                        B: int = 2000) -> dict:
    """rec: (N,) chosen option under TRUE feedback. rec_perm: (N,) or (N,P) chosen option
    under permuted feedback (from the ablation). best_fed: (N,) argmax option of the TRUE
    fed outcomes (for the agreement statistic + permutation test)."""
    rec = np.asarray(rec)
    rec_perm = np.asarray(rec_perm)
    n = len(rec)
    if rec_perm.ndim == 1:
        flips = (rec_perm != rec).astype(float)
    else:
        flips = (rec_perm != rec[:, None]).mean(axis=1)
    T = float(flips.mean())

    rng = np.random.default_rng(seed)
    T_boot = np.array([flips[rng.integers(0, n, n)].mean() for _ in range(B)])
    T_ci = (float(np.quantile(T_boot, 0.05)), float(np.quantile(T_boot, 0.95)))

    out = {"T": T, "T_ci": T_ci, "n": n, "k_options": k_options,
           "null_flip_ceiling": (1 - 1 / k_options) if k_options else None}

    if best_fed is not None:
        best_fed = np.asarray(best_fed)
        S = float((rec == best_fed).mean())
        # permutation null: break the rec<->outcome pairing; under blindness S ~ chance.
        ge = 1
        for _ in range(n_perm):
            if (rec == rng.permutation(best_fed)).mean() >= S:
                ge += 1
        out.update({"S": S, "p_value": ge / (n_perm + 1),
                    "null_mean": (1.0 / k_options) if k_options else float("nan")})
    return out
