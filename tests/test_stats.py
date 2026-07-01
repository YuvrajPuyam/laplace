"""Deterministic unit tests for engine/stats.py.

Covers paired_compare (paired-t known-answer, degenerate sd==0 cases, the n<15
Wilcoxon branch, all-zero-after-drop), the relationship between the absolute
per-config CI (_abs_ci90) and the CRN-narrowed difference CI, and power_n_pairs.

All inputs are hand-constructed constants — NO RNG — so every assertion is a
known-answer check reproducible by hand or with scipy closed forms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from engine.stats import paired_compare, power_n_pairs


# A 20-sample difference vector (b - a, with a all zeros) that is roughly normal
# enough that Shapiro does NOT fire (p ~= 0.34 >= 0.01), so the method stays
# paired_t rather than dropping to Wilcoxon. n=20 >= 15 also avoids the n-gate.
_D = [-1.0, 2.0, 1.0, 3.0, 0.0, 4.0, 2.0, 1.0, 3.0, -2.0,
      2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 4.0, -1.0, 2.0, 2.0]


def test_paired_t_known_answer():
    a = [0.0] * len(_D)
    b = list(_D)
    r = paired_compare(a, b)

    assert r["method"] == "paired_t"
    assert r["n_pairs"] == 20

    # Hand-computable mean of the differences.
    assert r["diff_mean"] == pytest.approx(1.45)
    assert r["mean_a"] == pytest.approx(0.0)
    assert r["mean_b"] == pytest.approx(1.45)

    # Closed-form t CIs and p-value recomputed independently from scipy.
    d = np.asarray(_D)
    n = len(d)
    se = float(d.std(ddof=1)) / math.sqrt(n)
    lo90, hi90 = (1.45 + t * se for t in sps.t.ppf([0.05, 0.95], n - 1))
    lo95, hi95 = (1.45 + t * se for t in sps.t.ppf([0.025, 0.975], n - 1))
    assert r["ci90_diff"] == pytest.approx([lo90, hi90])
    assert r["ci95_diff"] == pytest.approx([lo95, hi95])
    assert r["p_value"] == pytest.approx(float(sps.ttest_rel(b, a).pvalue))

    # ci95 is wider than ci90 (sanity on ordering).
    assert (r["ci95_diff"][1] - r["ci95_diff"][0]) > (r["ci90_diff"][1] - r["ci90_diff"][0])

    # effect size is paired Cohen's d = mean_d / sd_d.
    assert r["effect_size_d"] == pytest.approx(1.45 / float(d.std(ddof=1)))


def test_degenerate_sd_zero_mean_zero():
    # Identical differences of 0 across all seeds: sd_d == 0, mean_d == 0.
    # n=16 (>=15) keeps method off the Wilcoxon-by-n gate; sd==0 skips Shapiro.
    a = [5.0] * 16
    b = [5.0] * 16
    r = paired_compare(a, b)

    assert r["diff_mean"] == 0.0
    assert r["p_value"] == 1.0
    assert r["ci90_diff"] == [0.0, 0.0]
    assert r["ci95_diff"] == [0.0, 0.0]
    assert r["effect_size_d"] == 0.0
    assert r["method"] == "paired_t"


def test_degenerate_sd_zero_mean_nonzero():
    # Constant non-zero difference: sd_d == 0, mean_d != 0 -> p=0, infinite effect.
    a = [0.0] * 16
    b = [3.0] * 16
    r = paired_compare(a, b)

    assert r["diff_mean"] == pytest.approx(3.0)
    assert r["p_value"] == 0.0
    assert r["ci90_diff"] == [3.0, 3.0]
    assert r["ci95_diff"] == [3.0, 3.0]
    assert math.isinf(r["effect_size_d"]) and r["effect_size_d"] > 0


def test_small_n_triggers_wilcoxon():
    # n < 15 forces the Wilcoxon signed-rank path regardless of normality.
    a = [0.0] * 10
    b = [1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0]
    r = paired_compare(a, b)

    assert r["n_pairs"] == 10
    assert r["method"] == "wilcoxon"
    # p-value should be a real probability in (0, 1].
    assert 0.0 < r["p_value"] <= 1.0


def test_wilcoxon_all_zero_after_drop_returns_sentinel_p1():
    # n<15 -> wilcoxon, but every difference is zero. scipy raises internally;
    # paired_compare swallows it and reports the documented p_value=1.0 sentinel.
    a = [1.0] * 10
    b = [1.0] * 10
    r = paired_compare(a, b)

    assert r["method"] == "wilcoxon"
    assert r["p_value"] == 1.0
    # sd_d == 0 so the degenerate branch also pins the CIs to the mean (0).
    assert r["diff_mean"] == 0.0
    assert r["ci90_diff"] == [0.0, 0.0]


def test_too_few_or_mismatched_samples_raise():
    with pytest.raises(ValueError):
        paired_compare([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])  # n < 5
    with pytest.raises(ValueError):
        paired_compare([], [])                            # empty
    with pytest.raises(ValueError):
        paired_compare([1.0] * 6, [1.0] * 5)              # length mismatch


def test_absolute_ci_wider_than_paired_diff_ci_on_correlated_data():
    # Positively-correlated arms: CRN cancels shared variance in the DIFFERENCE,
    # so ci90_diff is far tighter than either absolute per-config ci90. Built
    # deterministically (no RNG): a shared base signal + a small constant shift.
    base = [float(x) for x in range(30)]          # large across-seed spread
    a = base
    b = [x + 5.0 for x in base]                    # constant +5 shift, zero diff variance...
    # ...so give the difference a tiny bit of variance to make ci90_diff finite/non-degenerate
    b = [x + 5.0 + (0.1 if i % 2 else -0.1) for i, x in enumerate(base)]

    r = paired_compare(a, b)
    diff_width = r["ci90_diff"][1] - r["ci90_diff"][0]
    abs_a_width = r["ci90_a"][1] - r["ci90_a"][0]
    abs_b_width = r["ci90_b"][1] - r["ci90_b"][0]

    assert diff_width > 0
    assert abs_a_width > diff_width
    assert abs_b_width > diff_width


def test_power_n_pairs_floor_and_guards():
    # Normal-sized effect: returns a finite, sensible pair count > floor.
    n = power_n_pairs(2.0, 10.0)
    assert isinstance(n, int)
    assert n >= 5
    # Recompute the closed form to pin the exact value.
    d = 2.0 / 10.0
    za = sps.norm.ppf(1 - 0.05 / 2)
    zb = sps.norm.ppf(0.8)
    expected = max(math.ceil(((za + zb) / d) ** 2) + 1, 5)
    assert n == expected

    # Floor of 5: a sd<=0 guard short-circuits to 5 (cannot estimate effect).
    assert power_n_pairs(5.0, 0.0) == 5
    assert power_n_pairs(5.0, -1.0) == 5

    # Zero-effect guard: returns the huge 1e9 sentinel (cannot power a null effect).
    assert power_n_pairs(0.0, 5.0) == 10**9

    # A vanishingly small effect with real variance demands an enormous n
    # (>> floor) — confirms the floor does not mask tiny effects.
    assert power_n_pairs(0.001, 100.0) > 5
