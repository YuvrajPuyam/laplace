"""Tests for the ablation-matrix aggregation (eval/matrix.py).

Pure aggregation + paraphrase plumbing — no Max, no sim. The LLM-arm execution is
the same proven baselines.agent/llm_alone path; here we lock down the mean±sd math
and that every dev scenario has distinct paraphrases.
"""

from __future__ import annotations

import math

from eval import matrix
from eval.candidates import DEV_DECISIONS


def test_agg_det_accuracy_and_regret():
    graded = [{"correct": True, "regret": 0.0}, {"correct": False, "regret": 1.0},
              {"correct": True, "regret": 0.0}, {"correct": True, "regret": 0.0}]
    out = matrix._agg_det(graded)
    assert out["accuracy"] == 0.75 and out["mean_regret"] == 0.25
    assert out["n_scenarios"] == 4


def test_agg_stoch_mean_sd_and_coverage():
    passes = [
        {"accuracy": 1.0, "mean_regret": 0.0,
         "graded": [{"ci_covered": True}, {"ci_covered": True}]},
        {"accuracy": 0.5, "mean_regret": 0.1,
         "graded": [{"ci_covered": True}, {"ci_covered": False}]},
    ]
    out = matrix._agg_stoch(passes)
    assert out["accuracy_mean"] == 0.75
    assert math.isclose(out["accuracy_sd"], 0.25, abs_tol=1e-9)   # pstdev of [1.0, 0.5]
    assert out["ci_coverage"] == 0.75                              # 3 of 4 covered
    assert out["n_passes"] == 2


def test_agg_stoch_single_pass_zero_sd():
    out = matrix._agg_stoch([{"accuracy": 1.0, "mean_regret": 0.0, "graded": []}])
    assert out["accuracy_mean"] == 1.0 and out["accuracy_sd"] == 0.0
    assert out["ci_coverage"] is None                              # no CI data


def test_paraphrases_cover_all_dev_scenarios_distinctly():
    for sid in DEV_DECISIONS:
        ps = matrix.PARAPHRASES.get(sid)
        assert ps and len(ps) >= 3, f"{sid} needs >=3 paraphrases"
        assert len(set(ps)) == len(ps), f"{sid} paraphrases not distinct"


def test_para_indexing_wraps():
    sid = "braess_dev"
    assert matrix._para(sid, 0) == matrix.PARAPHRASES[sid][0]
    assert matrix._para(sid, 3) == matrix.PARAPHRASES[sid][0]      # wraps modulo
