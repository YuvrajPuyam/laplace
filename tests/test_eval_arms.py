"""Tests for the two Max eval arms (llm_alone, agent) — spec §5.4.

These exercise the *pure* parts that don't spend the Claude Max window: the
report→discrete-candidate mapping, equal-budget seed allocation, and abstention
grading. The Max-spending execution (running the agent / calling the model) is an
injectable seam (`run_episode` / `answer_fn`); here it's stubbed. No sim runs (the
GTs are loaded from cache), no Agent SDK import.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from eval import baselines as B
from eval import metrics
from eval.candidates import DEV_DECISIONS
from eval.gt_sweep import load_or_compute

# braess_dev candidate config hashes (from eval/gt_cache/braess_dev.gt.json)
NO_SHORTCUT_HASH = "0f23519132e3"


@pytest.fixture(scope="module")
def braess():
    return DEV_DECISIONS["braess_dev"]


@pytest.fixture(scope="module")
def braess_gt():
    return load_or_compute(DEV_DECISIONS["braess_dev"], list(range(24)))   # cached


@pytest.fixture(scope="module")
def dc():
    return DEV_DECISIONS["dc_pickzone_med"]


@pytest.fixture(scope="module")
def dc_gt():
    return load_or_compute(DEV_DECISIONS["dc_pickzone_med"], list(range(24)))  # cached


# ── the discrete-decision prompt ──────────────────────────────────────────────

def test_decision_space_text_enumerates_labels(braess):
    t = B.decision_space_text(braess)
    assert "no_shortcut" in t and "mid_cross_aisle" in t
    assert "exact label" in t.lower()


# ── report → candidate-label mapping (the fragile, important part) ─────────────

def test_match_by_unique_label(braess, braess_gt):
    report = {"recommendation": "Open it — I recommend mid_cross_aisle."}
    assert B._match_candidate(report, braess, braess_gt) == "mid_cross_aisle"


def test_config_hash_beats_misleading_text_and_metric(braess, braess_gt):
    """An exact recommended config_hash is the most reliable signal and wins even
    when the prose and the stated mean would point the other way."""
    report = {"recommendation": "the layout change",
              "primary_metric": {"recommended": {"config": NO_SHORTCUT_HASH,
                                                  "mean": 124.0}}}  # 124 ~ mid_cross
    assert B._match_candidate(report, braess, braess_gt) == "no_shortcut"


def test_match_by_metric_proximity_when_no_label(braess, braess_gt):
    near_mid = {"recommendation": "the connected layout wins",
                "primary_metric": {"recommended": {"mean": 124.0}}}
    near_base = {"recommendation": "stay put",
                 "primary_metric": {"recommended": {"mean": 120.1}}}
    assert B._match_candidate(near_mid, braess, braess_gt) == "mid_cross_aisle"
    assert B._match_candidate(near_base, braess, braess_gt) == "no_shortcut"


def test_fleet_label_match(dc, dc_gt):
    report = {"recommendation": "Recommend amr_6 — smallest fleet meeting the SLA."}
    assert B._match_candidate(report, dc, dc_gt) == "amr_6"


# ── report_to_arm_decision ────────────────────────────────────────────────────

def test_report_to_arm_decision_full(braess, braess_gt):
    episode = SimpleNamespace(
        accepted=True, num_turns=18, cost_usd=0.91,
        report={"recommendation": "recommend mid_cross_aisle",
                "primary_metric": {"name": "throughput_orders_per_hr",
                                    "recommended": {"mean": 124.2, "ci90": [123.0, 125.4]}}})
    ad = B.report_to_arm_decision(braess, braess_gt, episode,
                                  total_budget=6, allowed_seeds=[0, 1, 2], arm="agent")
    assert ad.arm == "agent" and ad.picked == "mid_cross_aisle"
    assert ad.metric_estimate == 124.2 and ad.ci90 == (123.0, 125.4)
    assert ad.rollouts_used == 6 and ad.seeds_used == [0, 1, 2]
    assert "18 turns" in ad.notes and "$0.91" in ad.notes


def test_mapping_handles_json_string_primary_metric(braess, braess_gt):
    """The runner returns the agent's RAW report, where primary_metric can be a
    JSON STRING (real shape that crashed the first live run). Must not crash, and
    must still recover the pick + CI."""
    import json as _json
    pm_str = _json.dumps({"name": "throughput_orders_per_hr",
                          "recommended": {"mean": 124.16, "ci90": [123.18, 125.13]}})
    report = {"recommendation": "mid_cross_aisle", "primary_metric": pm_str}
    assert B._match_candidate(report, braess, braess_gt) == "mid_cross_aisle"
    est, ci = B._recommended_ci_and_estimate(report)
    assert est == 124.16 and ci == (123.18, 125.13)


def test_report_to_arm_decision_abstains_when_unaccepted(braess, braess_gt):
    episode = SimpleNamespace(accepted=False, report=None,
                              error="agent ended without an accepted report")
    ad = B.report_to_arm_decision(braess, braess_gt, episode,
                                  total_budget=48, allowed_seeds=list(range(24)), arm="agent")
    assert ad.picked is None and ad.ci90 is None and ad.rollouts_used == 48
    assert "abstained" in ad.notes
    # an abstention grades as a wrong decision at worst-case regret, not a crash
    g = metrics.grade(ad, braess_gt)
    assert g["correct"] is False and g["regret"] == 1.0 and g["ci_covered"] is None


def test_grade_handles_unmappable_pick(braess_gt):
    ad = B.ArmDecision(arm="agent", scenario_id="braess_dev", picked="nonexistent",
                       metric_estimate=None, ci90=None, rollouts_used=10, notes="")
    g = metrics.grade(ad, braess_gt)
    assert g["correct"] is False and g["regret"] == 1.0 and g["picked_metric"] is None


# ── the agent arm: equal budget + GT seed prefix (stubbed execution) ───────────

def test_agent_arm_equal_budget_and_prefix(braess, braess_gt):
    captured = {}

    def stub(**kw):
        captured.update(kw)
        return SimpleNamespace(
            accepted=True, num_turns=12, cost_usd=0.5,
            report={"recommendation": "mid_cross_aisle",
                    "primary_metric": {"recommended": {"mean": 124.2, "ci90": [123, 125]}}})

    ad = B.agent(braess, budget=200, gt=braess_gt, run_episode=stub)
    assert ad.arm == "agent" and ad.picked == "mid_cross_aisle"
    # equal budget: the agent gets exactly grid_search's capped spend — the
    # 24-seed GT prefix × 2 candidates = 48 (see test_eval_fairness).
    assert ad.rollouts_used == 48
    assert captured["budgets"]["rollouts"] == 48
    assert captured["seed_base"] == 0                 # → run_rollouts seeds are the GT prefix
    assert captured["scenario_id"] == "braess_dev"
    assert "mid_cross_aisle" in captured["question"]  # discrete options enumerated


def test_agent_arm_requires_gt(braess):
    with pytest.raises(ValueError):
        B.agent(braess, budget=200, gt=None, run_episode=lambda **kw: None)


# ── the llm-alone arm: scene summary, no sim, structured answer (stubbed) ──────

def test_llm_alone_with_injected_answer(braess, braess_gt):
    ans = '{"pick": "no_shortcut", "metric_estimate": 121.0, "ci90": [118.0, 124.0], "confidence": 0.6}'
    ad = B.llm_alone(braess, budget=200, gt=braess_gt, answer_fn=lambda prompt: ans)
    assert ad.arm == "llm_alone" and ad.picked == "no_shortcut"
    assert ad.metric_estimate == 121.0 and ad.ci90 == (118.0, 124.0)
    assert ad.rollouts_used == 0                       # H1: spends no rollouts
    assert "confidence=0.6" in ad.notes


def test_llm_alone_prompt_has_scene_and_no_sim(braess, braess_gt):
    seen = {}

    def cap(prompt):
        seen["p"] = prompt
        return '{"pick": "no_shortcut", "metric_estimate": 120, "ci90": [118, 122], "confidence": 0.5}'

    B.llm_alone(braess, gt=braess_gt, answer_fn=cap)
    p = seen["p"]
    assert "Warehouse 'braess_dev'" in p               # the real get_scene_summary text
    assert "do NOT have a simulator" in p
    assert '"pick"' in p and "throughput_orders_per_hr" in p


def test_llm_alone_maps_invalid_pick_by_metric(braess, braess_gt):
    answer = {"pick": "the connected option", "metric_estimate": 124.0,
              "ci90": [122, 126], "confidence": 0.5}
    ad = B._llm_answer_to_arm_decision(braess, braess_gt, answer, arm="llm_alone")
    assert ad.picked == "mid_cross_aisle"


# ── JSON extraction robustness ────────────────────────────────────────────────

def test_parse_json_answer_from_prose():
    raw = 'Sure! Here is my call:\n{"pick": "amr_6", "confidence": 0.7}\nHope that helps.'
    d = B._parse_json_answer(raw)
    assert d["pick"] == "amr_6" and d["confidence"] == 0.7
    assert B._parse_json_answer("no json here") == {}
    assert B._parse_json_answer({"pick": "x"}) == {"pick": "x"}
