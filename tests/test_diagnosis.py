"""Tests for the Class B diagnosis machinery (eval/diagnosis.py).

Pure: question construction, report->cause mapping, grading, and the arms with
injected stubs (no Max, no sim). The live agent/llm execution reuses the proven
baselines path.
"""

from __future__ import annotations

from types import SimpleNamespace

from eval import diagnosis as D


def _scn():
    return D.DIAGNOSES["diag_edge"]


def test_question_has_symptom_baseline_and_candidates():
    q = D.diagnosis_question(_scn())
    assert "corridor_med" in q
    assert "111" in q and "185" in q                 # degraded + healthy throughput
    assert "edge occupancy" in q and "station-wait" in q  # the discriminating fingerprint
    for name in ("service_variance", "edge_capacity", "demand"):
        assert name in q
    assert "EXPERIMENT" in q


def test_map_cause_takes_leading_conclusion():
    d = _scn()
    # the realistic shape: name the pick first, then rule the others out
    assert D.map_cause({"recommendation": "service_variance -- not edge_capacity or "
                        "demand; raising sigma reproduces it."}, d) == "service_variance"
    assert D.map_cause({"recommendation": "The cause is edge_capacity."}, d) == "edge_capacity"
    assert D.map_cause({"recommendation": "looks like demand to me"}, d) == "demand"
    # recommendation names none -> fall back to a unique mention in the mechanism
    assert D.map_cause({"recommendation": "see below", "mechanism": "the edge_capacity "
                        "drop is the cause"}, d) == "edge_capacity"
    assert D.map_cause(None, d) is None


def test_grade():
    d = _scn()
    assert D.grade("edge_capacity", d, arm="agent")["correct"] is True
    assert D.grade("demand", d, arm="agent")["correct"] is False
    assert D.grade(None, d, arm="agent")["correct"] is False


def test_llm_alone_with_injected_answer():
    d = _scn()
    g = D.llm_alone(d, answer_fn=lambda p: '{"cause": "edge_capacity", "confidence": 0.6}')
    assert g["arm"] == "llm_alone" and g["picked"] == "edge_capacity" and g["correct"] is True
    g2 = D.llm_alone(d, answer_fn=lambda p: '{"cause": "demand", "confidence": 0.4}')
    assert g2["picked"] == "demand" and g2["correct"] is False


def test_agent_with_injected_episode():
    d = _scn()
    ep = SimpleNamespace(accepted=True, num_turns=12, cost_usd=0.7,
                         report={"recommendation": "The cause is edge_capacity — the "
                                 "corridor choke reproduces the drop."})
    g = D.agent(d, budget=60, run_episode=lambda **kw: ep)
    assert g["arm"] == "agent" and g["picked"] == "edge_capacity" and g["correct"] is True


def test_agent_abstains_on_no_report():
    d = _scn()
    ep = SimpleNamespace(accepted=False, report=None, error="ended without a report")
    g = D.agent(d, budget=60, run_episode=lambda **kw: ep)
    assert g["picked"] is None and g["correct"] is False and "abstained" in g["notes"]
