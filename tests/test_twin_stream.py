"""Tests for the staged-streaming decision twin (engine/stages.py + /twin/ask).

Covers the pure trace->stages mapping (on a synthetic trace and, if present, the
real recorded Braess run) and the SSE replay endpoint. No Max spend.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from engine.api import create_app
from engine.stages import _fix_text, trace_to_stages


def test_fix_text_repairs_mojibake_and_noops_on_clean():
    # em-dash U+2014 (utf-8 E2 80 94) mis-decoded as cp1252 -> "â€”"
    assert _fix_text("No â€” open it") == "No — open it"
    assert _fix_text("p95 8.5 min, no marker") == "p95 8.5 min, no marker"   # no-op
    assert _fix_text("clean — em-dash") == "clean — em-dash"        # already clean
    assert _fix_text(None) is None and _fix_text(42) == 42

SYNTH = [
    {"event": "episode_created", "question": "Open the shortcut?"},
    {"tool": "get_scene_summary", "params": {"scenario_id": "x"},
     "result_summary": {"config_hash": "h"}},
    {"tool": "get_budget", "params": {}, "result_summary": {}},
    {"tool": "propose_config", "params": {"label": "shortcut"},
     "result_summary": {"config_hash": "abc", "diff_summary": "added edge"}},
    {"tool": "run_rollouts", "params": {"config_hashes": ["h", "abc"]},
     "result_summary": {"seeds_used": [0, 1, 2, 3]}},
    {"tool": "compare_configs", "params": {"metric": "throughput_orders_per_hr"},
     "result_summary": {"diff_mean": -12.8, "p_value": 0.002}},
    {"tool": "power_check", "params": {}, "result_summary": {"n_pairs_required": 5}},
    {"tool": "submit_report",
     "params": {"recommendation": "No", "confidence": 0.93,
                "primary_metric": {"name": "throughput_orders_per_hr"}},
     "result_summary": {"accepted": True}},
]


# ── pure stage mapping ────────────────────────────────────────────────────────

def test_stages_sequence_and_filtering():
    stages = trace_to_stages(SYNTH)
    kinds = [(s["stage"], s["kind"]) for s in stages]
    assert kinds[0] == ("plan", "question")
    assert stages[0]["detail"] == "Open the shortcut?"
    assert ("plan", "orient") in kinds
    assert ("experiment", "propose") in kinds
    assert ("experiment", "rollouts") in kinds
    assert ("experiment", "comparison") in kinds
    assert ("preliminary", "signal") in kinds        # synthesized at 1st significance
    assert ("refined", "power") in kinds
    assert kinds[-1] == ("report", "final")
    # get_budget is collapsed out (low-signal narration)
    assert all(s["kind"] != "budget" for s in stages)
    assert all("get_budget" not in str(s) for s in stages)


def test_stages_surface_only_sourced_numbers():
    stages = trace_to_stages(SYNTH)
    comp = next(s for s in stages if s["kind"] == "comparison")
    assert comp["detail"]["diff_mean"] == -12.8 and comp["detail"]["p_value"] == 0.002
    rep = next(s for s in stages if s["kind"] == "final")
    assert rep["detail"]["recommendation"] == "No"
    assert rep["detail"]["confidence"] == 0.93 and rep["detail"]["accepted"] is True


def test_recommended_config_resolved_from_structured_field():
    # The apply-patch must come from primary_metric.recommended.config even when the
    # recommendation PROSE never spells out the 12-hex hash — this is the "I asked to add a
    # lane and nothing changed" fix: the viewer applies the structurally-recommended config.
    patch = {"layout.extra_edges": [{"from": "A3_15", "to": "A4_15"}]}
    trace = [
        {"event": "episode_created", "question": "Add a lane?"},
        {"tool": "propose_config", "params": {"label": "lane", "patch": patch},
         "result_summary": {"config_hash": "deadbeef0001", "diff_summary": "added edge"}},
        {"tool": "submit_report",
         "params": {"recommendation": "Yes — add a cross-aisle between A3 and A4.",
                    "confidence": 0.95,
                    "primary_metric": {"name": "throughput_orders_per_hr",
                                       "recommended": {"config": "deadbeef0001"}}},
         "result_summary": {"accepted": True}},
    ]
    rep = next(s for s in trace_to_stages(trace) if s["stage"] == "report")
    assert rep["kind"] == "final"
    assert rep["detail"]["config_patch"] == patch


def test_baseline_recommendation_applies_no_change():
    # Recommending the baseline (a hash that is NOT a proposed change-candidate) must apply
    # nothing — "keep current" is a real answer, not a dropped change.
    trace = [
        {"event": "episode_created", "question": "Add a lane?"},
        {"tool": "propose_config", "params": {"label": "lane", "patch": {"fleet.amr_count": 12}},
         "result_summary": {"config_hash": "cand00000001"}},
        {"tool": "submit_report",
         "params": {"recommendation": "Keep the current layout.", "confidence": 0.8,
                    "primary_metric": {"name": "throughput_orders_per_hr",
                                       "recommended": {"config": "base00000000"}}},
         "result_summary": {"accepted": True}},
    ]
    rep = next(s for s in trace_to_stages(trace) if s["stage"] == "report")
    assert rep["detail"]["config_patch"] is None


def test_preliminary_only_after_first_significant():
    # a non-significant comparison must NOT trigger the preliminary marker
    trace = [{"event": "episode_created", "question": "q"},
             {"tool": "compare_configs", "params": {"metric": "m"},
              "result_summary": {"diff_mean": 0.1, "p_value": 0.4}}]
    stages = trace_to_stages(trace)
    assert all(s["stage"] != "preliminary" for s in stages)


def test_stages_on_real_braess_trace_if_present():
    real = Path("runs/ep_2d5abed55b/trace.jsonl")
    if not real.exists():
        return
    records = [json.loads(line) for line in real.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    stages = trace_to_stages(records)
    assert {"plan", "experiment", "report"} <= {s["stage"] for s in stages}
    diffs = [s["detail"]["diff_mean"] for s in stages if s["kind"] == "comparison"]
    assert any(d is not None and d < -10 for d in diffs)   # the Braess throughput drop


# ── SSE replay endpoint ───────────────────────────────────────────────────────

def _sse_stages(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith("data: ")]


def _client_with_trace(tmp_path: Path) -> TestClient:
    (tmp_path / "ep_test").mkdir()
    with open(tmp_path / "ep_test" / "trace.jsonl", "w", encoding="utf-8") as f:
        for rec in SYNTH:
            f.write(json.dumps(rec) + "\n")
    return TestClient(create_app(runs_dir=str(tmp_path)))


def test_sse_replay_streams_stages(tmp_path):
    client = _client_with_trace(tmp_path)
    r = client.get("/twin/ask?replay=ep_test&delay=0")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    streamed = _sse_stages(r.text)
    assert streamed[-1]["stage"] == "done"
    assert streamed[:-1] == trace_to_stages(SYNTH)       # identical to the pure mapping


def test_twin_meta_returns_baseline_without_sim(tmp_path):
    # /twin/meta powers the pre-flight wizard: baseline params + layout, no sim run.
    client = TestClient(create_app(runs_dir=str(tmp_path)))
    ids = client.get("/health").json()["scenarios"]
    assert ids, "store should expose at least one dev scenario"
    m = client.get(f"/twin/meta?scenario={ids[0]}").json()
    assert m["scenario"] == ids[0]
    assert isinstance(m["fleet"], int) and m["fleet"] >= 1
    assert isinstance(m["demand"], (int, float)) and m["demand"] > 0
    assert m["aisles"] >= 1 and isinstance(m["cross_aisles"], list)
    # stations summary powers the guided pre-flight setup (per-type id/node/slots)
    assert set(m["stations"]) == {"pick", "pack", "charge", "dock"}
    assert all("id" in s and "node" in s for s in m["stations"]["pick"])
    assert client.get("/twin/meta?scenario=nope").status_code == 404


def test_twin_command_endpoint(tmp_path):
    # the instant-edit grammar endpoint: recognised -> patch; question -> not recognised
    # (viewer falls back to the agent); bad scenario/empty are clean errors.
    client = TestClient(create_app(runs_dir=str(tmp_path)))
    ids = client.get("/health").json()["scenarios"]
    sid = "braess_dev" if "braess_dev" in ids else ids[0]
    r = client.post("/twin/command", json={"scenario_id": sid, "command": "use 7 robots"}).json()
    assert r["recognized"] is True and r["kind"] == "fleet"
    assert r["patch"] == {"fleet.amr_count": 7}
    q = client.post("/twin/command", json={"scenario_id": sid, "command": "why is throughput low?"}).json()
    assert q["recognized"] is False
    assert client.post("/twin/command", json={"scenario_id": "nope", "command": "use 7 robots"}).status_code == 404
    assert client.post("/twin/command", json={"scenario_id": sid, "command": ""}).status_code == 400


def test_twin_episodes_lists_recorded(tmp_path):
    client = _client_with_trace(tmp_path)
    eps = client.get("/twin/episodes").json()["episodes"]
    assert any(e["episode_id"] == "ep_test" and e["question"] == "Open the shortcut?"
               for e in eps)


def test_sse_errors(tmp_path):
    client = _client_with_trace(tmp_path)
    assert client.get("/twin/ask?replay=ep_missing").status_code == 404
    assert client.get("/twin/ask?replay=bad/id").status_code == 404      # traversal guard
    assert client.get("/twin/ask").status_code == 400                    # no q, no replay
    assert client.get("/twin/ask?q=hi&scenario=nope").status_code == 404  # bad scenario
