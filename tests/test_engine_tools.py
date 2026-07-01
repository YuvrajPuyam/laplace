"""WS2 handler tests: budgets, CRN pairing, structured errors, report rules.

Rollouts here use horizon_minutes=60 (the schema minimum) to stay fast; the
horizon override intentionally registers derived configs, so tests track the
derived hashes returned in results.
"""

from __future__ import annotations

import json

import pytest

from engine.episode import Episode, RenderBackend
from engine.errors import ToolError
from engine.store import ScenarioStore
from engine.tools import dispatch


class InstantRenderBackend(RenderBackend):
    def submit(self, job, episode):  # noqa: ARG002
        job["status"] = "done"
        job["uri"] = f"renders/{job['job_id']}.mp4"


@pytest.fixture()
def ep(tmp_path):
    return Episode("baseline_small", "is a 5th AMR worth it?",
                   store=ScenarioStore(dirs=("examples",)),
                   runs_dir=tmp_path, render_backend=InstantRenderBackend(),
                   max_workers=1)


def short(ep_, hashes, n):
    return dispatch(ep_, "run_rollouts",
                    {"config_hashes": hashes, "n_seeds": n, "horizon_minutes": 60})


def test_scene_summary_no_metrics(ep):
    out = dispatch(ep, "get_scene_summary", {"scenario_id": "baseline_small"})
    assert out["config_hash"] == ep.baseline_hash
    assert "editable_bounds" in out and "fleet.amr_count" in out["editable_bounds"]
    text = out["summary_text"].lower()
    for forbidden in ("throughput", "latency", "utilization"):
        assert forbidden not in text


def test_propose_validate_and_reject(ep):
    out = dispatch(ep, "propose_config",
                   {"base": "baseline_small", "patch": {"fleet.amr_count": 5},
                    "label": "5 AMRs"})
    assert len(out["config_hash"]) == 12
    assert "fleet.amr_count: 4 -> 5" in out["diff_summary"]

    with pytest.raises(ToolError) as ei:
        dispatch(ep, "propose_config",
                 {"base": "baseline_small", "patch": {"fleet.amr_count": 99},
                  "label": "too many"})
    assert ei.value.code == "validation_error"
    assert any("99" in v for v in ei.value.details["violations"])

    with pytest.raises(ToolError) as ei:
        dispatch(ep, "propose_config",
                 {"base": "baseline_small", "patch": {"scenario_id": "evil"},
                  "label": "rename"})
    assert ei.value.code == "patch_path_not_editable"

    with pytest.raises(ToolError) as ei:
        dispatch(ep, "propose_config",
                 {"base": "nope", "patch": {"fleet.amr_count": 5}, "label": "x"})
    assert ei.value.code == "unknown_base"


def test_propose_accepts_stringified_patch(ep):
    # MCP/SDK clients sometimes deliver object params as JSON strings; dispatch
    # coerces them so the agent can register a config without flailing on format
    # (regression for runs/ep_cb46b1e7d4: stringified patch -> 18 wasted calls).
    out = dispatch(ep, "propose_config",
                   {"base": "baseline_small",
                    "patch": '{"fleet.amr_count": 5}', "label": "5 AMRs (str)"})
    assert len(out["config_hash"]) == 12
    assert "fleet.amr_count: 4 -> 5" in out["diff_summary"]
    # a genuinely malformed patch string still fails handler validation
    with pytest.raises(ToolError) as ei:
        dispatch(ep, "propose_config",
                 {"base": "baseline_small", "patch": "fleet.amr_count=5",
                  "label": "bad"})
    assert ei.value.code == "invalid_params"


def test_run_rollouts_accepts_stringified_args(ep):
    a = ep.baseline_hash
    out = dispatch(ep, "run_rollouts",
                   {"config_hashes": f'["{a}"]', "n_seeds": "3",
                    "horizon_minutes": "60"})
    assert len(out["results"]) == 3
    assert out["seeds_used"] == [0, 1, 2]


def test_submit_report_coerces_stringified_fields(ep):
    # Clients stringify structured report fields too (observed: primary_metric
    # as a JSON string, confidence as "0.95"). dispatch must coerce so the type
    # checks see real objects/numbers (regression for runs/ep_13cbfacde0).
    report = {
        "question": "q", "recommendation": "no",
        "primary_metric": json.dumps({
            "name": "throughput_orders_per_hr",
            "baseline": {"mean": 1.0, "ci90": [0.0, 2.0]},
            "recommended": {"mean": 1.0, "ci90": [0.0, 2.0]}}),
        "mechanism": "because", "confidence": "0.5",
        "evidence": "[]",
        "experiments": json.dumps([{"configs": ["x"], "seeds": "0-4 paired"}]),
        "caveats": "[]",
    }
    out = dispatch(ep, "submit_report", report)
    joined = " ".join(out["violations"])
    # coercion worked: the type/shape violations are gone ...
    assert "confidence must be a number" not in joined
    assert "primary_metric must be an object" not in joined
    assert "evidence must be a list" not in joined
    # ... but the substance check still (correctly) rejects: no rollouts here
    assert out["accepted"] is False
    assert any("no rollouts" in v for v in out["violations"])


def test_run_rollouts_crn_pairing_and_cache(ep):
    a = ep.baseline_hash
    b = dispatch(ep, "propose_config",
                 {"base": "baseline_small", "patch": {"fleet.amr_count": 5},
                  "label": "5 AMRs"})["config_hash"]
    out = short(ep, [a, b], 5)
    assert len(out["results"]) == 10
    assert out["seeds_used"] == [0, 1, 2, 3, 4]
    # CRN: same seed list for both configs
    da, db = out["results"][:5], out["results"][5:]
    assert [r["seed"] for r in da] == [r["seed"] for r in db] == out["seeds_used"]
    spent_once = ep.rollouts_spent
    assert spent_once == 10

    # repeating the call extends the seed sequence (more evidence, paired)
    da_hash = da[0]["config_hash"]
    out2 = short(ep, [a], 5)
    assert out2["seeds_used"] == [5, 6, 7, 8, 9]
    assert ep.rollouts_spent == spent_once + 5

    # asymmetric histories: the config that already covered seeds 5-9 is a
    # free cache hit; only the lagging config pays
    out3 = short(ep, [a, b], 5)
    assert out3["seeds_used"] == [5, 6, 7, 8, 9]
    assert ep.rollouts_spent == spent_once + 10
    assert {r["config_hash"] for r in out3["results"]} == \
        {r["config_hash"] for r in out["results"]}
    assert da_hash in {r["config_hash"] for r in out3["results"]}


def test_budget_exhausted_partial(tmp_path):
    ep_ = Episode("baseline_small", "q", store=ScenarioStore(dirs=("examples",)),
                  budgets={"rollouts": 3}, runs_dir=tmp_path, max_workers=1)
    with pytest.raises(ToolError) as ei:
        short(ep_, [ep_.baseline_hash], 5)
    assert ei.value.code == "budget_exhausted"
    assert len(ei.value.details["results"]) == 3
    assert ep_.rollouts_spent == 3


def test_tool_call_budget(tmp_path):
    ep_ = Episode("baseline_small", "q", store=ScenarioStore(dirs=("examples",)),
                  budgets={"tool_calls": 2}, runs_dir=tmp_path, max_workers=1)
    dispatch(ep_, "get_budget", {})
    dispatch(ep_, "get_budget", {})
    with pytest.raises(ToolError) as ei:
        dispatch(ep_, "get_budget", {})
    assert ei.value.code == "budget_exhausted"
    # submit_report must still be allowed past the cap
    out = dispatch(ep_, "submit_report", {})
    assert out["accepted"] is False


def test_compare_configs_and_power(ep):
    a = ep.baseline_hash
    b = dispatch(ep, "propose_config",
                 {"base": "baseline_small", "patch": {"fleet.amr_count": 8},
                  "label": "8 AMRs"})["config_hash"]
    out = short(ep, [a, b], 6)
    ha, hb = out["results"][0]["config_hash"], out["results"][6]["config_hash"]
    cmp_ = dispatch(ep, "compare_configs",
                    {"hash_a": ha, "hash_b": hb,
                     "metric": "throughput_orders_per_hr"})
    assert cmp_["n_pairs"] == 6
    assert cmp_["method"] in ("paired_t", "wilcoxon")
    assert cmp_["ci95_diff"][0] <= cmp_["diff_mean"] <= cmp_["ci95_diff"][1]
    assert cmp_["call_id"] == "cmp_0000"

    with pytest.raises(ToolError) as ei:
        dispatch(ep, "compare_configs",
                 {"hash_a": ha, "hash_b": hb, "metric": "station_wait_p95_min"})
    assert ei.value.code == "unknown_metric"

    pw = dispatch(ep, "power_check",
                  {"observed_effect": cmp_["diff_mean"] or 1.0,
                   "observed_sd_of_diff": 5.0})
    assert pw["n_pairs_required"] >= 5

    with pytest.raises(ToolError) as ei:
        dispatch(ep, "compare_configs",
                 {"hash_a": ha, "hash_b": ep.baseline_hash,
                  "metric": "throughput_orders_per_hr"})
    assert ei.value.code == "insufficient_pairs"


def test_render_and_report_lifecycle(ep):
    a = ep.baseline_hash
    out = short(ep, [a], 5)
    ha = out["results"][0]["config_hash"]
    seed = out["seeds_used"][0]

    with pytest.raises(ToolError) as ei:
        dispatch(ep, "render_evidence",
                 {"kind": "clip", "config_hashes": [ha], "seed": 999,
                  "t_range_min": [0, 10], "camera": "overview"})
    assert ei.value.code == "log_unavailable"

    job = dispatch(ep, "render_evidence",
                   {"kind": "clip", "config_hashes": [ha], "seed": seed,
                    "t_range_min": [0, 10], "camera": "overview"})
    budget = dispatch(ep, "get_budget", {})
    jobs = {j["job_id"]: j for j in budget["render_jobs"]}
    assert jobs[job["job_id"]]["status"] == "done"
    uri = jobs[job["job_id"]]["uri"]

    # un-grounded numbers are rejected, then a repaired report is accepted
    bad = {
        "question": "q", "recommendation": "do it",
        "primary_metric": {"name": "throughput_orders_per_hr",
                           "baseline": {"mean": 999.0, "ci90": [990, 1010]},
                           "recommended": {"mean": 999.0, "ci90": [990, 1010]}},
        "mechanism": "vibes", "confidence": 0.9,
        "evidence": [{"type": "clip", "uri": uri}],
        "experiments": [{"configs": [ha], "seeds": "0-4 paired"}],
        "caveats": [],
    }
    out1 = dispatch(ep, "submit_report", bad)
    assert out1["accepted"] is False
    assert any("does not trace" in v for v in out1["violations"])

    import math

    import numpy as np
    from scipy import stats as sps
    vals = [ep.results[(ha, s)]["metrics"]["throughput_orders_per_hr"]
            for s in out["seeds_used"]]
    arr = np.asarray(vals)
    se = arr.std(ddof=1) / math.sqrt(len(arr))
    t = sps.t.ppf(0.95, len(arr) - 1)
    good = dict(bad)
    good["primary_metric"] = {
        "name": "throughput_orders_per_hr",
        "baseline": {"mean": float(arr.mean()),
                     "ci90": [float(arr.mean() - t * se), float(arr.mean() + t * se)]},
        "recommended": {"mean": float(arr.mean()),
                        "ci90": [float(arr.mean() - t * se), float(arr.mean() + t * se)]},
    }
    out2 = dispatch(ep, "submit_report", good)
    assert out2["accepted"] is True, out2["violations"]
    assert ep.closed
    assert (ep.dir / "report.json").exists()
    assert (ep.dir / "trace.jsonl").exists()

    with pytest.raises(ToolError) as ei:
        dispatch(ep, "get_budget", {})
    assert ei.value.code == "episode_closed"
