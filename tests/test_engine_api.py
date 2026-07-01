"""HTTP-level tests: lifecycle wiring and the error envelope contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from engine.api import create_app
from engine.store import ScenarioStore


@pytest.fixture()
def client(tmp_path):
    app = create_app(store=ScenarioStore(dirs=("examples",)),
                     runs_dir=str(tmp_path))
    return TestClient(app)


def test_health_and_unknowns(client):
    h = client.get("/health").json()
    assert h["ok"] and "baseline_small" in h["scenarios"]

    r = client.post("/episodes", json={"scenario_id": "nope", "question": "q"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_scenario"

    r = client.post("/episodes/ep_missing/tools/get_budget", json={})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_episode"


def test_episode_tool_flow(client):
    r = client.post("/episodes", json={"scenario_id": "baseline_small",
                                       "question": "5th AMR?"})
    assert r.status_code == 200
    eid = r.json()["episode_id"]
    base = r.json()["baseline_hash"]

    def tool(name, params):
        return client.post(f"/episodes/{eid}/tools/{name}", json=params)

    out = tool("get_scene_summary", {"scenario_id": "baseline_small"}).json()
    assert out["config_hash"] == base

    r = tool("propose_config", {"base": "baseline_small",
                                "patch": {"fleet.amr_count": 99}, "label": "bad"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "validation_error"

    h5 = tool("propose_config", {"base": "baseline_small",
                                 "patch": {"fleet.amr_count": 5},
                                 "label": "5 AMRs"}).json()["config_hash"]
    run = tool("run_rollouts", {"config_hashes": [h5], "n_seeds": 5,
                                "horizon_minutes": 60}).json()
    assert len(run["results"]) == 5
    assert run["budget"]["rollouts_left"] == 195

    budget = tool("get_budget", {}).json()
    assert budget["rollouts_left"] == 195
    assert budget["tool_calls_used"] == 5

    trace = client.get(f"/episodes/{eid}/trace").json()["records"]
    assert trace[0]["event"] == "episode_created"
    assert [t.get("tool") for t in trace[1:]] == \
        ["get_scene_summary", "propose_config", "propose_config",
         "run_rollouts", "get_budget"]
