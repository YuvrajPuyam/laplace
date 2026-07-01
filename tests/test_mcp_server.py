"""MCP server tests: protocol handshake in-process, plus one real stdio
round-trip through a subprocess (what a third-party client actually does)."""

from __future__ import annotations

import json
import subprocess
import sys

from engine.mcp_server import McpServer
from engine.store import ScenarioStore


def rpc(method, params=None, id_=1):
    return {"jsonrpc": "2.0", "id": id_, "method": method,
            "params": params or {}}


def test_handshake_and_tools_list():
    s = McpServer(store=ScenarioStore(dirs=("examples",)))
    init = s.handle(rpc("initialize", {"protocolVersion": "2025-06-18"}))
    assert init["result"]["serverInfo"]["name"] == "laplace-env"
    assert s.handle({"jsonrpc": "2.0",
                     "method": "notifications/initialized"}) is None
    tools = s.handle(rpc("tools/list", id_=2))["result"]["tools"]
    assert {t["name"] for t in tools} == {
        "get_scene_summary", "propose_config", "run_rollouts",
        "compare_configs", "power_check", "render_evidence", "get_budget",
        "submit_report"}


def test_episode_lazy_start_and_tool_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("LAPLACE_RUNS_DIR", str(tmp_path))
    s = McpServer(store=ScenarioStore(dirs=("examples",)))
    # tool call before any scenario context -> isError tool result
    r = s.handle(rpc("tools/call", {"name": "get_budget", "arguments": {}}))
    assert r["result"]["isError"]
    body = json.loads(r["result"]["content"][0]["text"])
    assert body["error"]["code"] == "no_episode"

    r = s.handle(rpc("tools/call", {
        "name": "get_scene_summary",
        "arguments": {"scenario_id": "baseline_small"}}, id_=2))
    assert not r["result"]["isError"]
    out = json.loads(r["result"]["content"][0]["text"])
    assert len(out["config_hash"]) == 12

    r = s.handle(rpc("tools/call", {
        "name": "propose_config",
        "arguments": {"base": "baseline_small",
                      "patch": {"fleet.amr_count": 99}, "label": "bad"}}, id_=3))
    assert r["result"]["isError"]
    body = json.loads(r["result"]["content"][0]["text"])
    assert body["error"]["code"] == "validation_error"


def test_proxy_mode_shares_one_episode(tmp_path, monkeypatch):
    """Two McpServer instances (as when an MCP client respawns the server)
    must hit the SAME engine-side episode: one ledger, one config registry."""
    import socket
    import threading
    import time

    import uvicorn

    from engine.api import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(store=ScenarioStore(dirs=("examples",)),
                     runs_dir=str(tmp_path))
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        assert time.time() < deadline, "uvicorn did not start"
        time.sleep(0.05)
    try:
        client = TestClientShim(port)
        eid = client.post_json(f"http://127.0.0.1:{port}/episodes",
                               {"scenario_id": "baseline_small",
                                "question": "q"})["episode_id"]
        monkeypatch.setenv("LAPLACE_ENGINE_URL", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("LAPLACE_EPISODE_ID", eid)

        s1, s2 = McpServer(), McpServer()
        assert s1.proxy_mode and s2.proxy_mode
        r1 = s1.handle(rpc("tools/call", {
            "name": "get_scene_summary",
            "arguments": {"scenario_id": "baseline_small"}}, id_=1))
        assert not r1["result"]["isError"]
        # second server instance, same episode: ledger continues
        r2 = s2.handle(rpc("tools/call", {"name": "get_budget",
                                          "arguments": {}}, id_=2))
        budget = json.loads(r2["result"]["content"][0]["text"])
        assert budget["tool_calls_used"] == 2

        # stringified object params are coerced before proxying
        r3 = s2.handle(rpc("tools/call", {
            "name": "propose_config",
            "arguments": {"base": "baseline_small",
                          "patch": '{"fleet.amr_count": 5}',
                          "label": "5 AMRs"}}, id_=3))
        out = json.loads(r3["result"]["content"][0]["text"])
        assert not r3["result"]["isError"]
        assert len(out["config_hash"]) == 12
    finally:
        server.should_exit = True
        thread.join(timeout=10)


class TestClientShim:
    """Tiny urllib helper (httpx TestClient can't reach a live uvicorn)."""

    def __init__(self, port):
        self.port = port

    def post_json(self, url, body):
        import urllib.request
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))


def test_stdio_subprocess_round_trip(tmp_path):
    lines = "\n".join(json.dumps(m) for m in [
        rpc("initialize", {"protocolVersion": "2025-06-18"}, id_=1),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        rpc("tools/list", id_=2),
        rpc("tools/call", {"name": "get_scene_summary",
                           "arguments": {"scenario_id": "baseline_small"}}, id_=3),
    ]) + "\n"
    import os
    env = {**os.environ, "LAPLACE_RUNS_DIR": str(tmp_path)}
    proc = subprocess.run([sys.executable, "-m", "engine.mcp_server"],
                          input=lines, capture_output=True, text=True,
                          timeout=60, cwd=".", env=env)
    responses = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
    by_id = {r["id"]: r for r in responses}
    assert by_id[1]["result"]["serverInfo"]["name"] == "laplace-env"
    assert len(by_id[2]["result"]["tools"]) == 8
    summary = json.loads(by_id[3]["result"]["content"][0]["text"])
    assert summary["config"]["scenario_id"] == "baseline_small"
