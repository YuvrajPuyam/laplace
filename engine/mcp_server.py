"""laplace-env MCP server: the eight tools over JSON-RPC 2.0 on stdio.

Dependency-free on purpose (the official SDK pulls pywin32, which Windows
Application Control blocks on some machines; a benchmark environment should
be zero-friction to connect to). Implements the MCP subset every client
needs: initialize, notifications/initialized, ping, tools/list, tools/call.

Two modes (tools.md: "the CLI and the MCP server are thin clients over the
same endpoints"):

- **Proxy mode** (LAPLACE_ENGINE_URL + LAPLACE_EPISODE_ID set): every
  tools/call is forwarded to the FastAPI engine, which owns the episode
  state. This makes the MCP server stateless and safe under client
  reconnects / parallel connections — some MCP clients spawn several server
  processes for one session, and episode budgets must not fragment.
- **Standalone mode** (otherwise): one server process = one episode, created
  lazily on the first tool call against LAPLACE_SCENARIO (default: the
  scenario_id passed to that first get_scene_summary call). Budgets via
  LAPLACE_BUDGET_ROLLOUTS / _RENDERS / _TOOL_CALLS; seed base via
  LAPLACE_SEED_BASE. Convenient for third parties; subject to the reconnect
  caveat above.

Run:  python -m engine.mcp_server
Register (Claude Code):  claude mcp add laplace-env -- python -m engine.mcp_server
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from .episode import Episode
from .errors import ToolError
from .store import ScenarioStore
from .tools import HANDLERS, dispatch

PROTOCOL_VERSION = "2025-06-18"

TOOL_DESCRIPTIONS = {
    "get_scene_summary": "Orient yourself: layout, stations, fleet, demand for a scenario. No performance metrics — those cost rollouts.",
    "propose_config": "Register a config variant as a dot-path patch against a scenario_id or prior config_hash. The only way configs enter the system.",
    "run_rollouts": "Spend rollout budget to simulate configs. Seeds are engine-allocated and PAIRED across configs (common random numbers).",
    "compare_configs": "Paired statistical comparison of two rolled-out configs on one metric. The only legitimate source of comparative numbers.",
    "power_check": "How many seed-pairs would resolve the observed effect at target power? Answers 'can my remaining budget settle this?'",
    "render_evidence": "Queue a visual evidence render (clip/still/side-by-side) of a rolled-out seed. Async; poll via get_budget.",
    "get_budget": "Remaining rollout/render budget, tool-call count, and render job statuses.",
    "submit_report": "Submit the final report (spec §6.4 shape). Validated; one repair attempt on rejection. Terminates the episode.",
}

# Input schemas mirror schemas/tools.md so clients send the right shapes.
# The engine's own validation stays authoritative for semantics (bounds,
# Contract A) — these only pin the wire format.
_STR = {"type": "string"}
_INT = {"type": "integer"}
_NUM = {"type": "number"}
TOOL_SCHEMAS = {
    "get_scene_summary": {
        "type": "object",
        "properties": {"scenario_id": _STR},
        "required": ["scenario_id"]},
    "propose_config": {
        "type": "object",
        "properties": {
            "base": {"type": "string",
                     "description": "scenario_id or a prior config_hash"},
            "patch": {"type": "object",
                      "description": "dot-path -> replacement value, e.g. "
                      '{"fleet.amr_count": 5}. Arrays are replaced wholesale. '
                      'Node ids look like "A3_15" (aisle 3, position 15 m); '
                      'edge ids like "A3_15->A4_15". extra_edges items: '
                      '{"from", "to", "bidirectional"?}; edge_overrides items: '
                      '{"edge", "capacity"?, "one_way"?, "max_speed_mps"?}.',
                      "additionalProperties": True},
            "label": _STR},
        "required": ["base", "patch", "label"]},
    "run_rollouts": {
        "type": "object",
        "properties": {
            "config_hashes": {"type": "array", "items": _STR,
                              "minItems": 1, "maxItems": 8},
            "n_seeds": {"type": "integer", "minimum": 1, "maximum": 100},
            "horizon_minutes": {"type": ["integer", "null"]}},
        "required": ["config_hashes", "n_seeds"]},
    "compare_configs": {
        "type": "object",
        "properties": {"hash_a": _STR, "hash_b": _STR,
                       "metric": {"type": "string",
                                  "description": "canonical Contract B.1 scalar "
                                  "metric, e.g. throughput_orders_per_hr"}},
        "required": ["hash_a", "hash_b", "metric"]},
    "power_check": {
        "type": "object",
        "properties": {"observed_effect": _NUM, "observed_sd_of_diff": _NUM,
                       "target_power": {"type": "number", "default": 0.8}},
        "required": ["observed_effect", "observed_sd_of_diff"]},
    "render_evidence": {
        "type": "object",
        "properties": {
            "kind": {"enum": ["clip", "still", "side_by_side"]},
            "config_hashes": {"type": "array", "items": _STR,
                              "minItems": 1, "maxItems": 2},
            "seed": _INT,
            "t_range_min": {"type": "array", "items": _NUM,
                            "minItems": 2, "maxItems": 2},
            "camera": {"enum": ["overview", "congestion_closeup", "follow_amr"]}},
        "required": ["kind", "config_hashes", "seed", "t_range_min", "camera"]},
    "get_budget": {"type": "object", "properties": {}},
    "submit_report": {
        "type": "object",
        "description": "The final report (spec §6.4 shape). Terminates the episode.",
        "additionalProperties": True},
}


class McpServer:
    def __init__(self, store: ScenarioStore | None = None):
        self.engine_url = os.environ.get("LAPLACE_ENGINE_URL", "").rstrip("/")
        self.episode_id = os.environ.get("LAPLACE_EPISODE_ID", "")
        self.proxy_mode = bool(self.engine_url and self.episode_id)
        self.store = None if self.proxy_mode else (store or ScenarioStore())
        self.episode: Episode | None = None

    # light tools answer in seconds; rollouts/renders can legitimately take minutes
    _FAST_TOOLS = {"get_scene_summary", "propose_config", "compare_configs",
                   "power_check", "get_budget"}

    def _proxy_call(self, tool: str, args: dict) -> dict:
        url = f"{self.engine_url}/episodes/{self.episode_id}/tools/{tool}"
        data = json.dumps(args).encode("utf-8")
        timeout = 30 if tool in self._FAST_TOOLS else 600
        body = None
        for attempt in range(2):            # one bounded retry, connection-level errors only
            req = urllib.request.Request(
                url, data=data, headers={"content-type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:     # carries the tool envelope — pass straight through
                body = json.loads(e.read().decode("utf-8"))
                break
            except urllib.error.URLError as e:
                # Retry ONCE on a pure connection failure (refused/reset) — the request never ran,
                # so it's safe even for budgeted tools. Do NOT retry a read-timeout: the call may
                # still be executing engine-side and re-issuing it could double-spend.
                if attempt == 0 and not isinstance(getattr(e, "reason", None), TimeoutError):
                    import time as _t
                    _t.sleep(0.5)
                    continue
                raise ToolError("engine_unreachable",
                                f"engine at {self.engine_url} unreachable: {e}") from e
            except TimeoutError as e:               # read-timeout: may be in flight, do not retry
                raise ToolError("engine_unreachable",
                                f"engine at {self.engine_url} timed out: {e}") from e
        if isinstance(body, dict) and "error" in body:
            err = body["error"]
            raise ToolError(err.get("code", "engine_error"),
                            err.get("message", "engine error"),
                            err.get("details") or {})
        return body

    def _episode_for(self, tool: str, params: dict) -> Episode:
        if self.episode is None:
            scenario = os.environ.get("LAPLACE_SCENARIO") or (
                params.get("scenario_id") if tool == "get_scene_summary" else None)
            if not scenario:
                raise ToolError(
                    "no_episode",
                    "no episode yet: call get_scene_summary first (its "
                    "scenario_id starts the episode) or set LAPLACE_SCENARIO")
            budgets = {}
            for key, env in (("rollouts", "LAPLACE_BUDGET_ROLLOUTS"),
                             ("renders", "LAPLACE_BUDGET_RENDERS"),
                             ("tool_calls", "LAPLACE_BUDGET_TOOL_CALLS")):
                if os.environ.get(env):
                    budgets[key] = int(os.environ[env])
            max_workers = os.environ.get("LAPLACE_MAX_WORKERS")
            self.episode = Episode(
                scenario_id=scenario,
                question=os.environ.get("LAPLACE_QUESTION", ""),
                store=self.store, budgets=budgets or None,
                seed_base=int(os.environ.get("LAPLACE_SEED_BASE", "0")),
                runs_dir=os.environ.get("LAPLACE_RUNS_DIR", "runs"),
                max_workers=int(max_workers) if max_workers else None)
        return self.episode

    # ---- JSON-RPC method handlers -----------------------------------------

    def handle(self, msg: dict) -> dict | None:
        method = msg.get("method", "")
        msg_id = msg.get("id")
        if method == "notifications/initialized" or method.startswith("notifications/"):
            return None  # notifications get no response
        try:
            result = self._call(method, msg.get("params") or {})
        except ToolError as e:
            # tool-level errors are MCP tool results with isError, not
            # protocol errors — agents are expected to read and recover
            result = {"content": [{"type": "text",
                                   "text": json.dumps(e.envelope())}],
                      "isError": True}
            if method != "tools/call":
                return {"jsonrpc": "2.0", "id": msg_id,
                        "error": {"code": -32000, "message": e.message,
                                  "data": e.envelope()["error"]}}
        except Exception as e:  # noqa: BLE001 — protocol-level failure
            return {"jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32603, "message": f"internal error: {e}"}}
        if msg_id is None:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _call(self, method: str, params: dict):
        if method == "initialize":
            return {"protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "laplace-env", "version": "0.1.0"}}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [
                {"name": name, "description": TOOL_DESCRIPTIONS[name],
                 "inputSchema": TOOL_SCHEMAS[name]}
                for name in HANDLERS]}
        if method == "tools/call":
            tool = params.get("name", "")
            args = params.get("arguments") or {}
            if tool not in HANDLERS:
                raise ToolError("unknown_tool", f"no tool '{tool}'",
                                {"tools": list(HANDLERS)})
            args = _coerce_stringified(tool, args)
            if self.proxy_mode:
                result = self._proxy_call(tool, args)
            else:
                ep = self._episode_for(tool, args)
                result = dispatch(ep, tool, args)
            return {"content": [{"type": "text", "text": json.dumps(result)}],
                    "isError": False}
        raise ToolError("unknown_method", f"unsupported method '{method}'")


def _coerce_stringified(tool: str, args: dict) -> dict:
    """Clients sometimes send object/array params as JSON strings; decode
    them when the tool schema expects a non-string type."""
    props = TOOL_SCHEMAS.get(tool, {}).get("properties", {})
    out = dict(args)
    for key, value in args.items():
        expected = props.get(key, {}).get("type")
        if isinstance(value, str) and expected not in ("string", None):
            try:
                out[key] = json.loads(value)
            except json.JSONDecodeError:
                pass  # leave as-is; handler validation reports it
    return out


def main() -> None:
    server = McpServer()
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = server.handle(msg)
        if response is not None:
            out.write(json.dumps(response) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
