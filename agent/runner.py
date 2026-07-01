"""Agent loop runner.

ClaudeAgentRunner drives the Claude Agent SDK (which runs on the local
Claude Code runtime and its login — Max subscription OAuth or API key,
whichever is configured; no key handling here).

Episode state lives in the FastAPI engine, which the runner self-hosts as a
subprocess for the duration of the episode; the MCP server instances the
Claude client spawns are stateless proxies to it (LAPLACE_ENGINE_URL +
LAPLACE_EPISODE_ID). This survives MCP client reconnects — observed in the
wild: Claude Code running several server processes for one session, which
fragmented per-process episode state.

The runner is deliberately swappable (spec hedge): the eval harness depends
only on EpisodeResult + the trace file, so a raw-Messages-API runner can
replace this class without touching WS4.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .system_prompt import SYSTEM_PROMPT

MCP_SERVER_NAME = "laplace-env"
# One source of truth for the agent's step budget so the live engine (engine/api.py)
# and the runner can't drift apart. The eval harness (Half B) overrides this explicitly.
DEFAULT_MAX_TURNS = 50
TOOL_NAMES = ["get_scene_summary", "propose_config", "run_rollouts",
              "compare_configs", "power_check", "render_evidence",
              "get_budget", "submit_report"]
# Built-in Claude Code tools the agent must not use: the eight tools are its
# entire world (spec §4.1).
DISALLOWED_BUILTINS = ["Bash", "PowerShell", "Read", "Write", "Edit", "Glob",
                       "Grep", "WebSearch", "WebFetch", "Task", "NotebookEdit",
                       "TodoWrite", "AskUserQuestion", "EnterPlanMode",
                       "ExitPlanMode", "Monitor", "KillShell", "BashOutput"]


def _clean_env() -> dict[str, str]:
    """Environment for the spawned claude process, scrubbed of this parent
    Claude Code session's own variables. Running the agent from inside a
    Claude Code session otherwise makes the child attach to the parent
    harness (its tool set leaks in — observed: PowerShell, Monitor, etc.),
    which both breaks isolation and destabilizes MCP server lifecycles.
    Auth vars (CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_*) are kept."""
    keep_prefixes = ("CLAUDE_CODE_OAUTH", "ANTHROPIC_")
    env = {}
    for key, value in os.environ.items():
        if key.upper().startswith(("CLAUDE", "CLAUDECODE")) and \
                not key.upper().startswith(keep_prefixes):
            continue
        env[key] = value
    return env


@dataclasses.dataclass
class EpisodeResult:
    question: str
    scenario_id: str
    accepted: bool
    report: dict | None
    violations: list[str]
    num_turns: int
    duration_s: float
    cost_usd: float | None
    trace_path: str
    episode_dir: str | None  # engine-side runs/<episode_id>/ (trace + report)
    error: str | None = None


def _serialize(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


class ClaudeAgentRunner:
    def __init__(self, runs_dir: str | Path = "runs", model: str | None = None,
                 max_turns: int = DEFAULT_MAX_TURNS, max_workers: int = 4,
                 query_fn=None):
        """query_fn: injection point for tests (defaults to the real SDK)."""
        self.runs_dir = Path(runs_dir)
        self.model = model
        self.max_turns = max_turns
        self.max_workers = max_workers
        self._query_fn = query_fn

    # ---- engine lifecycle --------------------------------------------------

    def _start_engine(self) -> tuple[subprocess.Popen, str]:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "LAPLACE_RUNS_DIR": str(self.runs_dir),
               "OPENBLAS_NUM_THREADS": "1"}
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "engine.api:app",
             "--port", str(port), "--log-level", "warning"],
            cwd=str(Path(__file__).resolve().parent.parent), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=2):
                    return proc, url
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if proc.poll() is not None:
                    raise RuntimeError("engine process exited during startup")
                time.sleep(0.3)
        proc.terminate()
        raise RuntimeError("engine did not become healthy within 30s")

    def _create_episode(self, url: str, scenario_id: str, question: str,
                        budgets: dict | None, seed_base: int,
                        config: dict | None = None) -> str:
        body = {"scenario_id": scenario_id, "question": question,
                "budgets": budgets, "seed_base": seed_base,
                "max_workers": self.max_workers}
        if config is not None:                      # eval-harness inline config (held scenarios)
            body["config"] = config
        req = urllib.request.Request(
            f"{url}/episodes", data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        if "episode_id" not in out:
            raise RuntimeError(f"episode creation failed: {out}")
        return out["episode_id"]

    # ---- agent session ------------------------------------------------------

    def _options(self, engine_url: str, episode_id: str):
        from claude_agent_sdk import ClaudeAgentOptions

        env = {
            "LAPLACE_ENGINE_URL": engine_url,
            "LAPLACE_EPISODE_ID": episode_id,
        }
        return ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={MCP_SERVER_NAME: {
                "command": sys.executable,
                "args": ["-m", "engine.mcp_server"],
                "env": env,
            }},
            allowed_tools=[f"mcp__{MCP_SERVER_NAME}__{t}" for t in TOOL_NAMES],
            disallowed_tools=DISALLOWED_BUILTINS,
            max_turns=self.max_turns,
            model=self.model,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=_clean_env(),
        )

    async def run(self, question: str, scenario_id: str,
                  budgets: dict | None = None, seed_base: int = 0,
                  config: dict | None = None, on_episode_start=None) -> EpisodeResult:
        """on_episode_start(episode_id): optional callback fired the moment the
        episode is created, so a caller streaming this run's trace (the live
        /twin/ask path) can attach to the EXACT episode id instead of guessing
        it by globbing runs/ — which could attach to a concurrent episode."""
        if self._query_fn is None:
            from claude_agent_sdk import query as query_fn
        else:
            query_fn = self._query_fn

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        trace_path = self.runs_dir / f"agent_{scenario_id}_{stamp}.trace.jsonl"

        engine_proc, engine_url = self._start_engine()
        try:
            episode_id = self._create_episode(engine_url, scenario_id, question,
                                              budgets, seed_base, config)
        except Exception:
            engine_proc.terminate()
            raise
        if on_episode_start is not None:
            try:
                on_episode_start(episode_id)
            except Exception:                           # a caller's callback must never abort the run
                pass

        # the agent must know which scenario to summon; everything else it
        # must discover through the tools
        prompt = f"Scenario: {scenario_id}\n\nQuestion: {question}"
        options = self._options(engine_url, episode_id)
        t0 = time.time()
        submitted_report: dict | None = None
        accepted = False
        violations: list[str] = []
        num_turns = 0
        cost = None
        error = None
        report_tool_ids: set[str] = set()

        try:
            with open(trace_path, "w", encoding="utf-8") as trace:
                async for message in query_fn(prompt=prompt, options=options):
                    record = {"ts": time.time(),
                              "type": type(message).__name__,
                              "data": _serialize(message)}
                    trace.write(json.dumps(record) + "\n")
                    trace.flush()

                    # synthetic error messages (e.g. subscription rate limits)
                    data = record["data"]
                    if isinstance(data, dict) and data.get("error"):
                        blocks = data.get("content") or []
                        text = next((b.get("text") for b in blocks
                                     if isinstance(b, dict) and b.get("text")), "")
                        error = f"{data['error']}: {text}".strip()

                    for block in getattr(message, "content", None) or []:
                        bname = type(block).__name__
                        if bname == "ToolUseBlock" and \
                                getattr(block, "name", "").endswith("submit_report"):
                            submitted_report = _serialize(block.input)
                            report_tool_ids.add(block.id)
                        elif bname == "ToolResultBlock" and \
                                getattr(block, "tool_use_id", None) in report_tool_ids:
                            out = _tool_result_json(block)
                            if out is not None:
                                accepted = bool(out.get("accepted"))
                                violations = out.get("violations", [])

                    if type(message).__name__ == "ResultMessage":
                        num_turns = getattr(message, "num_turns", 0)
                        cost = getattr(message, "total_cost_usd", None)
                        if getattr(message, "is_error", False) and not error:
                            error = str(getattr(message, "result", "agent error"))
        except Exception as e:  # noqa: BLE001 — episode failure is a result, not a crash
            # The Agent SDK can raise a terminal whose subtype is actually
            # "success" (the run completed; the agent may simply not have landed
            # an accepted report). Don't disguise that as a transport crash —
            # only a genuine error becomes `error` here; the missing-report case
            # is handled uniformly below.
            if "error result: success" not in str(e) and error is None:
                error = f"{type(e).__name__}: {e}"
        finally:
            engine_proc.terminate()
            try:
                engine_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                engine_proc.kill()

        # A clean end with no accepted report (hit the step budget, submitted a report
        # that was rejected and not repaired, or chose to abstain) is still a real
        # outcome the caller must render — never a silent null answer. Set it whether the
        # loop ended normally OR via the success-terminal above; never overwrite a real
        # error, never fire when a report was accepted.
        if not accepted and error is None:
            error = "agent ended without an accepted report"

        episode_dir = str(self.runs_dir / episode_id)

        return EpisodeResult(
            question=question, scenario_id=scenario_id,
            accepted=accepted, report=submitted_report, violations=violations,
            num_turns=num_turns, duration_s=time.time() - t0, cost_usd=cost,
            trace_path=str(trace_path), episode_dir=episode_dir, error=error)


def _tool_result_json(block) -> dict | None:
    """MCP tool results arrive as content lists of text blocks holding JSON."""
    content = getattr(block, "content", None)
    if isinstance(content, str):
        texts = [content]
    elif isinstance(content, list):
        texts = [c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
                 for c in content]
    else:
        return None
    for text in texts:
        try:
            out = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(out, dict) and "accepted" in out:
            return out
    return None
