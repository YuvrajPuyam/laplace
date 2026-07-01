"""WS3 runner tests with a fake query function — no LLM calls, no SDK
subprocess. The real end-to-end episode is exercised manually / by WS4."""

from __future__ import annotations

import asyncio
import dataclasses
import json

from agent.runner import ClaudeAgentRunner, MCP_SERVER_NAME, TOOL_NAMES
from agent.system_prompt import SYSTEM_PROMPT


@dataclasses.dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclasses.dataclass
class ToolResultBlock:
    tool_use_id: str
    content: list


@dataclasses.dataclass
class TextBlock:
    text: str


@dataclasses.dataclass
class AssistantMessage:
    content: list


@dataclasses.dataclass
class UserMessage:
    content: list


@dataclasses.dataclass
class ResultMessage:
    num_turns: int
    total_cost_usd: float | None
    is_error: bool
    result: str


REPORT = {"question": "q", "recommendation": "do nothing", "confidence": 0.9}


def fake_query(*, prompt, options):
    async def gen():
        yield AssistantMessage(content=[TextBlock(text="planning...")])
        yield AssistantMessage(content=[
            ToolUseBlock(id="tu_1", name=f"mcp__{MCP_SERVER_NAME}__submit_report",
                         input=REPORT)])
        yield UserMessage(content=[
            ToolResultBlock(tool_use_id="tu_1", content=[
                {"type": "text",
                 "text": json.dumps({"accepted": True, "violations": []})}])])
        yield ResultMessage(num_turns=3, total_cost_usd=None, is_error=False,
                            result="done")
    return gen()


def test_system_prompt_carries_the_contract():
    for required in ("compare_configs", "decision space", "power_check",
                     "submit_report", "caveats"):
        assert required in SYSTEM_PROMPT


def test_options_wire_mcp_proxy_and_tool_allowlist(tmp_path):
    runner = ClaudeAgentRunner(runs_dir=tmp_path)
    opts = runner._options("http://127.0.0.1:9999", "ep_abc123")
    server = opts.mcp_servers[MCP_SERVER_NAME]
    assert server["args"] == ["-m", "engine.mcp_server"]
    env = server["env"]
    assert env["LAPLACE_ENGINE_URL"] == "http://127.0.0.1:9999"
    assert env["LAPLACE_EPISODE_ID"] == "ep_abc123"
    assert set(opts.allowed_tools) == \
        {f"mcp__{MCP_SERVER_NAME}__{t}" for t in TOOL_NAMES}
    assert "Bash" in opts.disallowed_tools
    assert "PowerShell" in opts.disallowed_tools
    # parent Claude Code session vars must not leak into the child
    assert not any(k.upper().startswith("CLAUDECODE") for k in opts.env)


class _FakeProc:
    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def _stub_engine(runner):
    runner._start_engine = lambda: (_FakeProc(), "http://127.0.0.1:9999")
    runner._create_episode = lambda *a, **k: "ep_fake01"


def test_run_parses_report_and_writes_trace(tmp_path):
    runner = ClaudeAgentRunner(runs_dir=tmp_path, query_fn=fake_query)
    _stub_engine(runner)
    result = asyncio.run(runner.run("q", "baseline_small"))
    assert result.accepted is True
    assert result.report["recommendation"] == "do nothing"
    assert result.error is None
    assert result.num_turns == 3

    lines = [json.loads(l) for l in
             open(result.trace_path, encoding="utf-8")]
    assert [l["type"] for l in lines] == \
        ["AssistantMessage", "AssistantMessage", "UserMessage", "ResultMessage"]


def test_run_surfaces_failures_as_result(tmp_path):
    def boom(*, prompt, options):
        async def gen():
            raise RuntimeError("transport died")
            yield  # noqa
        return gen()

    runner = ClaudeAgentRunner(runs_dir=tmp_path, query_fn=boom)
    _stub_engine(runner)
    result = asyncio.run(runner.run("q", "baseline_small"))
    assert result.accepted is False
    assert "transport died" in result.error


def test_sdk_success_terminal_after_report_is_not_an_error(tmp_path):
    # The Agent SDK can raise a terminal whose subtype is "success" after a
    # normally-completed run; with an accepted report that must NOT surface as
    # an error (regression for runs/first_episode.out).
    def q(*, prompt, options):
        async def gen():
            yield AssistantMessage(content=[
                ToolUseBlock(id="tu_1",
                             name=f"mcp__{MCP_SERVER_NAME}__submit_report",
                             input=REPORT)])
            yield UserMessage(content=[
                ToolResultBlock(tool_use_id="tu_1", content=[
                    {"type": "text",
                     "text": json.dumps({"accepted": True, "violations": []})}])])
            raise Exception("Claude Code returned an error result: success")
            yield  # noqa
        return gen()

    runner = ClaudeAgentRunner(runs_dir=tmp_path, query_fn=q)
    _stub_engine(runner)
    result = asyncio.run(runner.run("q", "baseline_small"))
    assert result.accepted is True
    assert result.error is None


def test_sdk_success_terminal_without_report_is_honest(tmp_path):
    def q(*, prompt, options):
        async def gen():
            yield AssistantMessage(content=[TextBlock(text="thinking")])
            raise Exception("Claude Code returned an error result: success")
            yield  # noqa
        return gen()

    runner = ClaudeAgentRunner(runs_dir=tmp_path, query_fn=q)
    _stub_engine(runner)
    result = asyncio.run(runner.run("q", "baseline_small"))
    assert result.accepted is False
    assert result.error == "agent ended without an accepted report"
