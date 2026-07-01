# agent/ — WS3: the Claude tool-use loop

`laplace ask "<question>" --scenario <id>` runs one episode: the agent gets
the eight laplace-env tools (via the MCP server, spawned as a subprocess per
episode) and nothing else. Engine enforces budgets/seeds/report validation
server-side; this package only carries the loop, the system prompt
([system_prompt.py](system_prompt.py), spec §4.2 contract), and trace logging.

## Auth

Runs on the **Claude Agent SDK** over the local Claude Code runtime — whatever
login Claude Code has (currently Yuv's Max subscription OAuth; no API key in
the repo). Swap path if/when eval volume outgrows Max limits: implement a
second runner against the raw Messages API behind the same `EpisodeResult`
interface; WS4 depends only on that interface + trace.jsonl.

## Isolation notes (hard-won)

- The spawned `claude` process gets a **scrubbed environment**
  (`_clean_env()`): running from inside a Claude Code session otherwise leaks
  the parent harness into the child (its tool set appears, MCP lifecycles get
  flaky). Auth vars are kept.
- `allowed_tools` = exactly the eight; common built-ins are explicitly
  disallowed. The agent must not be able to read the repo (it could find
  eval/dev scenario definitions or the sim source — that's information the
  real system wouldn't have).
- Dev work uses `eval/dev_scenarios/` ONLY. The held-out suite is never run
  through prompt-development iterations (CLAUDE.md non-negotiable #4).

## Artifacts per episode

- `runs/agent_<scenario>_<stamp>.trace.jsonl` — full client-side transcript
  (assistant text, thinking, tool calls/results, result message).
- `runs/<episode_id>/trace.jsonl` — engine-side tool-call ledger (WS5 streams
  this), plus `report.json` when a report is accepted.
