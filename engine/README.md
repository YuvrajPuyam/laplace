# engine/ — WS2: tool API, budget ledger, episode lifecycle

Implements the eight frozen tools (schemas/tools.md) once, in
[tools.py](tools.py), and exposes them over two transports:

- **HTTP** — [api.py](api.py), FastAPI. `uvicorn engine.api:app --port 8000`
- **MCP (`laplace-env`)** — [mcp_server.py](mcp_server.py), JSON-RPC 2.0 over
  stdio, dependency-free. `claude mcp add laplace-env -- python -m engine.mcp_server`
  One server process = one episode (lazy-started; see module docstring for
  `LAPLACE_*` env vars).

## Episode semantics (engine-enforced, not honor-system)

- **Seeds**: the agent never picks them. `run_rollouts` targets indices
  `[s, s+n)` of the episode seed sequence where `s` is the smallest index any
  requested config is missing — every config in a call gets the SAME seed
  list (CRN pairing), and configs that already covered those seeds are free
  cache hits. Repeating a call extends the sequence (more evidence).
- **Budgets**: rollouts debited per simulation actually executed; if the
  remaining budget covers only part of a request, the affordable prefix runs
  and `budget_exhausted` carries the partial results in `details`. Tool-call
  cap blocks everything except `submit_report`.
- **Horizon override** registers derived configs (`horizon.sim_minutes`
  replaced) so result hashes stay honest per Contract B.1.
- **submit_report** enforces grounding: numbers must trace (±0.5%) to a
  recorded `compare_configs` call or to engine-computed per-config stats;
  evidence URIs must be completed render jobs; >0.9 confidence with
  overlapping CI90s is rejected. One repair attempt, then the episode closes.
- Every tool call is appended to `runs/<episode_id>/trace.jsonl` (WS5 streams
  this); accepted reports land at `runs/<episode_id>/report.json`.

## Contract frictions flagged (Yuv to rule; nothing changed silently)

1. `compare_configs` responses include a `call_id` not present in the frozen
   return schema — tools.md §8 *requires* reports to cite "a compare_configs
   call id", so an id has to be surfaced to the agent somewhere.
2. Spec §6.4 reports per-config `ci90`, but no tool returns per-config CIs
   (only diff CIs). The report validator recomputes per-config stats from
   stored results to keep grounding checkable; surfacing per-config CIs in a
   tool response would be cleaner.
3. There is no `report.schema.json` in `schemas/` — §6.4 lives only in the
   spec. The validator implements it in Python ([report.py](report.py));
   freezing it as a schema file is pending.

## Render queue

`render_evidence` debits budget and queues jobs against a pluggable
`RenderBackend` (default keeps jobs `queued`; tests inject an instant fake).
WS6 plugs the Isaac/Three.js renderer in here without touching handlers.
