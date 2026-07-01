# schemas/, frozen contracts

These files are **Contract A** (`config.schema.json`), **Contract B**
(`results.schema.json` + `events.schema.md`), and the **tool API**
(`tools.md`). Every workstream builds against them; they are what allows six
workstreams to proceed in parallel with mocks.

**Freeze rule:** nothing in this directory is modified silently, not by
Claude Code, not in a refactor, not "just renaming for clarity." A change
requires: (1) a PR whose description states the change, the reason, and every
consumer affected; (2) a `schema_version` bump if any instance produced under
the old schema would be invalid or reinterpreted under the new one; (3) Yuv's
explicit approval.

If a schema feels wrong while building, the correct move is to stop and flag
it, early friction here is cheap; silent divergence is how integration weeks
die.

Validation requirements for the codebase:
- Every config entering the engine validates against `config.schema.json`
  (defaults filled, then hashed).
- Every rollout result validates against `results.schema.json` before being
  returned by any tool.
- WS1 ships the replay-sufficiency unit test defined in `events.schema.md`.
- `examples/` instances are validated in CI; if an example fails, the build
  fails.
