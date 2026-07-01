"""Episode lifecycle + budget ledger (tools.md, engine side).

One Episode = one question. It owns:
- the config registry (every config the agent proposed, by hash),
- the results cache keyed (config_hash, seed) — identical (config, seed) is
  deterministic, so re-running is never useful and never charged,
- the seed sequence (seed_base + index). The agent NEVER picks seeds: targets
  are allocated so that every config in a run_rollouts call gets the SAME
  seed list (CRN pairing, tools.md §3),
- the budget ledger (rollouts / renders / tool calls), engine-enforced,
- the render job queue (backend pluggable; WS6 supplies a real one),
- trace.jsonl persistence under runs/<episode_id>/.

Budget semantics for run_rollouts: cost = rollouts actually executed (cache
hits are free). If the remaining budget covers only part of the request, the
affordable prefix (config-major, seed-minor order) is executed and
budget_exhausted is raised with the partial results in details — matching
"partial results returned for completed work".
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from sim.config import ConfigError, apply_patch, config_hash, fill_defaults, validate_config
from sim.runner import run_many

from .errors import ToolError
from .store import ScenarioStore

DEFAULT_BUDGETS = {"rollouts": 200, "renders": 4, "tool_calls": 25}


class RenderBackend:
    """Render queue backend interface. The default keeps jobs queued forever
    (WS6 replaces this); tests inject an instantly-completing fake."""

    def submit(self, job: dict, episode: "Episode") -> None:  # noqa: ARG002
        pass


class Episode:
    def __init__(self, scenario_id: str, question: str,
                 store: ScenarioStore | None = None,
                 budgets: dict | None = None,
                 seed_base: int = 0,
                 runs_dir: str | Path = "runs",
                 render_backend: RenderBackend | None = None,
                 max_workers: int | None = None,
                 config: dict | None = None):
        self.store = store or ScenarioStore()
        # EVAL-HARNESS path: an inline `config` (e.g. a held-out scenario the public store
        # deliberately does NOT serve) is used directly, so the agent arm can be graded on
        # held scenarios without ever exposing them via the store / GET /health (CLAUDE.md #4).
        # Public callers pass no config and keep the unchanged store-lookup behaviour.
        baseline = config if config is not None else self.store.get(scenario_id)
        if baseline is None:
            raise ToolError("unknown_scenario", f"no scenario '{scenario_id}'",
                            {"known": self.store.ids()})
        self.episode_id = f"ep_{uuid.uuid4().hex[:10]}"
        self.scenario_id = scenario_id
        self.question = question
        self.seed_base = seed_base
        self.max_workers = max_workers
        self.render_backend = render_backend or RenderBackend()

        b = dict(DEFAULT_BUDGETS)
        b.update(budgets or {})
        self.budgets = b
        self.rollouts_spent = 0
        self.renders_spent = 0
        self.tool_calls_used = 0

        self.dir = Path(runs_dir) / self.episode_id
        self.log_dir = self.dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # config registry: hash -> {"config", "label", "base"}
        self.configs: dict[str, dict] = {}
        self.baseline_hash = self._register(baseline, label=f"baseline:{scenario_id}",
                                            base=scenario_id)
        # results cache: (hash, seed) -> Contract B.1 result
        self.results: dict[tuple[str, int], dict] = {}
        self.compare_calls: list[dict] = []   # call_id -> recorded comparison
        self.render_jobs: list[dict] = []
        self.report: dict | None = None
        self.report_attempts = 0
        self.closed = False

    # ---- config registry -------------------------------------------------

    def _register(self, config: dict, label: str, base: str) -> str:
        cfg = fill_defaults(config)
        h = config_hash(cfg)
        self.configs.setdefault(h, {"config": cfg, "label": label, "base": base})
        return h

    def resolve_base(self, base: str) -> dict:
        """base = scenario_id or a previously registered config_hash."""
        if base in self.configs:
            return json.loads(json.dumps(self.configs[base]["config"]))
        cfg = self.store.get(base)
        if cfg is None:
            raise ToolError("unknown_base", f"'{base}' is neither a scenario_id "
                            "nor a registered config_hash",
                            {"scenarios": self.store.ids(),
                             "config_hashes": list(self.configs)})
        return cfg

    def propose(self, base: str, patch: dict, label: str) -> tuple[str, dict]:
        base_cfg = self.resolve_base(base)
        try:
            cfg = apply_patch(base_cfg, patch)
            validate_config(cfg)
        except ConfigError as e:
            raise ToolError("validation_error", str(e),
                            {"violations": e.violations}) from e
        h = self._register(cfg, label, base)
        return h, base_cfg

    # ---- rollouts + seeds (CRN) ------------------------------------------

    def seed_for(self, index: int) -> int:
        return self.seed_base + index

    def _next_index(self, chash: str) -> int:
        i = 0
        while (chash, self.seed_for(i)) in self.results:
            i += 1
        return i

    def run_rollouts(self, hashes: list[str], n_seeds: int,
                     horizon_minutes: int | None = None) -> dict:
        for h in hashes:
            if h not in self.configs:
                raise ToolError("unknown_config", f"config '{h}' was never proposed",
                                {"known": list(self.configs)})
        if horizon_minutes is not None:
            # A horizon override is semantically a different config; register
            # the derived configs so hashes stay honest (Contract B.1).
            derived = []
            for h in hashes:
                dh, _ = self.propose(h, {"horizon.sim_minutes": horizon_minutes},
                                     label=f"{self.configs[h]['label']} @{horizon_minutes}min")
                derived.append(dh)
            hashes = derived

        start = min(self._next_index(h) for h in hashes)
        target = list(range(start, start + n_seeds))
        seeds_used = [self.seed_for(i) for i in target]
        missing = [(h, s) for h in hashes for s in seeds_used
                   if (h, s) not in self.results]

        left = self.budgets["rollouts"] - self.rollouts_spent
        affordable = missing[:left]
        for h, seeds in self._group(affordable):
            cfg = self.configs[h]["config"]
            for res in run_many(cfg, seeds, log_dir=self.log_dir,
                                max_workers=self.max_workers):
                self.results[(h, res["seed"])] = res
        self.rollouts_spent += len(affordable)

        if len(affordable) < len(missing):
            done = [self.results[k] for k in affordable]
            raise ToolError(
                "budget_exhausted",
                f"rollout budget exhausted: needed {len(missing)} rollouts, "
                f"could run {len(affordable)}",
                {"results": done, "seeds_used": seeds_used,
                 "budget": self.budget_state()["budget"]})

        results = [self.results[(h, s)] for h in hashes for s in seeds_used]
        return {"results": results, "seeds_used": seeds_used,
                "budget": {"rollouts_spent": self.rollouts_spent,
                           "rollouts_left": self.budgets["rollouts"] - self.rollouts_spent}}

    @staticmethod
    def _group(pairs: list[tuple[str, int]]) -> list[tuple[str, list[int]]]:
        by_hash: dict[str, list[int]] = {}
        for h, s in pairs:
            by_hash.setdefault(h, []).append(s)
        return list(by_hash.items())

    def paired_metric(self, hash_a: str, hash_b: str, metric: str) -> tuple[list, list, list]:
        """(values_a, values_b, common_seeds) for a scalar metric."""
        seeds = sorted(s for (h, s) in self.results if h == hash_a
                       and (hash_b, s) in self.results)
        va, vb = [], []
        for s in seeds:
            ma = self.results[(hash_a, s)]["metrics"]
            mb = self.results[(hash_b, s)]["metrics"]
            if metric not in ma or not isinstance(ma[metric], (int, float)):
                raise ToolError("unknown_metric",
                                f"'{metric}' is not a scalar Contract B.1 metric",
                                {"scalar_metrics": [k for k, v in ma.items()
                                                    if isinstance(v, (int, float))]})
            va.append(float(ma[metric]))
            vb.append(float(mb[metric]))
        return va, vb, seeds

    def abandonment_warning(self, chash: str) -> str | None:
        rs = [r for (h, _), r in self.results.items() if h == chash]
        if not rs:
            return None
        ab = sum(r["metrics"]["orders_abandoned"] for r in rs)
        arrived = sum(r["metrics"]["orders_completed"] + r["metrics"]["orders_abandoned"]
                      for r in rs)
        if arrived and ab / arrived > 0.05:
            return (f"orders_abandoned >5% of arrivals in {chash} "
                    f"({100 * ab / arrived:.1f}%) — throughput comparison may be invalid")
        return None

    # ---- budget ledger -----------------------------------------------------

    def charge_tool_call(self, tool: str) -> None:
        if tool != "submit_report" and self.tool_calls_used >= self.budgets["tool_calls"]:
            raise ToolError("budget_exhausted",
                            f"tool-call budget ({self.budgets['tool_calls']}) exhausted; "
                            "only submit_report is still allowed",
                            {"budget": "tool_calls"})
        self.tool_calls_used += 1

    def budget_state(self) -> dict:
        return {
            "budget": {
                "rollouts_left": self.budgets["rollouts"] - self.rollouts_spent,
                "renders_left": self.budgets["renders"] - self.renders_spent,
                "tool_calls_used": self.tool_calls_used,
                "tool_calls_max": self.budgets["tool_calls"],
            },
            "render_jobs": [
                {"job_id": j["job_id"], "status": j["status"], "uri": j.get("uri")}
                for j in self.render_jobs
            ],
        }

    # ---- trace -------------------------------------------------------------

    def trace(self, record: dict) -> None:
        record = {"ts": time.time(), "episode_id": self.episode_id, **record}
        with open(self.dir / "trace.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
