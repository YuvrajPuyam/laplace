"""Scenario store: scenario_id -> baseline config.

Scenarios are *.config.json files (Contract A instances) found in the search
directories, keyed by their scenario_id field. v1 search path: examples/ and
eval/dev_scenarios/. eval/scenarios/ (held out) is served in eval runs by
passing it explicitly — it is never on the default path, so day-to-day agent
work cannot touch it by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

from sim.config import fill_defaults, validate_config

DEFAULT_DIRS = ("examples", "eval/dev_scenarios")


class ScenarioStore:
    def __init__(self, dirs: tuple[str, ...] | list[str] = DEFAULT_DIRS,
                 root: str | Path = "."):
        self.root = Path(root)
        self.dirs = list(dirs)
        self._cache: dict[str, dict] = {}
        self._scan()

    def _scan(self) -> None:
        for d in self.dirs:
            base = self.root / d
            if not base.is_dir():
                continue
            for path in sorted(base.glob("*.config.json")):
                with open(path, encoding="utf-8") as f:
                    cfg = json.load(f)
                validate_config(cfg)
                self._cache[cfg["scenario_id"]] = fill_defaults(cfg)

    def get(self, scenario_id: str) -> dict | None:
        cfg = self._cache.get(scenario_id)
        return json.loads(json.dumps(cfg)) if cfg else None

    def ids(self) -> list[str]:
        return sorted(self._cache)
