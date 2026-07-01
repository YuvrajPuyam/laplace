"""Contract A handling: validation, default-filling, canonical hashing, patches.

Canonical form = defaults filled per config.schema.json, keys sorted, compact
separators. config_hash = first 12 hex chars of sha256 over that JSON — the
identity used everywhere (Contract B.1).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"

with open(SCHEMA_DIR / "config.schema.json", encoding="utf-8") as f:
    CONFIG_SCHEMA = json.load(f)
with open(SCHEMA_DIR / "results.schema.json", encoding="utf-8") as f:
    RESULTS_SCHEMA = json.load(f)

_CONFIG_VALIDATOR = jsonschema.Draft202012Validator(CONFIG_SCHEMA)
_RESULTS_VALIDATOR = jsonschema.Draft202012Validator(RESULTS_SCHEMA)


class ConfigError(ValueError):
    """Config failed JSON-Schema or semantic validation."""

    def __init__(self, message: str, violations: list[str] | None = None):
        super().__init__(message)
        self.violations = violations or [message]


def validate_config(config: dict) -> None:
    """JSON-Schema validation against Contract A. Raises ConfigError."""
    errors = sorted(_CONFIG_VALIDATOR.iter_errors(config), key=lambda e: list(e.absolute_path))
    if errors:
        violations = [
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors
        ]
        raise ConfigError(f"config failed schema validation ({len(violations)} violations)", violations)


def validate_result(result: dict) -> None:
    """Validate a rollout result against Contract B.1. Raises ConfigError."""
    errors = sorted(_RESULTS_VALIDATOR.iter_errors(result), key=lambda e: list(e.absolute_path))
    if errors:
        violations = [
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors
        ]
        raise ConfigError(f"result failed schema validation ({len(violations)} violations)", violations)


def fill_defaults(config: dict) -> dict:
    """Return a deep copy with every schema default filled in.

    Filling defaults before hashing makes semantically identical configs hash
    identically (tools.md, propose_config).
    """
    cfg = copy.deepcopy(config)
    layout = cfg["layout"]
    layout.setdefault("extra_edges", [])
    layout.setdefault("edge_overrides", [])
    for ee in layout["extra_edges"]:
        ee.setdefault("bidirectional", True)
    for ov in layout["edge_overrides"]:
        ov.setdefault("one_way", False)

    fleet = cfg["fleet"]
    fleet.setdefault("speed_mps", 1.5)
    fleet.setdefault("battery_capacity_m", 4000)
    fleet.setdefault("charge_minutes", 20)
    fleet.setdefault("routing", "shortest_path")

    cfg["demand"].setdefault("pack_assignment", "round_robin")

    horizon = cfg.setdefault("horizon", {})
    horizon.setdefault("sim_minutes", 480)
    horizon.setdefault("warmup_minutes", 30)
    return cfg


def canonical_json(config: dict) -> str:
    return json.dumps(fill_defaults(config), sort_keys=True, separators=(",", ":"))


def config_hash(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:12]


def apply_patch(base_config: dict, patch: dict) -> dict:
    """Apply a propose_config patch: dot-paths replace values wholesale.

    Arrays are replaced, not merged (tools.md §2). Returns a new config dict;
    the caller validates it.
    """
    cfg = copy.deepcopy(base_config)
    for dotted, value in patch.items():
        parts = dotted.split(".")
        node = cfg
        for key in parts[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise ConfigError(f"patch path not found: {dotted}")
            node = node[key]
        if not isinstance(node, dict):
            raise ConfigError(f"patch path not found: {dotted}")
        node[parts[-1]] = copy.deepcopy(value)
    return cfg


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    validate_config(cfg)
    return fill_defaults(cfg)
