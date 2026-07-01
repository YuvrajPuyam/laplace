"""GT-cache hash regression guard (review fix, 2026-06-28).

The ground-truth cache (eval/gt_cache/*.gt.json) is keyed by config_hash. If any
change to the config schema, fill_defaults, apply_patch, or a scenario's base config
shifts a hash, the cache silently goes STALE and every regret / decision-accuracy
number drawn from it is then computed against the wrong optimum.

This test re-derives every cached hash from the live config code and asserts a
byte-identical match. It is the actual guarantee behind the "lean warehouse" plan:
when a policy lever (dispatch / routing / charging) is later added, it MUST be
schema-optional and NOT injected into fill_defaults (the `aisle_spacing_m` pattern) —
if someone instead default-fills a new field, every warehouse hash changes and THIS
test fails loudly, before the benchmark is corrupted.

Pure hash integrity: configs are treated opaquely; no scenario optimum or trap design
is referenced, so the held-out integrity property (CLAUDE.md #4) is preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.gt_sweep import ALL_DECISIONS, _candidate_config, cache_path
from sim.config import config_hash, load_config

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "eval" / "gt_cache"

CACHED_IDS = sorted(p.name[: -len(".gt.json")] for p in GT_DIR.glob("*.gt.json"))


def test_gt_cache_dir_present():
    assert CACHED_IDS, "no eval/gt_cache/*.gt.json found — guard would be vacuous"


@pytest.mark.parametrize("sid", CACHED_IDS)
def test_gt_cache_hashes_unchanged(sid):
    cached = json.loads(cache_path(sid).read_text(encoding="utf-8"))
    decision = ALL_DECISIONS.get(sid)
    if decision is None:
        pytest.skip(f"cached scenario {sid!r} has no Decision in ALL_DECISIONS")

    base = load_config(decision.base_path())
    assert config_hash(base) == cached["base_config_hash"], (
        f"{sid}: base_config_hash changed -> GT cache STALE. A new config field must be "
        f"schema-optional and NOT injected into fill_defaults (the aisle_spacing_m pattern), "
        f"or re-run `make gt-sweeps`."
    )

    by_label = {c.label: c for c in decision.candidates}
    for crow in cached["candidates"]:
        cand = by_label.get(crow["label"])
        assert cand is not None, (
            f"{sid}: cached candidate {crow['label']!r} no longer in the Decision — "
            f"cache and candidate set diverged."
        )
        cfg = _candidate_config(base, cand.patch)
        assert config_hash(cfg) == crow["config_hash"], (
            f"{sid}/{crow['label']}: candidate config_hash changed -> GT cache STALE "
            f"(check fill_defaults / apply_patch / schema canonicalization)."
        )
