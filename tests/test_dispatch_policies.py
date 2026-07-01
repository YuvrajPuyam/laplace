"""Tests for the agent-controllable dispatch/routing/charging levers (lean-warehouse
keystone). Proves the three review-mandated safety properties: (1) the new optional
fields don't change existing config hashes (absent -> frozen behavior), (2) the order-
arrival stream is byte-identical across dispatch policies (CRN preserved), and (3) each
policy is deterministic. Plus: the levers are actually live (policies differ).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sim.config import config_hash, load_config
from sim.runner import run_rollout

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "examples" / "baseline_small.config.json"
LOADED = ROOT / "eval" / "dev_scenarios" / "braess_dev.config.json"


def _rows(cfg: dict, seed: int = 0):
    _, rows = run_rollout(cfg, seed, write_log=False)
    return rows


def _events(rows, event: str):
    return [r for r in rows if r[3] == event]


# --- (1) hash safety: optional, not default-filled --------------------------------
def test_absent_levers_preserve_hash_present_changes_it():
    base = load_config(str(BASELINE))          # dev/example configs omit the new fields
    h0 = config_hash(base)
    with_field = copy.deepcopy(base)
    with_field["fleet"]["dispatch"] = "nearest_idle"
    # Present field -> different canonical form -> different hash. Proves the field is
    # NOT silently default-filled (if it were, h0 would already include it).
    assert config_hash(with_field) != h0
    # ...and the absent form still hashes to the original value.
    assert config_hash(load_config(str(BASELINE))) == h0


def test_absent_dispatch_behaves_as_nearest_idle():
    base = load_config(str(LOADED))
    explicit = copy.deepcopy(base)
    explicit["fleet"]["dispatch"] = "nearest_idle"
    assert _rows(base) == _rows(explicit), "absent dispatch must equal explicit nearest_idle"


# --- (2) CRN: arrival stream independent of policy --------------------------------
@pytest.mark.parametrize("policy", ["atc", "covert"])
def test_arrival_stream_policy_independent(policy):
    base = load_config(str(LOADED))
    other = copy.deepcopy(base)
    other["fleet"]["dispatch"] = policy
    a = _events(_rows(base), "order_arrived")
    b = _events(_rows(other), "order_arrived")
    assert a == b, f"order_arrived stream changed under dispatch={policy} (CRN broken)"


# --- (3) determinism per policy --------------------------------------------------
@pytest.mark.parametrize("policy", ["nearest_idle", "atc", "covert"])
def test_policy_deterministic(policy):
    cfg = load_config(str(LOADED))
    cfg["fleet"]["dispatch"] = policy
    assert _rows(cfg, seed=3) == _rows(cfg, seed=3)


# --- the levers are actually live ------------------------------------------------
def test_policies_can_differ():
    base = load_config(str(LOADED))
    atc = copy.deepcopy(base); atc["fleet"]["dispatch"] = "atc"
    near = _events(_rows(base), "task_assigned")
    other = _events(_rows(atc), "task_assigned")
    # On a loaded multi-AMR scenario ATC's travel+aging weighting reorders assignments.
    assert near != other, "atc produced identical assignments to nearest_idle — lever inert"


def test_routing_penalty_and_charge_threshold_run():
    cfg = load_config(str(LOADED))
    cfg["fleet"]["routing"] = "congestion_aware"
    cfg["fleet"]["congestion_penalty"] = 4.0
    cfg["fleet"]["charge_threshold_pct"] = 0.25
    result, _ = run_rollout(cfg, 0, write_log=False)
    assert result["metrics"]["throughput_orders_per_hr"] >= 0.0
    # determinism with the knobs set
    assert _rows(cfg, 1) == _rows(cfg, 1)
