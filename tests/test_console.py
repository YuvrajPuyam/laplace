"""Tests for engine/console.py - the testable console backend (edit->patch, fast preview,
and the estimate-cannot-become-a-card firewall). Pure client over the frozen contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.console import (CONSOLE_EXPOSED_PATHS, ConsoleSession, EditError,
                            build_patch, editable_objects, fast_preview, render_card)
from engine.summary import EDITABLE_BOUNDS, EDITABLE_ROOTS
from sim.config import apply_patch, config_hash, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(str(ROOT / "eval" / "dev_scenarios" / "braess_dev.config.json"))


def test_exposed_paths_are_a_subset_and_editable():
    # The console can only ever NARROW editability, never widen it.
    for p in CONSOLE_EXPOSED_PATHS:
        assert p.split(".")[0] in EDITABLE_ROOTS
        assert p in EDITABLE_BOUNDS, f"{p} exposed but not in EDITABLE_BOUNDS"


def test_editable_objects_cover_levers_and_stations():
    objs = {o["object_id"]: o for o in editable_objects(CFG)}
    assert "fleet" in objs and "policy:dispatch" in objs and "extra_edges" in objs
    assert any(oid.startswith("station:pick:") for oid in objs)
    # every affordance path is a console-exposed path
    for o in objs.values():
        for a in o["affordances"]:
            assert a["path"] in CONSOLE_EXPOSED_PATHS


def test_build_patch_scalar_and_roundtrip():
    p = build_patch(CFG, "fleet", "amr_count_delta", 1)
    assert p == {"fleet.amr_count": CFG["fleet"]["amr_count"] + 1}
    validate_config(apply_patch(CFG, p))             # valid Contract A patch


def test_build_patch_dispatch_and_policies():
    assert build_patch(CFG, "policy:dispatch", "set_dispatch", "atc") == {"fleet.dispatch": "atc"}
    assert build_patch(CFG, "policy:charge", "set_charge_threshold", 0.25) == {"fleet.charge_threshold_pct": 0.25}
    validate_config(apply_patch(CFG, {"fleet.dispatch": "atc"}))


def test_build_patch_station_replaces_whole_array():
    pick = CFG["stations"]["pick"]
    sid = pick[0]["id"]
    p = build_patch(CFG, f"station:pick:{sid}", "slots_delta", 1)
    assert set(p) == {"stations.pick"}
    arr = p["stations.pick"]
    assert len(arr) == len(pick)                     # whole array, not a path into it
    assert arr[0]["slots"] == pick[0]["slots"] + 1
    validate_config(apply_patch(CFG, p))


def test_build_patch_extra_edge_toggle():
    edge = {"from": "A3_8", "to": "A4_8", "bidirectional": True}
    on = build_patch(CFG, "extra_edges", "toggle_extra_edge", edge)
    assert edge in on["layout.extra_edges"]
    validate_config(apply_patch(CFG, on))
    cfg_on = apply_patch(CFG, on)
    off = build_patch(cfg_on, "extra_edges", "toggle_extra_edge", edge)
    assert edge not in off["layout.extra_edges"]     # toggles back off


@pytest.mark.parametrize("oid,op,val", [
    ("fleet", "amr_count_delta", 999),               # exceeds max
    ("policy:dispatch", "set_dispatch", "teleport"),  # bad enum
    ("policy:congestion", "set_congestion_penalty", 9.0),  # out of range
    ("policy:charge", "set_charge_threshold", 0.99),  # out of range
    ("fleet", "bogus_op", 1),                         # unknown op
])
def test_build_patch_rejects_out_of_bounds(oid, op, val):
    with pytest.raises(EditError):
        build_patch(CFG, oid, op, val)


def test_fast_preview_shape_and_crn():
    patch = build_patch(CFG, "fleet", "amr_count_delta", 1)
    out = fast_preview(CFG, patch, n_seeds=6)
    assert out["fidelity"] == "quick_estimate"       # never claims validated
    assert out["seeds_used"] == list(range(6))
    assert out["base_hash"] != out["patched_hash"]
    assert set(out["deltas"]) == {
        "throughput_orders_per_hr", "p50_order_latency_min",
        "p95_order_latency_min", "amr_utilization_pct"}
    for d in out["deltas"].values():
        assert "diff_mean" in d and "ci90" in d and d["direction"] in ("up", "down")


def test_firewall_estimate_cannot_become_a_card():
    estimate = fast_preview(CFG, {"fleet.amr_count": CFG["fleet"]["amr_count"]}, n_seeds=6)
    with pytest.raises(EditError):
        render_card(estimate)                        # a quick_estimate is not a report


def test_firewall_accepts_validated_report():
    report = {
        "recommendation": "Add one pack station.",
        "primary_metric": {"name": "throughput_orders_per_hr",
                           "baseline": {"mean": 41.2, "ci90": [40.1, 42.3]},
                           "recommended": {"mean": 48.6, "ci90": [47.0, 50.2]}},
        "mechanism": "Pack was the bottleneck.", "confidence": 0.9,
    }
    card = render_card(report)
    assert card["primary_metric"]["recommended"]["mean"] == 48.6
    assert card["recommendation"].startswith("Add")


def test_console_session_flow():
    sess = ConsoleSession(CFG)
    pv = sess.preview_edit("fleet", "amr_count_delta", 1, n_seeds=6)
    assert pv["edit_id"] == "edit_000" and pv["fidelity"] == "quick_estimate"
    h0 = config_hash(sess.working_cfg)
    res = sess.apply_edit("edit_000")
    assert res["config_hash"] != h0                  # working config advanced
    with pytest.raises(EditError):
        sess.card_from_report(pv)                    # estimate -> card blocked
