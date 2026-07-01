"""Tests for the deterministic live-command grammar (engine/commands.py).

Pure parsing, no engine/LLM. Locks the canonical commands + the 'return None so the
caller falls back to the LLM' contract for anything unrecognised.
"""

from __future__ import annotations

from engine.commands import parse_command

CFG = {
    "fleet": {"amr_count": 9},
    "demand": {"arrival_rate_per_min": 2.2},
    "stations": {"pick": [{"id": "P1", "node": "A3_05"}, {"id": "P2", "node": "A3_13"}],
                 "pack": [{"id": "K1", "node": "A4_16"}]},
}


def test_add_lane_between_nodes():
    r = parse_command("add a cross-aisle between A3_15 and A4_15", CFG)
    assert r["kind"] == "lane"
    assert r["patch"]["layout.extra_edges"] == [
        {"from": "A3_15", "to": "A4_15", "bidirectional": True}]
    assert "A3_15" in r["summary"] and "A4_15" in r["summary"]


def test_fleet_absolute():
    for q in ("use 12 robots", "set the fleet to 12", "run with 12 AMRs"):
        r = parse_command(q, CFG)
        assert r["kind"] == "fleet" and r["patch"] == {"fleet.amr_count": 12}, q


def test_fleet_delta_uses_current():
    assert parse_command("add 3 robots", CFG)["patch"] == {"fleet.amr_count": 12}   # 9 + 3
    assert parse_command("remove 4 robots", CFG)["patch"] == {"fleet.amr_count": 5}  # 9 - 4
    assert parse_command("remove 20 robots", CFG)["patch"] == {"fleet.amr_count": 1}  # floored at 1


def test_demand_per_min_and_per_hour():
    assert parse_command("set demand to 3", CFG)["patch"] == {"demand.arrival_rate_per_min": 3.0}
    assert parse_command("4 orders per minute", CFG)["patch"] == {"demand.arrival_rate_per_min": 4.0}
    # per-hour is normalised to per-minute
    assert parse_command("180 orders per hour", CFG)["patch"] == {"demand.arrival_rate_per_min": 3.0}


def test_one_way_edge():
    r = parse_command("make the edge A3_15 to A4_15 one-way", CFG)
    assert r["kind"] == "oneway"
    assert r["patch"]["layout.edge_overrides"] == [{"edge": "A3_15->A4_15", "one_way": True}]


def test_move_station():
    r = parse_command("move P1 to A3_10", CFG)
    assert r["kind"] == "station" and r["edits"] == {"P1": "A3_10"}
    # 'move' must win over the fleet/number heuristics
    assert "patch" not in r


def test_unrecognised_returns_none_for_llm_fallback():
    for q in ("should we add a lane?", "why is throughput low?", "hello", "",
              "what happens if demand spikes"):
        assert parse_command(q, CFG) is None, q
