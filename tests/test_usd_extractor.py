"""Scan-to-sim extractor tests (pure layer, no Isaac).

Synthetic rack clouds stand in for what `cloud_from_usd` pulls from a real .usd,
so the extraction logic is tested without the Isaac venv. Numbers mirror the
probed `warehouse_multiple_shelves.usd` (rack rows at x ~ -10/0/+9, 40x50 m).
"""

from __future__ import annotations

import pytest

import json

from facility.usd_extractor import (
    ExtractError,
    RackCloud,
    extract_layout,
    scenario_from_cloud,
    scenario_from_layout,
)
from sim.config import canonical_json, validate_config
from sim.navgraph import NavGraph
from sim.runner import run_rollout


def make_cloud(row_xs, y_positions, *, floor_z=0.0, mpu=1.0, up="Z"):
    pitch, length = [], []
    for x in row_xs:
        for y in y_positions:
            pitch.append(float(x))
            length.append(float(y))
    return RackCloud(pitch=pitch, length=length, floor_z=floor_z,
                     meters_per_unit=mpu, up_axis=up, source="synthetic.usd")


def test_three_rows_yield_two_aisles_at_midpoints():
    # warehouse_multiple_shelves.usd: rows at x ~ -10/0/+9, length 0..50.
    cloud = make_cloud([-10, 0, 9], list(range(0, 51, 2)))
    layout = extract_layout(cloud)

    assert layout.n_rack_rows == 3
    assert layout.aisles == 2                       # aisles = rows - 1
    assert layout.aisle_length_m == 50
    assert layout.cross_aisles == [0, 50]           # single block: ends only
    assert layout.aisle_world == [-5.0, 4.5]        # row midpoints
    assert layout.real_pitch_m == pytest.approx(9.5)
    assert not layout.cropped


def test_extracted_config_is_valid_contract_a_and_runs():
    cloud = make_cloud([-10, 0, 9], list(range(0, 51, 2)))
    layout = extract_layout(cloud)
    config, prov = scenario_from_layout(
        layout, scenario_id="real_test_wh", source="synthetic.usd")

    validate_config(config)                         # raises if invalid
    assert config["scenario_id"] == "real_test_wh"
    # every station sits on a real grid node
    g = NavGraph(config)
    for kind in ("pick", "pack", "charge", "dock"):
        for st in config["stations"][kind]:
            g.node_index(st["node"])                # raises if missing

    result, _rows = run_rollout(config, seed=0, write_log=False)
    assert result["metrics"]["orders_completed"] > 0


def test_extracted_config_runs_at_real_pitch():
    cloud = make_cloud([-10, 0, 9], list(range(0, 51, 2)))
    layout = extract_layout(cloud)
    config, prov = scenario_from_layout(
        layout, scenario_id="real_pitch_wh", source="s.usd")
    # the real pitch is written into the contract and drives the sim geometry
    assert config["layout"]["grid"]["aisle_spacing_m"] == pytest.approx(9.5)
    g = NavGraph(config)
    assert g.spacing == pytest.approx(9.5)
    # coordinate_map now reports sim == real pitch (no render-only gap)
    assert prov["coordinate_map"]["sim_pitch_m"] == pytest.approx(9.5)


def test_coordinate_map_places_nodes_on_real_aisles():
    cloud = make_cloud([-10, 0, 9], list(range(0, 51, 2)), floor_z=0.05)
    layout = extract_layout(cloud)
    _config, prov = scenario_from_layout(
        layout, scenario_id="real_test_wh", source="synthetic.usd")

    cmap = prov["coordinate_map"]
    assert cmap["frame"]["up_axis"] == "Z"
    assert cmap["frame"]["floor_z"] == 0.05
    assert cmap["aisle_world"] == [-5.0, 4.5]
    assert cmap["sim_pitch_m"] == pytest.approx(9.5)  # sim now runs at real pitch
    # node A2_25 -> real world via the documented map
    a, p = 2, 25
    world_x = cmap["aisle_world"][a - 1]
    world_y = cmap["length_origin_m"] + p * cmap["length_scale"]
    assert world_x == 4.5
    assert world_y == pytest.approx(25.0)


def test_two_blocks_detect_interior_cross_aisle():
    # racks in y 0..20 and 30..48: a ~10 m cross-aisle gap splits two blocks.
    ys = list(range(0, 21, 2)) + list(range(30, 49, 2))
    cloud = make_cloud([-5, 0, 5], ys)
    layout = extract_layout(cloud)

    assert layout.aisles == 2
    assert layout.cross_aisles[0] == 0
    assert layout.cross_aisles[-1] == layout.aisle_length_m
    interior = [c for c in layout.cross_aisles
                if 0 < c < layout.aisle_length_m]
    assert len(interior) == 1
    assert interior[0] == pytest.approx(25, abs=3)  # mid-gap


def test_crop_to_contract_a_aisle_cap():
    rows = list(range(0, 12 * 5, 5))                # 12 rows -> 11 aisles
    cloud = make_cloud(rows, list(range(0, 41, 2)))
    layout = extract_layout(cloud)

    assert layout.n_rack_rows == 12
    assert layout.aisles == 10                      # clamped to Contract A cap
    assert layout.cropped


def test_extraction_is_deterministic():
    cloud = make_cloud([-10, 0, 9], list(range(0, 51, 2)))
    c1, p1 = scenario_from_layout(extract_layout(cloud),
                                  scenario_id="real_det_wh", source="s.usd")
    c2, p2 = scenario_from_layout(extract_layout(cloud),
                                  scenario_id="real_det_wh", source="s.usd")
    assert canonical_json(c1) == canonical_json(c2)
    assert p1 == p2


def test_scenario_from_cloud_json(tmp_path):
    # mirrors the two-env flow: renderer.dump_rack_cloud writes this JSON in the
    # Isaac venv; the main env loads it here (no Isaac, no pxr).
    rows, ys = [-10, 0, 9], list(range(0, 51, 2))
    cloud = {
        "source": "full_warehouse.usd", "meters_per_unit": 1.0, "up_axis": "Z",
        "floor_z": 0.0,
        "pitch": [float(x) for x in rows for _ in ys],
        "length": [float(y) for _ in rows for y in ys],
    }
    p = tmp_path / "cloud.json"
    p.write_text(json.dumps(cloud), encoding="utf-8")

    config, prov = scenario_from_cloud(str(p), scenario_id="real_fw_test")
    assert config["scenario_id"] == "real_fw_test"
    assert config["layout"]["grid"]["aisles"] == 2
    assert config["layout"]["grid"]["aisle_spacing_m"] == pytest.approx(9.5)
    assert prov["coordinate_map"]["source_usd"] == "full_warehouse.usd"


def test_dense_band_trims_sparse_front_and_grounds_picks():
    # a dense storage block (Y 20..40) plus a few sparse staging racks up front
    # (Y 0..10): the extractor must clamp the zone to the dense block so robots
    # work in the racks, not the open front — and pick faces sit on real rows.
    rows = [-10, 0, 9]
    pitch, length = [], []
    for x in rows:                          # dense block: a rack every metre
        for y in range(20, 41):
            pitch.append(float(x))
            length.append(float(y))
    for y in (0, 5, 10):                    # sparse front staging racks
        pitch.append(-10.0)
        length.append(float(y))
    cloud = RackCloud(pitch=pitch, length=length, floor_z=0.0,
                      meters_per_unit=1.0, up_axis="Z", source="s.usd")

    layout = extract_layout(cloud)
    lo, hi = layout.dense_band_m
    assert lo >= 18.0 and hi <= 42.0        # front staging trimmed off
    assert layout.aisle_length_m == pytest.approx(hi - lo, abs=1)
    assert layout.rack_positions            # grounded pick depths
    assert all(0 < p < layout.aisle_length_m for p in layout.rack_positions)

    config, _ = scenario_from_layout(layout, scenario_id="real_dense_wh",
                                     source="s.usd")
    # every pick face lands on a real rack row (a grounded position)
    for st in config["stations"]["pick"]:
        assert int(st["node"].split("_")[1]) in layout.rack_positions


def test_single_row_is_not_extractable():
    cloud = make_cloud([0], list(range(0, 41, 2)))  # one rack row -> no aisle
    with pytest.raises(ExtractError):
        extract_layout(cloud)


def test_non_z_up_is_rejected():
    cloud = make_cloud([-10, 0, 9], list(range(0, 51, 2)), up="Y")
    with pytest.raises(ExtractError):
        extract_layout(cloud)
