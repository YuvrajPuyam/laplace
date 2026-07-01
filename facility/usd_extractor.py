"""facility/usd_extractor.py — the scan-to-sim bridge (spec §10).

Extract a parallel-aisle layout FROM a real warehouse USD and lower it to a
frozen Contract A config (NO schema change), plus a provenance record carrying
the REAL geometry and a sim->world coordinate map. One USD becomes the single
source of truth for both the simulated dynamics and the rendered twin: the sim
runs on the extracted topology, and the renderer can place robots on the real
aisles via the coordinate map.

TWO-ENVIRONMENT SPLIT (mirrors export_scene/build_stage): the Isaac venv has
pxr but NOT this module's deps (sim -> jsonschema/numpy), so the USD *traversal*
cannot live here. It lives in `renderer/dump_rack_cloud.py` (pxr-only, Isaac
venv), which writes a rack-cloud JSON. THIS module (main env) loads that JSON
and builds the validated config:

  step 1 (Isaac venv):  python -m renderer.dump_rack_cloud <usd> --out cloud.json
  step 2 (main env):    python -m facility.usd_extractor --cloud cloud.json ...

  * `load_cloud` / `scenario_from_cloud`: JSON -> RackCloud -> config. Main env.
  * `extract_layout`, `scenario_from_layout`: pure, unit-testable on any
    RackCloud (synthetic or loaded). No pxr/Isaac dependency.

PITCH FIDELITY (Yuv-approved 2026-06-14): the extracted scenario sets the
optional Contract A field `layout.grid.aisle_spacing_m` to the real pitch, so
cross-aisle travel time, battery drain, routing and congestion all run at the
warehouse's true geometry — not the historic 3.0 m default. The field is read
lazily by `sim.navgraph.NavGraph` (grid.get, NOT default-filled), so scenarios
that omit it stay hash-identical to the pre-field canonical form. See
docs/PROJECT_STATE.md for the decision writeup.

  D:\\iv\\Scripts\\python.exe -m renderer.dump_rack_cloud \\
      /Isaac/Environments/Simple_Warehouse/full_warehouse.usd \\
      --out renderer/scenes/full_warehouse_cloud.json
  python -m facility.usd_extractor --cloud renderer/scenes/full_warehouse_cloud.json \\
      --scenario-id real_full_warehouse --out-dir eval/dev_scenarios
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import statistics
from pathlib import Path

from sim.config import config_hash
from sim.navgraph import AISLE_SPACING_M

from .spec import assemble_config

# Contract A caps (config.schema.json) the extractor must respect.
MAX_AISLES = 10
MIN_AISLE_LENGTH_M = 10
MAX_AISLE_LENGTH_M = 60
MAX_CROSS_AISLES = 6

# Prim-name substrings that mark a storage rack/shelf in the shortlist USDs.
RACK_KEYS = ("rack", "shelf", "pile", "pallet")


class ExtractError(ValueError):
    """The USD geometry does not yield an extractable parallel-aisle layout."""


@dataclasses.dataclass
class RackCloud:
    """Rack/shelf centroids pulled from a USD, normalised to METRES on a
    floor-plane frame: `pitch` runs across aisles (rack rows are separated along
    it), `length` runs along the aisles. `floor_z` is the rack-base height the
    renderer drops robots onto. Z-up only (the shortlist USDs are all Z-up)."""

    pitch: list[float]
    length: list[float]
    floor_z: float
    meters_per_unit: float
    up_axis: str
    source: str


@dataclasses.dataclass
class ExtractedLayout:
    """The parallel-aisle layout recovered from a RackCloud, in both real
    (metres) and Contract A (grid) terms."""

    # Contract A topology
    aisles: int
    aisle_length_m: int
    cross_aisles: list[int]
    # real geometry (metres) — drives the renderer coordinate map / provenance
    aisle_world: list[float]        # real pitch-axis coord per sim aisle 1..aisles
    length_origin_m: float          # real length-axis coord at sim position 0
    length_scale: float             # real metres per sim position unit
    real_pitch_m: float             # representative aisle pitch
    real_length_m: float            # real aisle length (pre-clamp)
    floor_z: float
    meters_per_unit: float          # source USD's metersPerUnit (map is in metres)
    n_rack_rows: int
    cropped: bool                   # True if more aisles than Contract A allows
    rack_positions: list[int]       # sim positions of REAL rack rows (grounded picks)
    dense_band_m: tuple[float, float]  # real length-axis span of the dense block
    notes: str


# --------------------------------------------------------------------------
# Pure extraction
# --------------------------------------------------------------------------
def _cluster_1d(values: list[float], merge_tol: float) -> list[list[float]]:
    """Gap-split 1-D clustering: a new cluster starts wherever the sorted gap
    exceeds `merge_tol`. Returns clusters (lists of member values), sorted."""
    if not values:
        return []
    xs = sorted(values)
    clusters: list[list[float]] = [[xs[0]]]
    for v in xs[1:]:
        if v - clusters[-1][-1] > merge_tol:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    return clusters


def _detect_interior_cross_aisles(length_coords: list[float], lo: float,
                                  hi: float, min_gap_m: float) -> list[float]:
    """Real length-axis positions of interior cross-aisles: contiguous bands
    (>= min_gap_m wide) with no rack coverage, between the front and back ends.
    These are the gaps that split storage blocks."""
    span = hi - lo
    if span <= 0:
        return []
    nbins = max(1, int(round(span)))
    occ = [False] * (nbins + 1)
    for v in length_coords:
        b = min(max(int((v - lo) / span * nbins), 0), nbins)
        occ[b] = True
    centers: list[float] = []
    i = 0
    while i <= nbins:
        if occ[i]:
            i += 1
            continue
        j = i
        while j <= nbins and not occ[j]:
            j += 1
        run_lo = lo + i / nbins * span
        run_hi = lo + j / nbins * span
        interior = i > 0 and j <= nbins          # not touching either end
        if interior and (run_hi - run_lo) >= min_gap_m:
            centers.append((run_lo + run_hi) / 2.0)
        i = j
    return centers


def _dense_band(length_coords: list[float], rel_thresh: float = 0.2,
                min_count: int = 2) -> tuple[float, float]:
    """[lo, hi] of the length axis holding the DENSE rack mass, trimming sparse
    outliers (e.g. a few staging pallets far from the storage block). Bins racks
    at ~1 m, keeps bins with count >= max(min_count, rel_thresh*peak), and spans
    the first..last dense bin. A uniformly-dense OR a genuine multi-block layout
    keeps its full span (the gap between blocks is interior, not a fringe); only
    sparse edges are trimmed. This is what stops the front staging area from
    stretching the extracted pick zone (the full_warehouse case)."""
    lo, hi = min(length_coords), max(length_coords)
    span = hi - lo
    if span <= 0:
        return lo, hi
    nbins = max(1, int(round(span)))
    counts = [0] * (nbins + 1)
    for v in length_coords:
        counts[min(max(int((v - lo) / span * nbins), 0), nbins)] += 1
    thresh = max(min_count, rel_thresh * max(counts))
    dense = [i for i, c in enumerate(counts) if c >= thresh]
    if not dense:
        return lo, hi
    return lo + dense[0] / nbins * span, lo + dense[-1] / nbins * span


def _crop_aisles(aisle_lines: list[float], pitch: list[float],
                 max_aisles: int) -> tuple[list[float], int]:
    """Pick the contiguous window of `max_aisles` aisle lines covering the most
    racks (the dense core). Returns (window, start_index)."""
    best, best_idx, best_count = aisle_lines[:max_aisles], 0, -1
    for s in range(len(aisle_lines) - max_aisles + 1):
        window = aisle_lines[s:s + max_aisles]
        lo, hi = window[0], window[-1]
        cnt = sum(1 for x in pitch if lo <= x <= hi)
        if cnt > best_count:
            best, best_idx, best_count = window, s, cnt
    return best, best_idx


def _thin_cross(cross: list[int], max_cross: int) -> list[int]:
    """Keep both ends, then evenly subsample the interior down to max_cross."""
    if len(cross) <= max_cross:
        return cross
    ends = [cross[0], cross[-1]]
    interior = cross[1:-1]
    keep = max_cross - 2
    if keep <= 0:
        return sorted(set(ends))
    step = (len(interior) - 1) / (keep - 1) if keep > 1 else 0
    picked = [interior[round(i * step)] for i in range(keep)]
    return sorted(set(ends) | set(picked))


def extract_layout(cloud: RackCloud, *, row_merge_tol_m: float = 2.0,
                   cross_min_gap_m: float = 2.5) -> ExtractedLayout:
    """Recover a parallel-aisle layout from rack centroids.

    Rack rows are clusters along the pitch axis; aisles are the walkable gaps
    between adjacent rows (so aisles = rows - 1, placed at row midpoints). Aisle
    length is the rack span along the length axis; interior cross-aisles are the
    length-axis gaps between storage blocks. The front (0) and back ends are
    always cross-aisles. The result is clamped to Contract A's caps, recording
    any crop in provenance.
    """
    if cloud.up_axis.upper() != "Z":
        raise ExtractError(
            f"up-axis {cloud.up_axis!r} unsupported; extractor assumes Z-up "
            "(all shortlist warehouse USDs are Z-up). Re-orient the stage first.")
    if len(cloud.pitch) < 2:
        raise ExtractError(
            f"found {len(cloud.pitch)} rack/shelf prim(s); need >=2 to form a row")

    rows = [statistics.fmean(c) for c in _cluster_1d(cloud.pitch, row_merge_tol_m)]
    if len(rows) < 2:
        raise ExtractError(
            f"only {len(rows)} rack row(s) detected (merge_tol={row_merge_tol_m} m); "
            "need >=2 adjacent rows to form an aisle")

    aisle_lines = [(rows[i] + rows[i + 1]) / 2.0 for i in range(len(rows) - 1)]
    cropped = False
    if len(aisle_lines) > MAX_AISLES:
        aisle_lines, _ = _crop_aisles(aisle_lines, cloud.pitch, MAX_AISLES)
        cropped = True
    aisles = len(aisle_lines)

    # Clamp the length axis to the DENSE rack block — drops sparse staging/
    # outlier racks that would otherwise stretch the zone into the open front.
    length_min, length_max = _dense_band(cloud.length)
    real_length = length_max - length_min
    if real_length <= 0:
        raise ExtractError("racks have zero extent along the length axis")
    aisle_length_m = max(MIN_AISLE_LENGTH_M,
                         min(int(round(real_length)), MAX_AISLE_LENGTH_M))
    length_scale = real_length / aisle_length_m

    band_ys = [y for y in cloud.length if length_min <= y <= length_max]
    interior = _detect_interior_cross_aisles(
        band_ys, length_min, length_max, cross_min_gap_m)
    cross = {0, aisle_length_m}
    for c in interior:
        p = int(round((c - length_min) / length_scale))
        if 0 < p < aisle_length_m:
            cross.add(p)
    cross_aisles = _thin_cross(sorted(cross), MAX_CROSS_AISLES)

    # grounded pick depths: sim positions of REAL rack rows in the dense block
    rack_positions = sorted({
        int(round((y - length_min) / length_scale)) for y in band_ys})
    rack_positions = [p for p in rack_positions if 0 < p < aisle_length_m]

    pitches = [aisle_lines[i + 1] - aisle_lines[i] for i in range(aisles - 1)]
    real_pitch = statistics.fmean(pitches) if pitches else AISLE_SPACING_M

    return ExtractedLayout(
        aisles=aisles, aisle_length_m=aisle_length_m, cross_aisles=cross_aisles,
        aisle_world=[round(x, 4) for x in aisle_lines],
        length_origin_m=round(length_min, 4),
        length_scale=round(length_scale, 6),
        real_pitch_m=round(real_pitch, 4),
        real_length_m=round(real_length, 4),
        floor_z=round(cloud.floor_z, 4), meters_per_unit=cloud.meters_per_unit,
        n_rack_rows=len(rows), cropped=cropped,
        rack_positions=rack_positions,
        dense_band_m=(round(length_min, 4), round(length_max, 4)),
        notes="aisles = rack rows - 1 at row midpoints; length clamped to the "
              "dense rack block (sparse staging racks trimmed); pick faces on "
              "real rack rows; cross-aisles at block ends + inter-block gaps; "
              f"clamped to Contract A caps (<= {MAX_AISLES} aisles, "
              f"<= {MAX_AISLE_LENGTH_M} m).",
    )


def _default_station_counts(aisles: int) -> tuple[int, int, int]:
    """Heuristic pick/pack/charge counts scaled to the layout, within Contract A
    limits (pick<=8, pack<=6, charge<=4). Geometry comes from the USD; these
    operational counts are a stated assumption recorded in provenance."""
    n_pick = min(8, max(2, aisles))
    n_pack = min(6, max(2, (aisles + 2) // 3))
    n_charge = min(4, max(1, (aisles + 3) // 4))
    return n_pick, n_pack, n_charge


def scenario_from_layout(
    layout: ExtractedLayout, *, scenario_id: str, source: str,
    n_pick: int | None = None, n_pack: int | None = None,
    n_charge: int | None = None,
    amr_count: int = 6, speed_mps: float = 1.5, battery_capacity_m: int = 8000,
    charge_minutes: float = 15, routing: str = "shortest_path",
    arrival_rate_per_min: float = 1.0, pack_assignment: str = "shortest_queue",
    sim_minutes: int = 480, warmup_minutes: int = 30,
    demand_source: str = "",
) -> tuple[dict, dict]:
    """Lower an ExtractedLayout to (Contract A config, provenance).

    Geometry (aisles, length, cross-aisles) is extracted from the USD. Station
    counts and the fleet/demand regime are operational parameters supplied by
    the caller (a USD has no fleet); their defaults mirror dc_pickzone_med and
    are disclosed in provenance. The provenance carries a `coordinate_map` the
    renderer uses to place sim nodes on the real aisles."""
    dp, dk, dc = _default_station_counts(layout.aisles)
    n_pick = dp if n_pick is None else n_pick
    n_pack = dk if n_pack is None else n_pack
    n_charge = dc if n_charge is None else n_charge

    config = assemble_config(
        scenario_id=scenario_id, aisles=layout.aisles,
        aisle_length_m=layout.aisle_length_m, cross=layout.cross_aisles,
        n_pick=n_pick, n_pack=n_pack, n_charge=n_charge,
        amr_count=amr_count, speed_mps=speed_mps,
        battery_capacity_m=battery_capacity_m, charge_minutes=charge_minutes,
        routing=routing, arrival_rate_per_min=arrival_rate_per_min,
        pack_assignment=pack_assignment, sim_minutes=sim_minutes,
        warmup_minutes=warmup_minutes,
        # Make the sim geometrically faithful: cross-aisle dynamics run at the
        # real extracted pitch via the optional Contract A field.
        aisle_spacing_m=round(layout.real_pitch_m, 4),
        # Place pick faces on the REAL rack rows of the dense block.
        pick_positions=layout.rack_positions,
    )

    provenance = {
        "facility_id": scenario_id,
        "name": f"Real warehouse extracted from {source}",
        "layout_source": (
            f"Scan-to-sim extraction from USD '{source}'. "
            f"{layout.n_rack_rows} rack rows -> {layout.aisles} aisles "
            f"(midpoints), aisle length {layout.real_length_m} m, real aisle "
            f"pitch {layout.real_pitch_m} m. " + layout.notes),
        "demand_source": demand_source or (
            "NOT from the USD: a warehouse mesh carries no order stream. Fleet "
            "and arrival rate are stated operational parameters; calibrate "
            "arrival_rate_per_min from real orders (facility/demand.py) before "
            "quoting absolute throughput."),
        "notes": (
            f"Geometry extracted from a real USD; sim runs the extracted "
            f"topology. Station counts (pick={n_pick}, pack={n_pack}, "
            f"charge={n_charge}) and fleet are operational assumptions, not in "
            f"the mesh." + (" Layout CROPPED to Contract A's 10-aisle cap; the "
            "densest contiguous band of aisles was kept." if layout.cropped
            else "")),
        "footprint_m": {
            "width": round(layout.aisle_world[-1] - layout.aisle_world[0], 2)
            if len(layout.aisle_world) > 1 else 0.0,
            "length": layout.real_length_m,
        },
        "cross_aisles": layout.cross_aisles,
        "n_pick_faces": len(config["stations"]["pick"]),
        "fleet_amr_count": amr_count,
        "arrival_rate_per_min": arrival_rate_per_min,
        # The scan-to-sim bridge: place sim node A{a}_{p} in the real USD at
        #   x = aisle_world[a-1]
        #   y = length_origin_m + p * length_scale
        #   z = floor_z
        # (Z-up, metres). Lets the renderer drop robots onto the real aisles
        # without the sim carrying real coordinates.
        "coordinate_map": {
            "frame": {"up_axis": "Z", "units": "metres",
                      "source_meters_per_unit": layout.meters_per_unit,
                      "floor_z": layout.floor_z},
            "pitch_axis": "x", "length_axis": "y",
            "aisle_world": layout.aisle_world,
            "length_origin_m": layout.length_origin_m,
            "length_scale": layout.length_scale,
            "real_pitch_m": layout.real_pitch_m,
            "real_length_m": layout.real_length_m,
            "sim_pitch_m": round(layout.real_pitch_m, 4),
            "source_usd": source,
            "note": ("Sim AND render run at the real pitch: the config sets "
                     "layout.grid.aisle_spacing_m = real_pitch_m, so cross-aisle "
                     "travel time, battery, routing and congestion are all "
                     "geometrically faithful (no longer a render-only value)."),
        },
    }
    return config, provenance


def _slug(usd_path: str) -> str:
    """Contract A scenario_id (^[a-z0-9_]{3,40}$) from a USD path stem."""
    stem = Path(usd_path).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "real_warehouse"
    slug = f"real_{slug}" if not slug.startswith("real") else slug
    return slug[:40].rstrip("_")


# --------------------------------------------------------------------------
# Main-env bridge: load the rack-cloud JSON (dumped by renderer.dump_rack_cloud
# in the Isaac venv) and build the validated config. No pxr/Isaac dependency.
# --------------------------------------------------------------------------
def load_cloud(cloud_json: str | Path) -> RackCloud:
    """Load a rack-cloud JSON (from renderer.dump_rack_cloud) into a RackCloud."""
    with open(cloud_json, encoding="utf-8") as f:
        d = json.load(f)
    return RackCloud(
        pitch=[float(v) for v in d["pitch"]],
        length=[float(v) for v in d["length"]],
        floor_z=float(d.get("floor_z", 0.0)),
        meters_per_unit=float(d.get("meters_per_unit", 1.0)),
        up_axis=str(d.get("up_axis", "Z")),
        source=str(d.get("source", str(cloud_json))))


def scenario_from_cloud(cloud_json: str | Path, *, scenario_id: str | None = None,
                        row_merge_tol_m: float = 2.0, cross_min_gap_m: float = 2.5,
                        **scenario_kwargs) -> tuple[dict, dict]:
    """Rack-cloud JSON -> (Contract A config, provenance). Main env.
    `scenario_kwargs` pass through to `scenario_from_layout`."""
    cloud = load_cloud(cloud_json)
    layout = extract_layout(cloud, row_merge_tol_m=row_merge_tol_m,
                            cross_min_gap_m=cross_min_gap_m)
    return scenario_from_layout(
        layout, scenario_id=scenario_id or _slug(cloud.source), source=cloud.source,
        **scenario_kwargs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="usd_extractor",
        description="Build a Contract A scenario from a rack-cloud JSON. Dump the "
                    "JSON first in the Isaac venv: D:\\iv\\Scripts\\python.exe -m "
                    "renderer.dump_rack_cloud <usd> --out cloud.json")
    ap.add_argument("--cloud", required=True,
                    help="rack-cloud JSON from renderer.dump_rack_cloud")
    ap.add_argument("--scenario-id", default=None)
    ap.add_argument("--out-dir", default="eval/dev_scenarios")
    ap.add_argument("--row-merge-tol-m", type=float, default=2.0)
    ap.add_argument("--cross-min-gap-m", type=float, default=2.5)
    ap.add_argument("--amr-count", type=int, default=6)
    ap.add_argument("--arrival-rate-per-min", type=float, default=1.0)
    args = ap.parse_args(argv)

    config, provenance = scenario_from_cloud(
        args.cloud, scenario_id=args.scenario_id,
        row_merge_tol_m=args.row_merge_tol_m, cross_min_gap_m=args.cross_min_gap_m,
        amr_count=args.amr_count, arrival_rate_per_min=args.arrival_rate_per_min)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sid = config["scenario_id"]
    (out / f"{sid}.config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    (out / f"{sid}.provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({
        "scenario_id": sid,
        "config": str(out / f"{sid}.config.json"),
        "provenance": str(out / f"{sid}.provenance.json"),
        "aisles": config["layout"]["grid"]["aisles"],
        "aisle_length_m": config["layout"]["grid"]["aisle_length_m"],
        "cross_aisles": config["layout"]["grid"]["cross_aisles"],
        "real_pitch_m": provenance["coordinate_map"]["real_pitch_m"],
        "cropped": "CROPPED" in provenance["notes"],
        "config_hash": config_hash(config),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

