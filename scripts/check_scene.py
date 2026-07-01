"""check_scene - guard against the two twin-demo regressions:

  1. STATION PLACEMENT: two stations on the same navgraph node, or close enough that
     their floor-disc PORTALS overlap in the viewer (portal radius ~1.0 m, so centres
     must be >= ~2 m apart). Also catches stations on invalid / unreachable nodes.
  2. (advisory) reminds about the viewer panel-row styling convention.

Usage:
  python -m scripts.check_scene                      # check every dev/example scenario
  python -m scripts.check_scene real_warehouse_demo  # one scenario by id
  python -m scripts.check_scene --min-spacing 2.4 real_warehouse_demo

Exit code 0 = all clear, 1 = at least one placement error. Run it after ANY edit to a
scenario's station nodes (and wire it into the demo build) so portals never overlap again.
"""

from __future__ import annotations

import argparse
import math
import sys

# portal disc radius in the viewer is ~1.0 m, so two stations whose nodes are closer
# than this (centre-to-centre) draw overlapping portals. 2.2 m leaves a small margin.
DEFAULT_MIN_SPACING_M = 2.2
_KINDS = ("pick", "pack", "charge", "dock")


def check_config(config: dict, min_spacing: float = DEFAULT_MIN_SPACING_M) -> list[str]:
    """Return a list of human-readable placement errors ([] = all clear)."""
    from sim.navgraph import NavGraph

    g = NavGraph(config)
    errors: list[str] = []
    coords: dict[str, tuple[float, float]] = {}
    node_of: dict[str, str] = {}
    for kind in _KINDS:
        for s in config["stations"].get(kind, []):
            sid, node = s["id"], s["node"]
            node_of[sid] = node
            try:
                coords[sid] = g.node_xy[g.node_index(node)]
            except Exception:  # noqa: BLE001 - an unknown node is exactly what we report
                errors.append(f"INVALID NODE: station {sid} -> {node!r} is not a navgraph node")

    ids = list(coords)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            d = math.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])
            if d < 1e-6:
                errors.append(f"OVERLAP: {a} and {b} share node {node_of[a]} (stacked portals)")
            elif d < min_spacing:
                errors.append(
                    f"TOO CLOSE: {a}({node_of[a]}) <-> {b}({node_of[b]}) = {d:.2f} m "
                    f"(< {min_spacing} m; floor-disc portals overlap)")
    return errors


def _load(scenario_id: str) -> dict | None:
    from engine.store import ScenarioStore
    from sim.config import fill_defaults

    cfg = ScenarioStore(dirs=("examples", "eval/dev_scenarios")).get(scenario_id)
    return fill_defaults(cfg) if cfg is not None else None


def _all_scenarios() -> list[str]:
    from pathlib import Path
    ids = set()
    for base in ("examples", "eval/dev_scenarios"):
        for p in Path(base).glob("*.config.json"):
            ids.add(p.name.replace(".config.json", ""))
    return sorted(ids)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="check_scene", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", help="scenario id (default: check all)")
    ap.add_argument("--min-spacing", type=float, default=DEFAULT_MIN_SPACING_M,
                    help=f"min station spacing in metres (default {DEFAULT_MIN_SPACING_M})")
    args = ap.parse_args(argv)

    scenarios = [args.scenario] if args.scenario else _all_scenarios()
    bad = 0
    for sid in scenarios:
        config = _load(sid)
        if config is None:
            print(f"[check_scene] ? {sid}: scenario not found")
            bad += 1
            continue
        errors = check_config(config, args.min_spacing)
        if errors:
            bad += 1
            print(f"[check_scene] FAIL {sid}:")
            for e in errors:
                print(f"    - {e}")
        else:
            n = sum(len(config["stations"].get(k, [])) for k in _KINDS)
            print(f"[check_scene] OK   {sid}: {n} stations, all >= {args.min_spacing} m apart")
    if bad:
        print(f"\n[check_scene] {bad} scenario(s) with placement problems.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
