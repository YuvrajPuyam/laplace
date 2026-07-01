"""Build a self-contained web-viewer HTML from a scenario config + a tracks
rollout — a geometrically-faithful SCHEMATIC twin (no USD, no Isaac, no backend).

Layout (floor, rack blocks, aisle lines, stations) is derived from the Contract A
config via the navgraph; motion + per-frame state come from a tracks.json
(renderer/export_tracks). The result is one HTML file you open in a browser.

  python -m ui.export_web_scene --scenario real_full_warehouse \\
      --tracks renderer/scenes/real_full_warehouse_tracks.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.config import load_config
from sim.navgraph import NavGraph

TEMPLATE = Path("ui/viewer_template.html")


def build_scene(config: dict, tracks: dict, hud: dict | None = None) -> dict:
    g = NavGraph(config)
    grid = config["layout"]["grid"]
    aisles, length = grid["aisles"], grid["aisle_length_m"]
    spacing = g.spacing
    xs = [(a - 1) * spacing for a in range(1, aisles + 1)]
    cross = sorted(grid["cross_aisles"])

    racks = []
    for i in range(len(xs) - 1):
        for j in range(len(cross) - 1):
            x0, x1, y0, y1 = xs[i], xs[i + 1], cross[j], cross[j + 1]
            racks.append({"cx": round((x0 + x1) / 2, 2), "cy": round((y0 + y1) / 2, 2),
                          "w": round((x1 - x0) * 0.62, 2), "d": round((y1 - y0) * 0.9, 2)})

    stations = []
    for kind in ("pick", "pack", "charge", "dock"):
        for s in config["stations"][kind]:
            x, y = g.node_xy[g.node_index(s["node"])]
            stations.append({"id": s["id"], "kind": kind,
                             "x": round(x, 2), "y": round(y, 2),
                             "slots": s.get("slots", 1)})

    agents = {}
    for aid, a in tracks["agents"].items():
        agents[aid] = {"x": [round(p[0], 3) for p in a["xy"]],
                       "y": [round(p[1], 3) for p in a["xy"]],
                       "band": a.get("band"), "on_lane": a.get("on_lane")}

    pad = spacing * 0.6
    return {
        "scenario": config["scenario_id"],
        "meta": {"aisles": aisles, "aisle_length_m": length, "spacing": round(spacing, 3),
                 "amr_count": config["fleet"]["amr_count"], "n_frames": tracks["n_frames"],
                 "dt": tracks["dt"], "t0": tracks["t0"],
                 "demand": config["demand"]["arrival_rate_per_min"]},
        "extent": {"xmin": min(xs) - pad, "xmax": max(xs) + pad,
                   "ymin": -2.0, "ymax": length + 2.0},
        "aisle_x": [round(x, 2) for x in xs], "cross_y": cross, "length": length,
        "racks": racks, "stations": stations, "agents": agents,
        "edges": tracks.get("edges"), "edge_occ": tracks.get("edge_occ"),
        "charge_occ": tracks.get("charge_occ"), "charge_cap": tracks.get("charge_cap"),
        "station_occ": tracks.get("station_occ"), "station_cap": tracks.get("station_cap"),
        "fleet": tracks.get("fleet"), "intent_color": tracks.get("intent_color"),
        "intent_label": tracks.get("intent_label"), "intent_order": tracks.get("intent_order"),
        "hud": hud["frames"] if hud else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="export_web_scene")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--hud", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg_path = args.config or f"eval/dev_scenarios/{args.scenario}.config.json"
    tracks_path = args.tracks or f"renderer/scenes/{args.scenario}_tracks.json"
    config = load_config(cfg_path)
    tracks = json.loads(Path(tracks_path).read_text(encoding="utf-8"))

    hud_path = args.hud or tracks_path.replace(".json", "_hud.json")
    hud = (json.loads(Path(hud_path).read_text(encoding="utf-8"))
           if Path(hud_path).exists() else None)

    scene = build_scene(config, tracks, hud)
    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("/*__SCENE__*/", json.dumps(scene, separators=(",", ":")))
            .replace("/*__BOOT__*/", json.dumps({"engine": None})))  # standalone: view-only

    out = Path(args.out or f"ui/out/{args.scenario}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(json.dumps({"out": str(out), "scenario": scene["scenario"],
                      "racks": len(scene["racks"]), "stations": len(scene["stations"]),
                      "agents": len(scene["agents"]), "frames": scene["meta"]["n_frames"],
                      "hud": scene["hud"] is not None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
