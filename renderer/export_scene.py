"""export_scene — MAIN env CLI: scenario_id -> TwinScene JSON.

Bridges the two Python environments: this runs where the sim lives (main env)
and writes a portable scene.json the Isaac venv renders. Domain is selected by
name from the registry, so `--domain warehouse` today and other domains later
use the same command.

  python -m renderer.export_scene --scenario braess_dev --out renderer/scenes/braess.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.store import ScenarioStore
from sim.config import apply_patch

from .domains import REGISTRY


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="export_scene")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--domain", default="warehouse", choices=sorted(REGISTRY))
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", default=None,
                    help='JSON dot-path patch, e.g. \'{"fleet.amr_count": 5}\' '
                         "or a path to a .json file holding the patch")
    ap.add_argument("--assets", choices=["prim", "realistic"], default="prim",
                    help="prim = fast primitives; realistic = real NVIDIA USD assets")
    ap.add_argument("--scenario-dirs", nargs="*",
                    default=["examples", "eval/dev_scenarios"])
    args = ap.parse_args(argv)

    store = ScenarioStore(dirs=tuple(args.scenario_dirs))
    config = store.get(args.scenario)
    if config is None:
        raise SystemExit(f"unknown scenario '{args.scenario}'; known: {store.ids()}")

    if args.patch:
        text = Path(args.patch).read_text(encoding="utf-8") \
            if Path(args.patch).exists() else args.patch
        config = apply_patch(config, json.loads(text))

    import inspect
    to_scene = REGISTRY[args.domain].to_scene
    kwargs = {"realistic": args.assets == "realistic"} \
        if "realistic" in inspect.signature(to_scene).parameters else {}
    scene = to_scene(config, **kwargs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scene.to_json(args.out)
    print(json.dumps({"out": args.out, "domain": scene.domain,
                      "scenario": scene.scenario_id,
                      "nodes": len(scene.nodes), "lanes": len(scene.lanes),
                      "props": len(scene.props), "agents": len(scene.agents),
                      "cameras": [c.name for c in scene.cameras]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
