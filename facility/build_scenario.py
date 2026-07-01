"""Emit a grounded scenario from a FacilitySpec.

  python -m facility.build_scenario --facility dc_pickzone_med \
      --out-dir eval/dev_scenarios

Writes <facility_id>.config.json (a frozen Contract A instance the sim, agent,
engine, and renderer all consume) plus <facility_id>.provenance.json recording
the real source — so every number traces back to a named footprint and dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .facilities import FACILITIES
from .spec import facility_to_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_scenario")
    ap.add_argument("--facility", required=True, choices=sorted(FACILITIES))
    ap.add_argument("--out-dir", default="eval/dev_scenarios")
    args = ap.parse_args(argv)

    spec = FACILITIES[args.facility]
    config, provenance = facility_to_config(spec)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / f"{spec.facility_id}.config.json"
    prov_path = out / f"{spec.facility_id}.provenance.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(json.dumps({"config": str(cfg_path), "provenance": str(prov_path),
                      "footprint_m": provenance["footprint_m"],
                      "pick_faces": provenance["n_pick_faces"],
                      "amrs": provenance["fleet_amr_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
