"""Quick fleet-size sweep on a grounded scenario (dev tooling).

  python scripts/fleet_sweep.py dc_pickzone_med 4 6 8 10
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import apply_patch, fill_defaults  # noqa: E402
from sim.runner import run_rollout  # noqa: E402

scenario = sys.argv[1] if len(sys.argv) > 1 else "dc_pickzone_med"
args = sys.argv[2:]
rate = None
amr_args = []
for a in args:
    if a.startswith("rate="):
        rate = float(a.split("=", 1)[1])
    else:
        amr_args.append(int(a))
amrs = amr_args or [4, 6, 8, 10]
base = json.loads(Path(f"eval/dev_scenarios/{scenario}.config.json").read_text())

print(f"{scenario}" + (f"  (arrival {rate}/min)" if rate else "") + ":")
for n in amrs:
    patch = {"fleet.amr_count": n}
    if rate is not None:
        patch["demand.arrival_rate_per_min"] = rate
    cfg = fill_defaults(apply_patch(base, patch))
    res, _ = run_rollout(cfg, seed=0, write_log=False)
    m = res["metrics"]
    print(f"  amr={n:2d}  thru={m['throughput_orders_per_hr']:6.1f}/hr  "
          f"util={m['amr_utilization_pct']:5.1f}%  "
          f"p95={m['p95_order_latency_min']:6.1f}min  "
          f"completed={m['orders_completed']:4d}  abandoned={m['orders_abandoned']:3d}")
