"""Where the option-C speed trick LEAKS: battery-bound regime.

Cross-aisle battery drain still uses length 3.0 under option C (drain = length,
unaffected by max_speed). So when charging binds, C under-drains vs real pitch
and spuriously looks better. Lower battery_capacity_m so charging binds, then
compare charge downtime + p95 across {real 9.5, abstract 3.0, option C}.
"""
from __future__ import annotations

import copy
import json
import statistics

import sim.navgraph as navgraph
from sim.runner import run_rollout

BASE = json.load(open("eval/dev_scenarios/braess_dev.config.json"))
BASE["fleet"]["battery_capacity_m"] = 2000          # make charging bind

C_SPEED = round(1.5 * 3.0 / 9.5, 4)
BASE_C = copy.deepcopy(BASE)
BASE_C["layout"]["edge_overrides"] = [
    {"edge": f"A{a}_{c:02d}->A{a + 1}_{c:02d}", "max_speed_mps": C_SPEED}
    for c in (0, 30) for a in range(1, 6)
]

SEEDS = list(range(8))


def run(config, pitch):
    navgraph.AISLE_SPACING_M = pitch
    p95, charge = [], []
    for s in SEEDS:
        m = run_rollout(config, seed=s, write_log=False)[0]["metrics"]
        p95.append(m["p95_order_latency_min"])
        charge.append(m["charge_downtime_pct"])
    return statistics.mean(p95), statistics.mean(charge)


print(f"battery_capacity_m = {BASE['fleet']['battery_capacity_m']} (charging binds)\n")
print(f"{'condition':>34} | {'p95_lat':>8} | {'charge_downtime%':>16}")
gt = run(BASE, 9.5)
ab = run(BASE, 3.0)
c = run(BASE_C, 3.0)
for label, (p95, ch) in (("REAL pitch 9.5 (ground truth)", gt),
                         ("abstract 3.0 (option A)", ab),
                         ("3.0 + speed trick (option C)", c)):
    print(f"{label:>34} | {p95:8.3f} | {ch:16.3f}")

print("\nError vs ground truth (real 9.5):")
for label, (p95, ch) in (("option A", ab), ("option C", c)):
    print(f"  {label:>10}: p95 {p95 - gt[0]:+7.3f} | charge_downtime% {ch - gt[1]:+7.3f}")
