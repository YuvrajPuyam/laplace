"""Pitch fidelity: two questions, empirically.

Q1 Is the headline Braess (open-a-cross-aisle) decision pitch-sensitive?
Q2 Does the zero-schema-change "slow the cross-aisle" trick (option C) reproduce
   real-pitch dynamics, or does length-based routing/battery leak the gap?

braess_dev, base vs "open a mid cross-aisle at p=15", CRN-paired across seeds.
Pitch changed by monkeypatching AISLE_SPACING_M (no file edits, no contract change).
"""
from __future__ import annotations

import copy
import json
import statistics

import sim.navgraph as navgraph
from sim.runner import run_rollout

BASE = json.load(open("eval/dev_scenarios/braess_dev.config.json"))

MID = [{"from": f"A{a}_15", "to": f"A{a + 1}_15"} for a in range(1, 6)]
TREATMENT = copy.deepcopy(BASE)
TREATMENT["layout"]["extra_edges"] = MID

# Option C: at sim pitch 3.0, slow cross-aisle edges so TRAVEL TIME matches the
# real 9.5 m pitch. speed' solves 3.0/(s'*60) = 9.5/(1.5*60) -> s' = 1.5*3/9.5.
C_SPEED = round(1.5 * 3.0 / 9.5, 4)


def cross_overrides(positions, include_mid):
    ov = [{"edge": f"A{a}_{c:02d}->A{a + 1}_{c:02d}", "max_speed_mps": C_SPEED}
          for c in positions for a in range(1, 6)]
    if include_mid:
        ov += [{"edge": f"A{a}_15->A{a + 1}_15", "max_speed_mps": C_SPEED}
               for a in range(1, 6)]
    return ov


BASE_C = copy.deepcopy(BASE)
BASE_C["layout"]["edge_overrides"] = cross_overrides([0, 30], include_mid=False)
TREAT_C = copy.deepcopy(TREATMENT)
TREAT_C["layout"]["edge_overrides"] = cross_overrides([0, 30], include_mid=True)

SEEDS = list(range(8))


def mean_metrics(config, pitch):
    navgraph.AISLE_SPACING_M = pitch
    thr, p95 = [], []
    for s in SEEDS:
        res, _ = run_rollout(config, seed=s, write_log=False)
        m = res["metrics"]
        thr.append(m["throughput_orders_per_hr"])
        p95.append(m["p95_order_latency_min"])
    return statistics.mean(thr), statistics.mean(p95)


def decision(label, base_cfg, treat_cfg, pitch):
    b_thr, b_p95 = mean_metrics(base_cfg, pitch)
    t_thr, t_p95 = mean_metrics(treat_cfg, pitch)
    d_thr, d_p95 = t_thr - b_thr, t_p95 - b_p95
    print(f"{label:>34} | base p95={b_p95:7.3f} thr={b_thr:6.2f} | "
          f"d_p95={d_p95:+7.3f} d_thr={d_thr:+6.3f}")
    return b_p95, b_thr, d_p95, d_thr


print(f"C_SPEED override = {C_SPEED} m/s\n")
print("Q1 + Q2 — base-level absolutes and the open-cross-aisle decision delta:")
gt_p95, gt_thr, gt_d_p95, gt_d_thr = decision(
    "REAL pitch 9.5 (ground truth)", BASE, TREATMENT, 9.5)
ab_p95, ab_thr, ab_d_p95, ab_d_thr = decision(
    "sim pitch 3.0 (abstract / option A)", BASE, TREATMENT, 3.0)
c_p95, c_thr, c_d_p95, c_d_thr = decision(
    "sim 3.0 + speed trick (option C)", BASE_C, TREAT_C, 3.0)

print("\n=== Fidelity error vs ground truth (real 9.5) ===")
print(f"{'approach':>34} | {'base p95 err':>12} | {'decision d_p95 err':>18}")
for label, p95, dp95 in (("option A (abstract 3.0)", ab_p95, ab_d_p95),
                         ("option C (speed trick)", c_p95, c_d_p95)):
    print(f"{label:>34} | {p95 - gt_p95:+12.3f} | {dp95 - gt_d_p95:+18.3f}")
