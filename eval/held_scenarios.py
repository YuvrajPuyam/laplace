"""Held-out T3 trap suite — the HEADLINE table's test cases (CLAUDE.md §4; spec §7).

These are deliberately kept OUT of eval/candidates.py: prompt/agent development uses
the dev set only, and these traps are never tuned against the agent. Each is a decision
where the *intuitive* answer is wrong and the sim's M/M/c station queueing (the formally
validated part of the model) produces a SIGNIFICANT reversal — so `llm_alone` walks into
the trap while `agent` runs experiments and escapes.

Why these three (and not fleet-collapse / Braess-hurts / charge-threshold): the sim is
deadlock-free by design, so edge congestion can't make more AMRs hurt and a shortcut can't
create a worse equilibrium; and the charge threshold is a frozen constant. The reversals
this sim CAN produce honestly are M/M/c station-queueing effects and connectivity relief —
which is exactly what's validated against the closed forms. See docs and the render-correctness
note for the "flag friction, don't work around it" call.

Each reversal validated at p<0.01 (paired bootstrap, 16-24 CRN seeds); see eval/gt_cache.
"""

from __future__ import annotations

from pathlib import Path

from eval.candidates import Candidate, Decision, _max_throughput

SCEN_DIR = Path("eval/scenarios")


def _min_p95(metrics: dict, params: dict) -> float:
    """Lower p95 order latency is better (encoded higher-is-better)."""
    return -metrics["p95_order_latency_min"]


def _picks(spec: list[tuple[str, int]], svc=(0.5, 0.45)) -> list[dict]:
    return [{"id": f"P{i + 1}", "node": node, "slots": slots, "service_lognorm": list(svc)}
            for i, (node, slots) in enumerate(spec)]


def _packs(spec: list[tuple[str, int]], svc=(0.45, 0.4)) -> list[dict]:
    return [{"id": f"K{i + 1}", "node": node, "slots": slots, "service_lognorm": list(svc)}
            for i, (node, slots) in enumerate(spec)]


def _mid_cross(n_aisles: int, pos: int) -> list[dict]:
    return [{"from": f"A{a}_{pos:02d}", "to": f"A{a + 1}_{pos:02d}"}
            for a in range(1, n_aisles)]


HELD_DECISIONS: dict[str, Decision] = {
    # TRAP 1 — server POOLING vs DISTRIBUTING (pick side). Intuition: spread pick
    # stations across aisles for locality/short travel. Reality: pooling the same total
    # slots into one M/M/c station cuts queue wait more than the extra travel costs.
    "pool_pickzone": Decision(
        scenario_id="pool_pickzone", base_dir=SCEN_DIR,
        question="Pick capacity is four server-slots. Should they be DISTRIBUTED as four "
                 "1-slot pick stations (one per aisle, for short travel), or POOLED into a "
                 "single 4-slot pick station? Choose the layout with the lower p95 order latency.",
        candidates=[
            Candidate("distributed_4x1", {}, {"layout": "distributed"}),
            Candidate("mid_2x2", {"stations.pick": _picks([("A2_10", 2), ("A3_10", 2)])},
                      {"layout": "mid"}),
            Candidate("pooled_1x4", {"stations.pick": _picks([("A2_10", 4)])},
                      {"layout": "pooled"}),
        ],
        score=_min_p95, metric_name="p95_order_latency_min",
        rationale="Pooling beats distributing: M/M/4 has far lower queue wait than 4x M/M/1 "
                  "at the same load, and in a compact zone the extra travel to one station is "
                  "small. The locality intuition (spread for short trips) is the trap."),

    # TRAP 2 — Braess 'shortcut HELPS' (defies the congestion/Braess fear). Stations sit
    # mid-aisle with cross-aisles only at the far ends, so without a shortcut switching aisles
    # is a long detour; a mid cross-aisle relieves it. Intuition: more connectivity congests.
    "braess_holdout": Decision(
        scenario_id="braess_holdout", base_dir=SCEN_DIR,
        question="Should we open a mid cross-aisle connecting all aisles at their midpoint, or "
                 "leave the layout with cross-aisles only at the ends? Choose the option with "
                 "higher throughput.",
        candidates=[
            Candidate("no_shortcut", {}, {"shortcut": False}),
            Candidate("mid_shortcut", {"layout.extra_edges": _mid_cross(5, 20)},
                      {"shortcut": True}),
        ],
        score=_max_throughput, metric_name="throughput_orders_per_hr",
        rationale="The shortcut HELPS here (relieves the end-detour), defying the naive Braess "
                  "fear that extra connectivity congests. The sim is deadlock-free, so added "
                  "edges relieve load rather than create a worse equilibrium."),

    # TRAP 3 — server POOLING vs DISTRIBUTING (pack side). Same M/M/c effect, different
    # station domain, tuned to the SUBTLE regime (distributed is stable but worse, not collapsed).
    "pool_packzone": Decision(
        scenario_id="pool_packzone", base_dir=SCEN_DIR,
        question="Pack capacity is four server-slots. Distribute them as four 1-slot pack "
                 "stations (one per aisle), or pool them into a single 4-slot pack station? "
                 "Choose the layout with the lower p95 order latency.",
        candidates=[
            Candidate("distributed_4x1", {}, {"layout": "distributed"}),
            Candidate("pooled_1x4", {"stations.pack": _packs([("A2_00", 4)])},
                      {"layout": "pooled"}),
        ],
        score=_min_p95, metric_name="p95_order_latency_min",
        rationale="Pack-side pooling effect: M/M/4 at the pack beats 4x M/M/1, even though "
                  "distributing looks more local. Tuned to the subtle regime so distributed is "
                  "stable-but-worse, not an obvious collapse."),
}
