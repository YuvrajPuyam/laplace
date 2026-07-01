"""Decision spaces for the DEV scenarios — the candidate set + the objective
that defines each scenario's ground-truth optimum.

This is the DEVELOPMENT set used to build and validate the eval harness. The
held-out trap suite in eval/scenarios/ is Yuv's to design (CLAUDE.md; spec §7
"you design the T3 traps") and is intentionally not here.

Each Decision pairs a natural-language question (what the agent is asked) with a
small discrete candidate set — each candidate a Contract A patch (dot-path,
tools.md) over the base scenario — and a `score(metrics, params) -> float`
objective (higher = better). GT optimum = the candidate with the best mean score
under common random numbers. Metrics come straight from the rollout result
(sim/metrics.py), so no number is invented here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

DEV_DIR = Path("eval/dev_scenarios")


@dataclasses.dataclass
class Candidate:
    label: str
    patch: dict             # dot-path patch over the base config (may be {})
    params: dict            # decision variables, for the objective + reporting


@dataclasses.dataclass
class Decision:
    scenario_id: str
    question: str
    candidates: list[Candidate]
    score: Callable[[dict, dict], float]   # (metrics, params) -> higher is better
    metric_name: str        # headline metric to surface in the table
    rationale: str          # why the optimum is what it is (demo narration)
    base_dir: Path = DEV_DIR   # held-out decisions point this at eval/scenarios

    def base_path(self) -> Path:
        return self.base_dir / f"{self.scenario_id}.config.json"


def base_config_path(scenario_id: str) -> Path:
    return DEV_DIR / f"{scenario_id}.config.json"


def _mid_cross_aisle(n_aisles: int, pos: int) -> list[dict]:
    """A full mid cross-aisle at `pos` across all aisles (the Braess lever)."""
    return [{"from": f"A{a}_{pos:02d}", "to": f"A{a + 1}_{pos:02d}"}
            for a in range(1, n_aisles)]


def _fleet_candidates(counts: tuple[int, ...]) -> list[Candidate]:
    return [Candidate(f"amr_{n}", {"fleet.amr_count": n}, {"amr_count": n})
            for n in counts]


def _smallest_fleet_meeting_p95(target_min: float) -> Callable[[dict, dict], float]:
    """SLA objective: any fleet whose p95 latency <= target beats any that
    doesn't; among those that meet it, fewer AMRs wins (cheapest adequate fleet);
    among those that miss, lower p95 wins. Encoded as one higher-is-better score."""
    def score(metrics: dict, params: dict) -> float:
        p95 = metrics["p95_order_latency_min"]
        amr = params["amr_count"]
        if p95 <= target_min:
            return 1e6 - amr            # meets SLA: prefer the smallest fleet
        return -p95                     # misses SLA: prefer the closest
    return score


def _max_throughput(metrics: dict, params: dict) -> float:
    return metrics["throughput_orders_per_hr"]


DEV_DECISIONS: dict[str, Decision] = {
    # Class A — layout lever (the Braess question; demo centerpiece family).
    "braess_dev": Decision(
        scenario_id="braess_dev",
        question="Should we open a mid cross-aisle (a shortcut connecting all "
                 "aisles at the midpoint) to improve throughput?",
        candidates=[
            Candidate("no_shortcut", {}, {"shortcut": False}),
            Candidate("mid_cross_aisle",
                      {"layout.extra_edges": _mid_cross_aisle(6, 15)},
                      {"shortcut": True}),
        ],
        score=_max_throughput, metric_name="throughput_orders_per_hr",
        rationale="Whether the extra connectivity raises throughput or (Braess) "
                  "congests the network is decided by the sim, not intuition."),

    # Class A — fleet sizing (the 'is the Nth AMR worth it?' family).
    "dc_pickzone_med": Decision(
        scenario_id="dc_pickzone_med",
        question="How many AMRs does this pick zone need to keep p95 order "
                 "latency under 10 minutes at peak demand RELIABLY — i.e. on "
                 "essentially every run, not merely on average? Pick the smallest "
                 "fleet that robustly meets the SLA.",
        candidates=_fleet_candidates((3, 4, 5, 6, 7)),
        score=_smallest_fleet_meeting_p95(10.0),
        metric_name="p95_order_latency_min",
        rationale="The optimum is the knee: the smallest fleet that still meets "
                  "the p95 SLA; more AMRs are wasted capital, fewer breach it."),

    "mfc_compact": Decision(
        scenario_id="mfc_compact",
        question="How many AMRs does this micro-fulfilment zone need to keep p95 "
                 "order latency under 12 minutes RELIABLY — on essentially every "
                 "run, not just on average? Pick the smallest fleet that robustly "
                 "meets the SLA.",
        candidates=_fleet_candidates((2, 3, 4, 5)),
        score=_smallest_fleet_meeting_p95(12.0),
        metric_name="p95_order_latency_min",
        rationale="Smallest fleet meeting the p95 SLA in a single-block zone."),

    # The scanned real-warehouse scenario, as a dev/demo arm (NOT held-out).
    "real_full_warehouse": Decision(
        scenario_id="real_full_warehouse",
        question="How many AMRs does the scanned warehouse pick zone need to "
                 "keep p95 order latency under 10 minutes RELIABLY — on essentially "
                 "every run, not just on average? Pick the smallest fleet that "
                 "robustly meets the SLA.",
        candidates=_fleet_candidates((4, 5, 6, 7, 8)),
        score=_smallest_fleet_meeting_p95(10.0),
        metric_name="p95_order_latency_min",
        rationale="Fleet knee on the real extracted footprint. NOTE: demand is "
                  "an uncalibrated placeholder — treat results as RELATIVE "
                  "(A-vs-B), not absolute, until demand is calibrated."),
}
