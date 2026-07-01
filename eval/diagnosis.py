"""Class B — diagnosis (spec §4.3): the agent isolates WHY a facility degraded.

Unlike Class A ("which option is best?"), there is NO brute-force grid for "why" —
so this is where the agent's experimental reasoning should *beat* grid, not just
tie it. The agent gets a healthy baseline (summonable) + the observed degraded
SYMPTOM (headline metrics only — not the internal fingerprint, so it must
experiment to disambiguate) + a candidate-cause set, and must reproduce each
hypothesis on the baseline to find which one matches.

diag_edge is on `corridor_med` (a forced-corridor layout where a single cross-aisle
is the only path), tuned so a slow cap-1 edge choke produces a 40% throughput drop
with an edge-occupancy fingerprint distinct from the service-variance and demand
causes (verified: experiments/diag_corridor.py). The planted cause is gated by Yuv.
"""

from __future__ import annotations

import dataclasses

from eval import baselines


@dataclasses.dataclass
class DiagnosisScenario:
    id: str
    baseline: str                 # healthy scenario_id the agent summons
    healthy: dict                 # healthy headline metrics (reference)
    symptom: dict                 # observed DEGRADED headline metrics
    candidates: list[dict]        # [{name, hint}] — the agent picks one
    correct: str                  # the planted cause name (Yuv-signed-off)


DIAGNOSES: dict[str, DiagnosisScenario] = {
    "diag_edge": DiagnosisScenario(
        id="diag_edge", baseline="corridor_med",
        # Headline metrics ALONE are under-determined (any cause can be magnitude-
        # tuned to match them). The type-discriminating FINGERPRINT — worst edge
        # occupancy vs worst station-wait — is what makes the cause identifiable: an
        # edge cause spikes edge occupancy with station-wait normal; a variance cause
        # does the opposite. (Verified: experiments/diag_corridor.py.)
        healthy={"throughput_orders_per_hr": 185, "p95_order_latency_min": 13,
                 "abandonment_pct": 4, "worst_edge_occupancy_pct": 5,
                 "worst_station_wait_p95_min": 2},
        symptom={"throughput_orders_per_hr": 111, "p95_order_latency_min": 111,
                 "abandonment_pct": 42, "worst_edge_occupancy_pct": 50,
                 "worst_station_wait_p95_min": 1},
        candidates=[
            {"name": "service_variance",
             "hint": "a pack station's service-time variance increased (higher sigma)"},
            {"name": "edge_capacity",
             "hint": "an aisle/cross-aisle edge's capacity and/or max speed dropped"},
            {"name": "demand", "hint": "the order arrival rate increased"}],
        correct="edge_capacity"),

    # TRAP diagnosis (the H1-for-"why" demo): the true cause is COUNTERINTUITIVE — a
    # narrow shortcut was ADDED (Braess), so throughput fell. A guesser won't suspect
    # "adding a shortcut" (intuition says connectivity helps) and reaches for the
    # obvious causes; the agent reproduces each and discovers the truth. Signature
    # verified discriminable (this session): shortcut -> edge-occupancy 2%->46% with
    # station-wait normal; demand -> throughput holds; station-slow -> throughput
    # crashes + station-wait spikes.
    "diag_braess": DiagnosisScenario(
        id="diag_braess", baseline="braess_dev",
        healthy={"throughput_orders_per_hr": 120, "p95_order_latency_min": 27,
                 "abandonment_pct": 10, "worst_edge_occupancy_pct": 2,
                 "worst_station_wait_p95_min": 2},
        symptom={"throughput_orders_per_hr": 107, "p95_order_latency_min": 50,
                 "abandonment_pct": 19, "worst_edge_occupancy_pct": 46,
                 "worst_station_wait_p95_min": 2},
        candidates=[
            {"name": "demand_increase", "hint": "the order arrival rate increased"},
            {"name": "station_slowdown",
             "hint": "a pack station's mean service time increased"},
            {"name": "shortcut_added",
             "hint": "a narrow, low-capacity cross-aisle shortcut was opened between "
                     "the pick aisle (A3) and the pack aisle (A4)"}],
        correct="shortcut_added"),
}


_METRIC_LABELS = {
    "throughput_orders_per_hr": "throughput (orders/hr)",
    "p95_order_latency_min": "p95 order latency (min)",
    "abandonment_pct": "order abandonment (%)",
    "worst_edge_occupancy_pct": "worst edge occupancy (%)",
    "worst_station_wait_p95_min": "worst station-wait p95 (min)",
}


def diagnosis_question(d: DiagnosisScenario) -> str:
    rows = "\n".join(
        f"    {_METRIC_LABELS.get(k, k):<32} {d.healthy[k]:>6}  ->  {d.symptom[k]:>6}"
        for k in d.symptom)
    cands = "\n".join(f"  ({i + 1}) {c['name']}: {c['hint']}"
                      for i, c in enumerate(d.candidates))
    names = ", ".join(c["name"] for c in d.candidates)
    return (
        f"The facility '{d.baseline}' is degraded. Diagnose WHY.\n\n"
        f"Observed metrics (healthy baseline -> degraded now):\n{rows}\n\n"
        f"Exactly ONE of these changed:\n{cands}\n\n"
        f"Identify which, by EXPERIMENT: reproduce each hypothesis on the "
        f"'{d.baseline}' baseline (propose the change, run paired rollouts) and check "
        f"which one reproduces the FULL observed signature above (note: any single "
        f"cause can be tuned to match throughput/latency alone — the edge-occupancy "
        f"and station-wait signals are what distinguish the cause TYPE). In your "
        f"report's recommendation, lead with the single cause name (one of: {names}).")


def map_cause(report: dict | None, d: DiagnosisScenario) -> str | None:
    """The cause the agent CONCLUDES — the candidate name it leads its recommendation
    with (agents typically name their pick first, then rule the others out, so a
    'unique mention' rule wrongly abstains). Falls back to a unique mention across
    recommendation+mechanism if the recommendation names none."""
    if not report:
        return None
    rec = str(report.get("recommendation", "")).lower()
    in_rec = [(rec.find(c["name"].lower()), c["name"])
              for c in d.candidates if c["name"].lower() in rec]
    if in_rec:
        return min(in_rec)[1]                        # earliest-named = the conclusion
    text = " ".join(str(report.get(k, "")) for k in ("recommendation", "mechanism")).lower()
    hits = {c["name"] for c in d.candidates if c["name"].lower() in text}
    return next(iter(hits)) if len(hits) == 1 else None


def grade(picked: str | None, d: DiagnosisScenario, *, arm: str,
          rollouts: int = 0, notes: str = "") -> dict:
    return {"arm": arm, "diagnosis_id": d.id, "picked": picked, "correct_cause": d.correct,
            "correct": picked == d.correct, "rollouts": rollouts, "notes": notes}


# ── arms ──────────────────────────────────────────────────────────────────────

def agent(d: DiagnosisScenario, budget: int = 60, *, run_episode=None,
          model: str | None = None) -> dict:
    """The Laplace agent: experiments on the baseline to isolate the cause."""
    run = run_episode or baselines._default_run_episode(model, 50, 720.0)
    ep = run(question=diagnosis_question(d), scenario_id=d.baseline,
             budgets={"rollouts": budget, "renders": 0, "tool_calls": 25}, seed_base=0)
    if not (getattr(ep, "accepted", False) and getattr(ep, "report", None)):
        return grade(None, d, arm="agent", rollouts=budget,
                     notes=f"abstained: {getattr(ep, 'error', None) or 'no report'}")
    picked = map_cause(ep.report, d)
    note = f"{getattr(ep, 'num_turns', 0)} turns"
    if getattr(ep, "cost_usd", None) is not None:
        note += f", ${ep.cost_usd:.2f}"
    return grade(picked, d, arm="agent", rollouts=budget, notes=note)


def llm_alone(d: DiagnosisScenario, *, answer_fn=None, model: str | None = None) -> dict:
    """Same symptom, NO simulator — must guess the cause. Quantifies H1 for diagnosis."""
    names = [c["name"] for c in d.candidates]
    prompt = (diagnosis_question(d) + "\n\nYou do NOT have a simulator — reason from the "
              "description alone. Respond with a single JSON object and nothing else: "
              f'{{"cause": "<one of: {", ".join(names)}>", "confidence": <0..1>}}')
    raw = answer_fn(prompt) if answer_fn else baselines._default_llm_answer(prompt, model)
    ans = baselines._parse_json_answer(raw)
    pick = ans.get("cause")
    picked = pick if pick in names else map_cause({"recommendation": str(pick)}, d)
    return grade(picked, d, arm="llm_alone", rollouts=0,
                 notes=f"no sim; confidence={ans.get('confidence')}")


def main(argv=None) -> int:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(prog="diagnosis")
    ap.add_argument("--diagnosis", default="diag_edge", choices=sorted(DIAGNOSES))
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--arms", nargs="+", default=["llm_alone", "agent"])
    args = ap.parse_args(argv)

    d = DIAGNOSES[args.diagnosis]
    print(f"Class B diagnosis: {d.id} (planted cause = {d.correct})\n")
    out = {"diagnosis": d.id, "correct_cause": d.correct, "results": {}}
    if "llm_alone" in args.arms:
        out["results"]["llm_alone"] = llm_alone(d)
    if "agent" in args.arms:
        out["results"]["agent"] = agent(d, budget=args.budget)
    for a, g in out["results"].items():
        mark = "CORRECT" if g["correct"] else "wrong"
        print(f"  {a:<10} picked={str(g['picked']):<18} [{mark}]  ({g['notes']})")
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path(f"eval/results/{d.id}.json").write_text(json.dumps(out, indent=2))
    print(f"\n  -> eval/results/{d.id}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
