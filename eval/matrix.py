"""Statistically-defensible ablation matrix (spec §5.4) + budget sweep.

A single run of `run_eval --live` is n=1 — and the LLM arms are STOCHASTIC (the
model samples differently each time), so one run can be luck. This runs the LLM
arms over PARAPHRASES (re-worded questions, to show the agent isn't keying off
exact phrasing) × REPEATS (independent runs) at one or more BUDGETS, and reports
accuracy as **mean ± sd** across the passes. The deterministic arms (grid_search,
ocba) run once per (scenario, budget) — repeating them is pointless.

  python -m eval.matrix                      # default 3 paraphrases × 3 repeats, budget 60
  python -m eval.matrix --paraphrases 3 --repeats 3 --budgets 20 60   # + budget sweep (#2)
  python -m eval.matrix --scenario braess_dev --repeats 2 --budgets 12   # cheap validation

Each suite-pass is saved incrementally to eval/results/matrix.json so a mid-run
stall (each agent episode is capped by baselines.agent's wall-clock timeout) never
loses completed work.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from collections import defaultdict
from pathlib import Path

from eval import baselines, metrics
from eval.candidates import DEV_DECISIONS
from eval.gt_sweep import load_or_compute
from eval.run_eval import ARMS, MAX_ARMS

RESULTS = Path("eval/results/matrix.json")

# Re-worded questions per scenario (paraphrase[0] == the canonical question).
# All phrasings preserve the SAME decision + (for fleet) the robust-SLA meaning.
PARAPHRASES: dict[str, list[str]] = {
    "braess_dev": [
        "Should we open a mid cross-aisle (a shortcut connecting all aisles at the "
        "midpoint) to improve throughput?",
        "Would adding a mid-point cross-aisle that links all the aisles raise overall "
        "throughput, or hurt it?",
        "Is cutting a central cross-aisle shortcut across the aisles worth it for "
        "throughput, or counterproductive?",
    ],
    "dc_pickzone_med": [
        "How many AMRs does this pick zone need to keep p95 order latency under 10 "
        "minutes at peak demand RELIABLY — on essentially every run, not just on "
        "average? Pick the smallest fleet that robustly meets the SLA.",
        "What is the fewest AMRs that dependably holds 95th-percentile order latency "
        "below 10 minutes at peak demand — on essentially every run, not merely in "
        "expectation?",
        "We need p95 order latency under 10 min at peak demand, met on essentially "
        "every run (not just on average). What is the smallest adequate AMR fleet?",
    ],
    "mfc_compact": [
        "How many AMRs does this micro-fulfilment zone need to keep p95 order latency "
        "under 12 minutes RELIABLY — on essentially every run, not just on average? "
        "Pick the smallest fleet that robustly meets the SLA.",
        "What is the minimum AMR fleet that dependably keeps 95th-percentile latency "
        "below 12 minutes here — on essentially every run, not just in expectation?",
        "We want p95 latency under 12 min met on essentially every run (not just on "
        "average) in this compact zone. Fewest AMRs that does it?",
    ],
    "real_full_warehouse": [
        "How many AMRs does the scanned warehouse pick zone need to keep p95 order "
        "latency under 10 minutes RELIABLY — on essentially every run, not just on "
        "average? Pick the smallest fleet that robustly meets the SLA.",
        "On this scanned warehouse footprint, what is the fewest AMRs that dependably "
        "holds p95 order latency under 10 minutes — on essentially every run?",
        "For the scanned warehouse, smallest AMR fleet that keeps p95 latency below 10 "
        "min on essentially every run (not merely on average)?",
    ],
}

DETERMINISTIC = set(ARMS)            # grid_search, ocba — run once
STOCHASTIC = set(MAX_ARMS)           # llm_alone, agent — run over the matrix


def _para(sid: str, i: int) -> str:
    ps = PARAPHRASES.get(sid) or [DEV_DECISIONS[sid].question]
    return ps[i % len(ps)]


def _grade_arm(arm_name: str, fn, decision, budget: int, gt: dict) -> dict:
    try:
        ad = fn(decision, budget, gt)
    except Exception as e:  # noqa: BLE001 — a stalled/failed arm is an abstention
        ad = baselines.ArmDecision(arm=arm_name, scenario_id=decision.scenario_id,
                                   picked=None, metric_estimate=None, ci90=None,
                                   rollouts_used=0, notes=f"crashed: {type(e).__name__}: {e}")
    return metrics.grade(ad, gt)


def run_matrix(scenarios: list[str], budgets: list[int], n_paraphrase: int,
               n_repeat: int, seeds: int = 24) -> dict:
    seed_list = list(range(seeds))
    gts = {sid: load_or_compute(DEV_DECISIONS[sid], seed_list) for sid in scenarios}
    out: dict = {"scenarios": scenarios, "budgets": budgets,
                 "paraphrases": n_paraphrase, "repeats": n_repeat, "by_budget": {}}

    for budget in budgets:
        # deterministic arms: one pass over scenarios
        det = {a: [_grade_arm(a, ARMS[a], DEV_DECISIONS[sid], budget, gts[sid])
                   for sid in scenarios] for a in DETERMINISTIC}
        # stochastic arms: n_paraphrase x n_repeat passes over scenarios
        passes: dict[str, list[dict]] = {a: [] for a in STOCHASTIC}  # per-pass accuracy/regret
        for p in range(n_paraphrase):
            for r in range(n_repeat):
                pass_graded: dict[str, list[dict]] = defaultdict(list)
                for sid in scenarios:
                    d = dataclasses.replace(DEV_DECISIONS[sid], question=_para(sid, p))
                    for a in STOCHASTIC:
                        pass_graded[a].append(_grade_arm(a, MAX_ARMS[a], d, budget, gts[sid]))
                for a in STOCHASTIC:
                    g = pass_graded[a]
                    passes[a].append({
                        "paraphrase": p, "repeat": r,
                        "accuracy": sum(x["correct"] for x in g) / len(g),
                        "mean_regret": sum(x["regret"] for x in g) / len(g),
                        "graded": g,
                    })
                _save({**out, "by_budget": {**out["by_budget"],
                       str(budget): {"deterministic": det, "stochastic_passes": passes,
                                     "partial": True}}})
        out["by_budget"][str(budget)] = {
            "deterministic": {a: _agg_det(det[a]) for a in det},
            "stochastic": {a: _agg_stoch(passes[a]) for a in passes},
            "stochastic_passes": passes,
        }
        _save(out)
    return out


def _agg_det(graded: list[dict]) -> dict:
    return {"accuracy": round(sum(g["correct"] for g in graded) / len(graded), 4),
            "mean_regret": round(sum(g["regret"] for g in graded) / len(graded), 4),
            "n_scenarios": len(graded)}


def _agg_stoch(passes: list[dict]) -> dict:
    accs = [p["accuracy"] for p in passes]
    regs = [p["mean_regret"] for p in passes]
    cov = [g["ci_covered"] for p in passes for g in p["graded"] if g["ci_covered"] is not None]
    sd = statistics.pstdev(accs) if len(accs) > 1 else 0.0
    return {"accuracy_mean": round(statistics.fmean(accs), 4),
            "accuracy_sd": round(sd, 4),
            "regret_mean": round(statistics.fmean(regs), 4),
            "ci_coverage": round(sum(cov) / len(cov), 4) if cov else None,
            "n_passes": len(passes)}


def _save(obj: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ── resilient per-pass runner (each pass = its own short-lived process, so memory
#    is freed between passes and an OOM/crash never kills the whole matrix) ───────
PASSES = Path("eval/results/matrix_passes.jsonl")


def run_one_pass(scenarios: list[str], budget: int, paraphrase: int,
                 seeds: int = 24) -> dict:
    """One independent pass over all scenarios for the stochastic arms (agent,
    llm_alone) at one paraphrase. Returns per-arm per-scenario graded results."""
    gts = {sid: load_or_compute(DEV_DECISIONS[sid], list(range(seeds))) for sid in scenarios}
    arms: dict = {}
    for a in STOCHASTIC:
        graded = [_grade_arm(a, MAX_ARMS[a],
                             dataclasses.replace(DEV_DECISIONS[sid], question=_para(sid, paraphrase)),
                             budget, gts[sid]) for sid in scenarios]
        arms[a] = {"accuracy": sum(g["correct"] for g in graded) / len(graded),
                   "mean_regret": sum(g["regret"] for g in graded) / len(graded),
                   "graded": graded}
    return {"paraphrase": paraphrase, "budget": budget, "arms": arms}


def aggregate_passes() -> dict:
    """Aggregate all completed passes (matrix_passes.jsonl) into mean +/- sd, and
    fold in the deterministic arms (run once, here)."""
    rows = [json.loads(l) for l in PASSES.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if PASSES.exists() else []
    scen = sorted({s for r in rows for a in r["arms"].values() for g in a["graded"]
                   for s in [g["scenario_id"]]}) or sorted(DEV_DECISIONS)
    out = {"n_passes": len(rows), "scenarios": scen, "stochastic": {}, "deterministic": {}}
    for a in STOCHASTIC:
        accs = [r["arms"][a]["accuracy"] for r in rows if a in r["arms"]]
        regs = [r["arms"][a]["mean_regret"] for r in rows if a in r["arms"]]
        cov = [g["ci_covered"] for r in rows if a in r["arms"]
               for g in r["arms"][a]["graded"] if g["ci_covered"] is not None]
        if accs:
            out["stochastic"][a] = {
                "accuracy_mean": round(statistics.fmean(accs), 4),
                "accuracy_sd": round(statistics.pstdev(accs) if len(accs) > 1 else 0.0, 4),
                "regret_mean": round(statistics.fmean(regs), 4),
                "ci_coverage": round(sum(cov) / len(cov), 4) if cov else None,
                "n_passes": len(accs)}
    budget = rows[0]["budget"] if rows else 60
    gts = {sid: load_or_compute(DEV_DECISIONS[sid], list(range(24))) for sid in scen}
    for a in DETERMINISTIC:
        det = [_grade_arm(a, ARMS[a], DEV_DECISIONS[sid], budget, gts[sid]) for sid in scen]
        out["deterministic"][a] = _agg_det(det)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="matrix")
    ap.add_argument("--scenario", default=None, choices=sorted(DEV_DECISIONS))
    ap.add_argument("--budgets", type=int, nargs="+", default=[60])
    ap.add_argument("--paraphrases", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--pass-index", type=int, default=None,
                    help="run ONE pass (its own process) and append to matrix_passes.jsonl")
    ap.add_argument("--budget", type=int, default=60, help="budget for --pass-index")
    ap.add_argument("--aggregate", action="store_true",
                    help="aggregate matrix_passes.jsonl into mean +/- sd")
    args = ap.parse_args(argv)

    scenarios = [args.scenario] if args.scenario else sorted(DEV_DECISIONS)

    if args.pass_index is not None:                  # resilient single-pass mode
        rec = run_one_pass(scenarios, args.budget, args.pass_index % args.paraphrases, args.seeds)
        rec["pass_index"] = args.pass_index
        PASSES.parent.mkdir(parents=True, exist_ok=True)
        with open(PASSES, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        a = rec["arms"]
        print(f"pass {args.pass_index} (paraphrase {rec['paraphrase']}): "
              + " ".join(f"{k}={v['accuracy']:.2f}" for k, v in a.items()))
        return 0

    if args.aggregate:
        agg = aggregate_passes()
        print(f"\nMATRIX ({agg['n_passes']} passes, budget {args.budget}):")
        print(f"  {'arm':<12}{'accuracy':>16}{'regret':>9}{'ci_cov':>9}")
        for a, m in agg["deterministic"].items():
            print(f"  {a:<12}{m['accuracy']:>16.2f}{m['mean_regret']:>9.3f}{'-':>9}")
        for a, m in agg["stochastic"].items():
            acc = f"{m['accuracy_mean']:.2f}+/-{m['accuracy_sd']:.2f}"
            cov = "-" if m["ci_coverage"] is None else f"{m['ci_coverage']:.2f}"
            print(f"  {a:<12}{acc:>16}{m['regret_mean']:>9.3f}{cov:>9}")
        _save({"aggregate": agg})
        return 0
    n_llm_runs = len(scenarios) * args.paraphrases * args.repeats * len(args.budgets) * len(STOCHASTIC)
    print(f"matrix: {scenarios} | budgets={args.budgets} | "
          f"{args.paraphrases} paraphrases x {args.repeats} repeats")
    print(f"  -> {n_llm_runs} stochastic-arm runs (each agent run spends Max + has a "
          f"wall-clock timeout)\n")

    res = run_matrix(scenarios, args.budgets, args.paraphrases, args.repeats, args.seeds)

    for budget, b in res["by_budget"].items():
        print(f"\n=== budget {budget} ===")
        print(f"  {'arm':<12}{'accuracy':>16}{'regret':>9}{'ci_cov':>9}")
        for a, m in b["deterministic"].items():
            print(f"  {a:<12}{m['accuracy']:>16.2f}{m['mean_regret']:>9.3f}{'-':>9}")
        for a, m in b["stochastic"].items():
            acc = f"{m['accuracy_mean']:.2f}+/-{m['accuracy_sd']:.2f}"
            cov = "-" if m["ci_coverage"] is None else f"{m['ci_coverage']:.2f}"
            print(f"  {a:<12}{acc:>16}{m['regret_mean']:>9.3f}{cov:>9}")
    print(f"\n  -> {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
