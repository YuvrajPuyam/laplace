"""Run the eval ablation on the dev scenarios and assemble the results table
(spec §5.4). `make eval`.

CPU arms (grid_search) run here against the cached GT (eval/gt_sweep). The Max
arms (llm_alone, agent) need the Claude Max window and are run separately — they
raise NotImplementedError on the CPU path; add them once wired (see
docs/PROJECT_STATE.md WS4 TODO). The held-out HEADLINE table additionally needs
Yuv's eval/scenarios/ traps; this dev-scenario table is the methodology demo.

  python -m eval.run_eval                       # all dev scenarios, grid arm
  python -m eval.run_eval --scenario braess_dev --budget 40
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval import baselines, metrics
from eval.candidates import DEV_DECISIONS
from eval.gt_sweep import ALL_DECISIONS, load_or_compute
from eval.held_scenarios import HELD_DECISIONS

RESULTS_DIR = Path("eval/results")
SUITES = {"dev": DEV_DECISIONS, "held": HELD_DECISIONS, "all": ALL_DECISIONS}

# arm name -> callable(decision, budget, gt) -> ArmDecision.
# Every arm is handed the SAME budget and the loaded GT (its seed list + candidate
# hashes), so the comparison is equal-budget + CRN-paired against the ground truth
# it is graded against (spec §5.4 H2; baselines.fair_seed_prefix).
ARMS = {  # CPU-only, no LLM budget — the default for `make eval`
    "grid_search": lambda d, budget, gt:
        baselines.grid_search(d, budget=budget, gt_seeds=gt["seeds"]),
    "ocba": lambda d, budget, gt: baselines.ocba(d, budget=budget, gt=gt),
}
# The Max arms (spend the Claude Max window). Only run with --live so `make eval`
# never spends budget by accident.
MAX_ARMS = {
    "llm_alone": lambda d, budget, gt: baselines.llm_alone(d, budget=budget, gt=gt),
    "agent": lambda d, budget, gt: baselines.agent(d, budget=budget, gt=gt),
}
CPU_ARMS = ARMS  # back-compat alias


def run(scenarios: list[str], seeds: int, budget: int,
        arms: dict | None = None) -> tuple[dict, dict]:
    arms = arms if arms is not None else ARMS
    graded: dict[str, list[dict]] = defaultdict(list)
    seed_list = list(range(seeds))
    for sid in scenarios:
        decision = ALL_DECISIONS[sid]
        gt = load_or_compute(decision, seed_list)        # cached; computes if missing
        for arm_name, fn in arms.items():
            try:
                ad = fn(decision, budget, gt)
            except Exception as e:  # noqa: BLE001 — one arm failing != lose the table
                ad = baselines.ArmDecision(
                    arm=arm_name, scenario_id=sid, picked=None, metric_estimate=None,
                    ci90=None, rollouts_used=0, notes=f"arm crashed: {type(e).__name__}: {e}")
            graded[arm_name].append(metrics.grade(ad, gt))
    table = {arm: metrics.aggregate(g) for arm, g in graded.items()}
    return table, dict(graded)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_eval")
    ap.add_argument("--scenario", default=None, choices=sorted(ALL_DECISIONS))
    ap.add_argument("--suite", default="dev", choices=tuple(SUITES),
                    help="which decision set to run when --scenario is omitted")
    ap.add_argument("--seeds", type=int, default=24, help="GT seeds")
    ap.add_argument("--budget", type=int, default=200, help="arm rollout budget")
    ap.add_argument("--live", action="store_true",
                    help="ALSO run the Max arms (llm_alone, agent) — spends the "
                         "Claude Max window. Default: grid_search only (CPU).")
    args = ap.parse_args(argv)

    arms = {**ARMS, **MAX_ARMS} if args.live else ARMS
    scenarios = [args.scenario] if args.scenario else sorted(SUITES[args.suite])
    table, graded = run(scenarios, args.seeds, args.budget, arms=arms)

    print(f"\nEval over {len(scenarios)} {args.suite} scenario(s), budget={args.budget}:")
    print(f"  {'arm':<14}{'accuracy':>10}{'regret':>9}{'rollouts':>10}"
          f"{'ci_cov':>9}")
    for arm, m in table.items():
        cov = "-" if m["ci_coverage"] is None else f"{m['ci_coverage']:.2f}"
        print(f"  {arm:<14}{m['decision_accuracy']:>10.2f}{m['mean_regret']:>9.3f}"
              f"{m['total_rollouts']:>10}{cov:>9}")
    print("  (llm_alone / agent arms: pending Max window -- WS4 TODO)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.suite}_table.json"
    out.write_text(json.dumps(
        {"scenarios": scenarios, "budget": args.budget, "seeds": args.seeds,
         "table": table, "graded": graded}, indent=2), encoding="utf-8")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
