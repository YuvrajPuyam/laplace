"""report_table — turn raw eval results into the README's headline artifact.

Pure PRESENTATION: computes no new numbers, only formats the per-arm aggregates and
per-scenario grades that run_eval already produced (eval/results/<suite>_table.json)
into (1) the three-way ablation table (accuracy / regret / calibration / cost) and
(2) the per-trap narrative (what each arm picked vs the GT optimum — the part that
makes a trap land). Re-runnable on cached results, so the table can be reformatted
without re-spending Claude budget.

  python -m eval.report_table --suite held      # -> eval/results/held_table.md (and stdout)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows cp1252 console
except Exception:  # noqa: BLE001
    pass

RESULTS_DIR = Path("eval/results")
GT_DIR = Path("eval/gt_cache")

ARM_ORDER = ["llm_alone", "agent", "grid_search", "ocba"]
ARM_LABEL = {"llm_alone": "LLM-alone (no sim)", "agent": "Laplace agent",
             "grid_search": "grid search", "ocba": "OCBA (strong OR)"}


def _fmt(x, nd=2, dash="—"):
    return dash if x is None else f"{x:.{nd}f}"


def _gt(sid: str) -> dict:
    p = GT_DIR / f"{sid}.gt.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def build(suite: str) -> str:
    data = json.loads((RESULTS_DIR / f"{suite}_table.json").read_text(encoding="utf-8"))
    table, graded = data["table"], data["graded"]
    scenarios = data["scenarios"]
    arms = [a for a in ARM_ORDER if a in table] + [a for a in table if a not in ARM_ORDER]

    L = []
    L.append(f"# Ablation — {suite} suite "
             f"({len(scenarios)} scenario{'s' if len(scenarios) != 1 else ''}, "
             f"budget {data['budget']}, {data['seeds']} GT seeds)\n")

    # (1) headline three-way table
    L.append("| arm | decision accuracy | mean regret | calibration (CI cov.) | cost (rollouts) |")
    L.append("|-----|------------------:|------------:|----------------------:|----------------:|")
    for a in arms:
        m = table[a]
        L.append(f"| {ARM_LABEL.get(a, a)} | {_fmt(m['decision_accuracy'])} | "
                 f"{_fmt(m['mean_regret'], 3)} | {_fmt(m.get('ci_coverage'))} | "
                 f"{m['total_rollouts']} |")
    L.append("")
    L.append("*Accuracy = fraction of decisions matching the GT optimum. Regret = normalized "
             "score gap (0 = optimum). Calibration = coverage of the arm's 90% CI (— = arm states "
             "no CI). Cost = sim rollouts used (LLM-alone uses none). All graded by the harness "
             "from GT sweeps — never from arm self-reports.*\n")

    # (2) per-trap narrative — pivot graded by scenario
    by_scen: dict[str, dict[str, dict]] = {s: {} for s in scenarios}
    for a in arms:
        for g in graded[a]:
            by_scen.setdefault(g["scenario_id"], {})[a] = g
    L.append("## Per-trap detail\n")
    for sid in scenarios:
        gt = _gt(sid)
        opt = next(iter(by_scen[sid].values()))["gt_optimum"] if by_scen[sid] else "?"
        L.append(f"### `{sid}` — GT optimum: **{opt}**")
        if gt.get("question"):
            L.append(f"> {gt['question']}")
        if gt.get("rationale"):
            L.append(f"\n*{gt['rationale']}*\n")
        L.append("| arm | picked | correct? | regret |")
        L.append("|-----|--------|:--------:|-------:|")
        for a in arms:
            g = by_scen[sid].get(a)
            if not g:
                continue
            mark = "✅" if g["correct"] else "❌"
            L.append(f"| {ARM_LABEL.get(a, a)} | {g['picked']} | {mark} | {_fmt(g['regret'], 3)} |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="report_table")
    ap.add_argument("--suite", default="held", choices=("dev", "held", "all"))
    args = ap.parse_args(argv)
    md = build(args.suite)
    out = RESULTS_DIR / f"{args.suite}_table.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
