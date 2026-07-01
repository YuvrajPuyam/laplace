"""Turn the raw ablation booleans into a DEFENSIBLE calibration + power story.

Reads the canonical dev graded decisions (eval/results/dev_table.json), and for each arm that
states a CI (llm_alone, agent) reports:
  - CI coverage (stated 90% CI contains the picked config's true metric) with a Wilson 90% CI,
  - decision accuracy with a Wilson 90% CI,
  - the power numbers: how many scenarios n would be needed to make the observed gaps
    statistically significant (eval/analysis/power.py).

The honest headline: at n=4 the "1.00 vs 0.75" gaps are DIRECTIONAL, not significant — the
Wilson intervals overlap heavily and the required n is ~30-40. Writes a markdown report and a
coverage figure (the seed of the paper's reliability diagram). No fabricated numbers: every
input is a harness-graded boolean from the GT sweep.

  python -m eval.analysis.make_calibration_report
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from eval.analysis.power import n_for_coverage_test, n_from_accuracy_gap

NOMINAL = 0.90
Z90 = 1.6448536269514722   # two-sided 90% normal quantile
RESULTS = Path("eval/results")


def wilson(k: int, n: int, z: float = Z90) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion (valid near 0/1, where calibration lives)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _suite_graded(*files: str) -> dict[str, list[dict]]:
    """Per-decision graded records pooled across the given suite files (skips missing)."""
    pooled: dict[str, list[dict]] = {}
    for fn in files:
        p = RESULTS / fn
        if not p.exists():
            continue
        g = json.loads(p.read_text(encoding="utf-8")).get("graded", {})
        for arm, decs in g.items():
            pooled.setdefault(arm, []).extend(decs)
    return pooled


def _load_graded() -> dict[str, list[dict]]:
    """The cited headline is the DEV suite (the '1.00 vs 0.75' the advisor referenced)."""
    return _suite_graded("dev_table.json")


def build() -> dict:
    graded = _load_graded()
    arms = {}
    for arm, decs in graded.items():
        cov = [g["ci_covered"] for g in decs if g.get("ci_covered") is not None]
        acc = [bool(g["correct"]) for g in decs]
        miss_cov = [g["scenario_id"] for g in decs if g.get("ci_covered") is False]
        miss_acc = [g["scenario_id"] for g in decs if not g["correct"]]
        rec = {"n": len(decs), "states_ci": len(cov) > 0,
               "acc": wilson(sum(acc), len(acc)), "acc_misses": miss_acc}
        if cov:
            rec["cov"] = wilson(sum(cov), len(cov))
            rec["cov_misses"] = miss_cov
        arms[arm] = rec

    # power: how many scenarios to make the observed gaps significant?
    a, l = arms["agent"], arms["llm_alone"]
    cov_margin = round(NOMINAL - l["cov"][0], 4)               # detect llm coverage vs nominal 0.90
    n_cov = n_for_coverage_test(cov_margin) if cov_margin > 0 else None
    acc_gap = round(a["acc"][0] - l["acc"][0], 4)
    # discordance = scenarios where exactly one of {agent,llm} is correct (paired/McNemar)
    decs_a = {g["scenario_id"]: bool(g["correct"]) for g in graded["agent"]}
    decs_l = {g["scenario_id"]: bool(g["correct"]) for g in graded["llm_alone"]}
    p_disc = sum(decs_a[s] != decs_l[s] for s in decs_a) / len(decs_a)
    n_acc = n_from_accuracy_gap(a["acc"][0], l["acc"][0], p_disc) if acc_gap > 0 and p_disc > 0 else None
    return {"arms": arms, "n_cov_needed": n_cov, "cov_margin": cov_margin,
            "n_acc_needed": n_acc, "acc_gap": acc_gap, "p_disc": round(p_disc, 4)}


def _fmt(t: tuple[float, float, float]) -> str:
    p, lo, hi = t
    return f"{p:.2f}  (90% Wilson [{lo:.2f}, {hi:.2f}])"


def _pooled_block() -> list[str]:
    """Strengthening view: dev + held pooled (only rendered once the held agent arm exists).
    Kept SEPARATE from the cited dev '1.00 vs 0.75' so we don't silently restate the headline."""
    g = _suite_graded("dev_table.json", "held_table.json")
    if not g.get("agent") or not g.get("llm_alone"):
        return []   # held agent arm not present yet (held not run, or agent arm absent)
    def cov(arm):
        b = [x["ci_covered"] for x in g[arm] if x.get("ci_covered") is not None]
        return wilson(sum(b), len(b)), len(b)
    def acc(arm):
        b = [bool(x["correct"]) for x in g[arm]]
        return wilson(sum(b), len(b)), len(b)
    (ac, an), (lc, ln) = cov("agent"), cov("llm_alone")
    (aa, _), (la, _) = acc("agent"), acc("llm_alone")
    # Honest, data-driven verdict (the held-out suite is allowed to NOT confirm the dev story).
    agent_better = ac[0] > lc[0] + 1e-9
    verdict = ("the dev calibration advantage REPLICATES (agent coverage above LLM-alone)."
               if agent_better else
               "the dev calibration advantage does NOT replicate here — pooled, the agent's CI "
               "coverage is at or below LLM-alone. This is the held-out suite doing its job; "
               "treat the dev 1.00-vs-0.75 as a hypothesis the held data has not yet confirmed.")
    return [
        "",
        f"## Held-out check — dev + held pooled (n={an})",
        "",
        "Now that the agent arm runs on the held scenarios, pooling them in (more scenarios = less",
        "underpowered). The cited dev headline above is unchanged; this is the harder, honest view.",
        "",
        "| arm | decision accuracy | CI coverage (stated 90%) |",
        "|-----|------------------:|-------------------------:|",
        f"| **Laplace agent** | {_fmt(aa)} | {_fmt(ac)} |",
        f"| **LLM-alone** | {_fmt(la)} | {_fmt(lc)} |",
        "",
        f"Pooled n={an}. Verdict: {verdict}",
        "",
        "Confound status: (1) the earlier `pool_pickzone` ABSTAIN was a budget cap — RESOLVED by "
        "raising tool_calls/max_turns; the agent now decides all 3 held correctly (accuracy 1.00). "
        "(2) n is still tiny (held n=3). (3) The CI miss is now characterized and looks REAL, not a "
        "units bug: the agent's stated 90% CIs are narrow AND its point estimates sit ~10-15% off the "
        "GT (braess_holdout truth 122.09 vs CI [123.18,125.13]; pool_packzone truth 5.18 vs CI "
        "[5.83,6.10]; pool_pickzone truth 7.61 vs CI [6.65,7.13]) — i.e. OVERCONFIDENT. A remaining "
        "fairness check (score the CI against the metric on the agent's OWN seeds, not the 12-seed GT "
        "mean) would separate 'overconfident' from 'sub-sample vs population', but the narrow-CI + "
        "biased-estimate pattern already points to genuine overconfidence on held.",
    ]


def write_markdown(r: dict) -> Path:
    a, l = r["arms"]["agent"], r["arms"]["llm_alone"]
    out = RESULTS / "calibration_report.md"
    lines = [
        "# Calibration + power — defending \"1.00 vs 0.75\" (dev suite, n=4)",
        "",
        "Computed from harness-graded GT-sweep decisions (`eval/results/dev_table.json`); every input",
        "is a boolean the harness assigned against the ground-truth optimum, never an agent self-report.",
        "",
        "## The two metrics that both read 1.00 vs 0.75",
        "",
        "| arm | decision accuracy | CI coverage (stated 90%) |",
        "|-----|------------------:|-------------------------:|",
        f"| **Laplace agent** | {_fmt(a['acc'])} | {_fmt(a['cov'])} |",
        f"| **LLM-alone** | {_fmt(l['acc'])} | {_fmt(l['cov'])} |",
        "",
        "- **Decision accuracy** = fraction of decisions matching the GT optimum.",
        "- **CI coverage** = fraction where the picked config's TRUE metric falls in the arm's stated",
        "  90% CI (target 0.90). The agent runs the experiment, so it is both correct AND calibrated;",
        "  the bare LLM, on the one scenario its prior misleads it, is wrong AND over-confident.",
        "",
        "## The honest caveat — concede this first",
        "",
        "- **n = 4.** Each gap is literally one scenario: 3/4 vs 4/4.",
        f"- The Wilson 90% intervals **overlap heavily** (agent coverage {a['cov'][1]:.2f}-{a['cov'][2]:.2f} vs",
        f"  LLM {l['cov'][1]:.2f}-{l['cov'][2]:.2f}) — so \"1.00 vs 0.75\" is **not statistically significant** at n=4.",
        f"- **Power:** to detect a coverage miss of {r['cov_margin']:.2f} below nominal 0.90 at 80% power needs",
        f"  **n ≈ {r['n_cov_needed']}** scenarios; to detect the accuracy gap (paired/McNemar, discordance",
        f"  {r['p_disc']:.2f}) needs **n ≈ {r['n_acc_needed']}**. We have 4. So this is a *directional dev-set",
        "  signal*, not a powered headline result.",
        "",
        "## A nuance worth stating (it makes the story more honest, not less)",
        "",
        f"- The LLM's **accuracy** miss is on `{', '.join(l['acc_misses']) or '—'}` (wrong call where intuition",
        "  misleads — the Braess case).",
        f"- Its **calibration** miss is on a *different* scenario, `{', '.join(l.get('cov_misses', [])) or '—'}`",
        "  (right call, but an over-confident interval that missed the truth). Two distinct failure modes,",
        "  both of which running-the-experiment fixes.",
        "",
        "## What makes it defensible (the plan, not a claim)",
        "",
        "1. Run the held-suite agent + LLM arms (adds scenarios under the integrity property).",
        "2. Add a second (manufacturing) domain → push n toward the ~30-40 the power analysis requires.",
        "3. Report coverage as a reliability diagram with Wilson/bootstrap CIs + ECE (figure below is the seed).",
        "4. Add the Gupta label-permutation responsiveness ablation (`eval/analysis/responsiveness.py`):",
        "   show the agent's decisions track the simulated outcomes (not ranking-and-selection in a costume).",
        "   *Status: framework ready; needs a dedicated permuted-feedback run — not yet computed (not faked).*",
        "",
        "## One-line defense",
        "",
        "> On the dev set the agent is both more accurate and better calibrated (1.00 vs 0.75 on each),",
        "> but the load-bearing claim is **calibration-as-mechanism**; it is underpowered at n=4 (needs",
        f"> ~{max(r['n_cov_needed'], r['n_acc_needed'])} scenarios) and becomes a real result once the held suite + a 2nd domain",
        "> are run, reported as a reliability diagram with CIs — not a 4-point ratio.",
        "",
        "![CI coverage with Wilson 90% intervals](reliability_diagram.svg)",
        "",
        "*Figure: stated-90%-CI empirical coverage per arm (dev suite, n=4) with Wilson 90% error bars.*",
        "*The bars span most of [0,1] — the visual statement that n=4 cannot distinguish these yet.*",
    ]
    lines.extend(_pooled_block())   # appends only if the held agent arm has been run
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_figure(r: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [("Laplace agent", r["arms"]["agent"]["cov"], "#6fb2d6"),
            ("LLM-alone", r["arms"]["llm_alone"]["cov"], "#d2705f")]
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=140)
    fig.patch.set_facecolor("#0c1118"); ax.set_facecolor("#0c1118")
    ax.axhline(NOMINAL, color="#76828f", ls="--", lw=1, label="nominal 0.90")
    for i, (name, (p, lo, hi), col) in enumerate(arms):
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", color=col, ms=9,
                    capsize=6, lw=2, mec="white", mew=0.6)
        ax.annotate(f"{p:.2f}", (i, p), textcoords="offset points", xytext=(12, -3),
                    color=col, fontsize=11, fontweight="bold")
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 1]); ax.set_xticklabels([a[0] for a in arms], color="#d6dee6", fontsize=10)
    ax.set_ylabel("empirical CI coverage", color="#d6dee6", fontsize=10)
    ax.set_title("Stated 90% CI coverage  ·  dev suite (n=4)", color="#d6dee6", fontsize=11)
    for s in ax.spines.values():
        s.set_color("#1d2730")
    ax.tick_params(colors="#76828f")
    leg = ax.legend(loc="lower right", fontsize=8, facecolor="#0c1118", edgecolor="#1d2730")
    for t in leg.get_texts():
        t.set_color("#d6dee6")
    fig.tight_layout()
    out = RESULTS / "reliability_diagram.svg"
    fig.savefig(out, facecolor=fig.get_facecolor()); plt.close(fig)
    return out


def main() -> int:
    r = build()
    md = write_markdown(r)
    fig = write_figure(r)
    a, l = r["arms"]["agent"], r["arms"]["llm_alone"]
    print("CALIBRATION + POWER REPORT")
    print(f"  agent   : acc {a['acc'][0]:.2f} {a['acc'][1:]}  cov {a['cov'][0]:.2f} {a['cov'][1:]}")
    print(f"  llm_alone: acc {l['acc'][0]:.2f} {l['acc'][1:]}  cov {l['cov'][0]:.2f} {l['cov'][1:]}")
    print(f"  power: n_cov~{r['n_cov_needed']} (margin {r['cov_margin']}), "
          f"n_acc~{r['n_acc_needed']} (gap {r['acc_gap']}, p_disc {r['p_disc']})")
    print(f"  wrote {md} + {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
