"""Eval baselines — the three-way ablation arms (spec §5.4).

Each arm answers a Decision's question by picking a candidate, and returns an
ArmDecision that eval/metrics.py grades against the ground-truth optimum
(eval/gt_sweep.py): decision accuracy, regret, calibration (stated CI vs GT).

- grid_search: uniformly allocate a rollout budget across the candidate set,
  run, pick the best mean score. No LLM, CPU-only. Quantifies H2 — does the
  agent's adaptive allocation + follow-up beat uniform spend at EQUAL budget?
- llm_alone / agent: the Max-window arms (need the Claude Max budget; not run on
  CPU). Interfaces are stubbed below and wired in the next session — see
  docs/PROJECT_STATE.md.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence

import numpy as np

from eval.candidates import Decision
from sim.config import apply_patch, fill_defaults, load_config
from sim.runner import run_many

# Dev GT seed count (eval/gt_sweep --seeds default). Only a fallback for direct
# grid_search() calls that don't pass the loaded GT's seed list; run_eval always
# threads the real gt["seeds"] through (eval/run_eval.ARMS), so the arm and the
# ground truth it is graded against share one seed set.
DEFAULT_GT_SEEDS = 24


@dataclasses.dataclass
class ArmDecision:
    arm: str                 # "grid_search" | "llm_alone" | "agent"
    scenario_id: str
    picked: str              # candidate label the arm recommends
    metric_estimate: float | None       # arm's estimate of that candidate's metric
    ci90: tuple[float, float] | None     # stated 90% CI (for calibration)
    rollouts_used: int
    notes: str
    seeds_used: list[int] | None = None  # exact CRN seeds the arm spent (provenance)
    measured_hash: str | None = None     # config_hash the report cited for its pick (identity)


def _candidate_config(base: dict, patch: dict) -> dict:
    return fill_defaults(apply_patch(base, patch)) if patch else fill_defaults(base)


def fair_seed_prefix(budget: int, n_candidates: int,
                     gt_seeds: Sequence[int] | None) -> list[int]:
    """The CRN seeds an equal-budget arm is allowed to spend PER candidate
    (spec §5.4 H2).

    Two fairness invariants, both required for the three-way ablation to mean
    anything (see docs/HANDOFF.md §3 "Eval fairness bug"):

    1. SHARED PREFIX. The seeds are a prefix of the ground-truth seed list, so
       every arm's comparison is CRN-paired with the GT it is graded against.
    2. NEVER OUT-RESOLVE THE TRUTH. The per-candidate seed count is capped at
       the number of GT seeds. An arm cannot 're-derive' the optimum by simply
       sampling more than the ground truth did — the bug this replaces gave
       grid_search ~100 seeds/candidate (range(0, budget//n)) while the GT used
       24, so brute force trivially matched GT on a seed-count artifact, not on
       strategy.
    """
    seeds = list(range(DEFAULT_GT_SEEDS)) if gt_seeds is None else list(gt_seeds)
    per = min(max(1, budget // n_candidates), len(seeds))
    return seeds[:per]


def grid_search(decision: Decision, budget: int = 200,
                gt_seeds: Sequence[int] | None = None) -> ArmDecision:
    """Equal-budget grid search (the honest H2 baseline): split the rollout
    budget uniformly across the candidate set over a shared GT seed prefix
    (CRN-paired), then pick the candidate with the best mean score.

    The arm draws its seeds via ``fair_seed_prefix`` so its information is a
    strict subset of the ground truth's — it cannot beat the agent by simply
    out-sampling the GT. ``gt_seeds`` is the loaded GT's ``seeds`` list."""
    n = len(decision.candidates)
    seeds = fair_seed_prefix(budget, n, gt_seeds)
    per = len(seeds)
    capped = per < max(1, budget // n)
    base = load_config(decision.base_path())

    best_label, best_score, best_metric = None, -np.inf, None
    for cand in decision.candidates:
        cfg = _candidate_config(base, cand.patch)
        results = run_many(cfg, seeds, write_log=False)
        score = float(np.mean(
            [decision.score(r["metrics"], cand.params) for r in results]))
        metric = float(np.mean(
            [r["metrics"][decision.metric_name] for r in results]))
        if score > best_score:
            best_label, best_score, best_metric = cand.label, score, metric

    note = f"uniform {per} seeds/candidate over {n} candidates"
    if capped:
        note += (f" (budget would allow {budget // n}/candidate; capped to the "
                 f"{per} GT seeds so it cannot out-resolve the ground truth)")
    return ArmDecision(
        arm="grid_search", scenario_id=decision.scenario_id, picked=best_label,
        metric_estimate=round(best_metric, 4), ci90=None, rollouts_used=per * n,
        notes=note, seeds_used=seeds)


# ── OCBA: the principled non-LLM allocator baseline (research-verified, R1) ────

def _ocba_alloc(means: Sequence[float], sds: Sequence[float],
                eps: float = 1e-9) -> list[float]:
    """OCBA target allocation proportions (Chen et al. 2000): normalized weights
    summing to 1. The best design b gets ``sd_b·sqrt(Σ_{i≠b}(w_i/sd_i)²)``; every
    other design gets ``w_i=(sd_i/δ_ib)²`` with ``δ_ib = mean_b − mean_i`` — so
    budget concentrates on 'critical' designs (high variance OR a small gap to the
    best). Pure + deterministic, so the allocation rule is unit-tested directly."""
    k = len(means)
    if k == 1:
        return [1.0]
    b = max(range(k), key=lambda i: means[i])
    s = [max(float(sd), eps) for sd in sds]
    w = [0.0] * k
    for i in range(k):
        if i != b:
            delta = max(means[b] - means[i], eps)
            w[i] = (s[i] / delta) ** 2
    w[b] = s[b] * sum((w[i] / s[i]) ** 2 for i in range(k) if i != b) ** 0.5
    tot = sum(w)
    return [1.0 / k] * k if tot <= 0 else [wi / tot for wi in w]


def ocba(decision: Decision, budget: int = 200, gt: dict | None = None, *,
         n0: int = 4) -> ArmDecision:
    """OCBA (Optimal Computing Budget Allocation, Chen et al. 2000) — the principled
    non-LLM allocator baseline (docs/RESEARCH_BACKLOG.md R1). Sequentially spends an
    EQUAL total budget (= grid_search's capped spend) across the candidates,
    concentrating CRN-paired rollouts on the 'critical' designs by means+variances,
    then picks the best mean score. This is the honest H2 yardstick: does the agent's
    adaptive allocation match an expert sim-optimization allocator?

    Fairness is identical to grid_search — seeds are a prefix of the GT list and
    per-candidate spend is capped at the GT resolution. NOTE: OCBA's PCS is an
    average-case target, not a worst-case guarantee (don't overclaim it)."""
    if gt is None:
        raise ValueError("ocba arm needs the cached GT dict (for grading + seeds)")
    n = len(decision.candidates)
    pool = list(gt["seeds"])
    max_per = len(pool)                                    # GT-resolution cap/candidate
    total = len(fair_seed_prefix(budget, n, pool)) * n     # equal budget = grid's spend
    base = load_config(decision.base_path())
    cfgs = [_candidate_config(base, c.patch) for c in decision.candidates]

    scores: list[list[float]] = [[] for _ in range(n)]
    metricv: list[list[float]] = [[] for _ in range(n)]
    spent = 0

    def run_to(i: int, target: int) -> None:
        nonlocal spent
        new = pool[len(scores[i]):min(target, max_per)]
        if not new:
            return
        for r in run_many(cfgs[i], new, write_log=False):
            scores[i].append(float(decision.score(r["metrics"], decision.candidates[i].params)))
            metricv[i].append(float(r["metrics"][decision.metric_name]))
        spent += len(new)

    start = min(max(2, n0), max(1, total // n), max_per)
    for i in range(n):
        if spent >= total:
            break
        run_to(i, start)

    chunk = max(1, total // 10)
    while spent < total and any(len(scores[i]) < max_per for i in range(n)):
        means = [float(np.mean(s)) for s in scores]
        sds = [float(np.std(s, ddof=1)) if len(s) > 1 else 0.0 for s in scores]
        props = _ocba_alloc(means, sds)
        cand = max((i for i in range(n) if len(scores[i]) < max_per),
                   key=lambda i: props[i] * total - len(scores[i]), default=None)
        if cand is None:
            break
        run_to(cand, len(scores[cand]) + min(chunk, total - spent,
                                             max_per - len(scores[cand])))

    means = [float(np.mean(s)) for s in scores]
    best = max(range(n), key=lambda i: means[i])
    return ArmDecision(
        arm="ocba", scenario_id=decision.scenario_id,
        picked=decision.candidates[best].label,
        metric_estimate=round(float(np.mean(metricv[best])), 4), ci90=None,
        rollouts_used=spent, seeds_used=pool[:max(len(s) for s in scores)],
        notes=f"OCBA: {[len(s) for s in scores]} seeds/candidate (n0={start})")


# ── report → discrete decision mapping (the testable core of the Max arms) ────
#
# Both Max arms ultimately produce an ArmDecision graded against the GT. The hard,
# fragile part is turning a free-form agent report (§6.4) into one of the discrete
# candidate labels. That mapping lives here as pure functions so it is exhaustively
# unit-testable with mock reports — the actual Max-spending execution (running the
# agent / calling the model) is a thin, injectable seam below.

def _describe(cand: "Candidate") -> str:
    p = cand.params
    if "amr_count" in p:
        return f"operate {p['amr_count']} AMRs"
    if "shortcut" in p:
        return ("add the mid cross-aisle shortcut" if p["shortcut"]
                else "keep the baseline layout (no shortcut)")
    return f"apply the patch {cand.patch}" if cand.patch else "keep the baseline (no change)"


def decision_space_text(decision: Decision) -> str:
    """The question handed to a Max arm: the open question plus the EXPLICIT
    discrete candidate set, with an instruction to name the chosen option by its
    exact label so the report maps cleanly back to a candidate (spec §5.1)."""
    opts = "\n".join(
        f"  - {c.label}: {_describe(c)}"
        + (f"\n      exact config patch: {json.dumps(c.patch)}" if c.patch
           else " (baseline; empty patch)")
        for c in decision.candidates)
    labels = ", ".join(c.label for c in decision.candidates)
    return (f"{decision.question}\n\nChoose exactly one of these options (each IS a precise "
            f"config patch over the baseline):\n{opts}\n\n"
            f"In your final report's recommendation, state the option you recommend "
            f"using its exact label (one of: {labels}).")


def _label_from_text(text: str, decision: Decision,
                     gt: dict, metric_hint: float | None) -> str | None:
    """Map free text (+ an optional metric estimate) to a candidate label:
    a unique exact-label mention wins; otherwise fall back to the candidate whose
    GT metric is closest to the arm's stated estimate; else abstain (None)."""
    import re

    labels = [c.label for c in decision.candidates]
    t = (text or "").lower()
    hits = {lab for lab in labels
            if re.search(r"\b" + re.escape(lab.lower()) + r"\b", t)}
    if len(hits) == 1:
        return next(iter(hits))
    # 0 hits or an ambiguous mention of several: let the stated metric estimate
    # break the tie; if there's none either, abstain.
    if isinstance(metric_hint, (int, float)):
        return min(gt["candidates"],
                   key=lambda c: abs(c["mean_metric"] - metric_hint))["label"]
    return None


def _as_dict(x) -> dict:
    """Coerce a report field to a dict. The runner returns the agent's RAW
    submit_report input, where nested objects (primary_metric, recommended) can
    arrive as JSON-encoded STRINGS (the same marshalling the engine coerces server
    side). Anything that isn't a dict / dict-JSON becomes {}."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
        except (json.JSONDecodeError, TypeError):
            return {}
        return v if isinstance(v, dict) else {}
    return {}


def _match_candidate(report: dict, decision: Decision, gt: dict) -> str | None:
    """Discrete pick from an agent report. Most reliable signal first: an exact
    recommended config_hash, then a unique label mention, then metric proximity."""
    hash_to_label = {c["config_hash"]: c["label"] for c in gt["candidates"]}
    rec = _as_dict(_as_dict(report.get("primary_metric")).get("recommended"))
    for key in ("config", "config_hash", "hash"):
        if rec.get(key) in hash_to_label:
            return hash_to_label[rec[key]]
    text = " ".join(str(report.get(k, "")) for k in ("recommendation", "mechanism"))
    mean = rec.get("mean")
    return _label_from_text(text, decision, gt,
                            mean if isinstance(mean, (int, float)) else None)


def _recommended_ci_and_estimate(report: dict) -> tuple[float | None, tuple[float, float] | None]:
    rec = _as_dict(_as_dict(report.get("primary_metric")).get("recommended"))
    mean = rec.get("mean")
    ci = rec.get("ci90")
    ci90 = tuple(ci) if isinstance(ci, (list, tuple)) and len(ci) == 2 else None
    estimate = float(mean) if isinstance(mean, (int, float)) else None
    return estimate, ci90


def report_to_arm_decision(decision: Decision, gt: dict, episode, *,
                           total_budget: int, allowed_seeds: Sequence[int],
                           arm: str = "agent") -> ArmDecision:
    """Turn a finished agent episode (EpisodeResult-shaped) into a graded
    ArmDecision. A missing/rejected report is an honest abstention, not a crash."""
    accepted = bool(getattr(episode, "accepted", False)) and getattr(episode, "report", None)
    if not accepted:
        why = getattr(episode, "error", None) or "no accepted report"
        return ArmDecision(
            arm=arm, scenario_id=decision.scenario_id, picked=None,
            metric_estimate=None, ci90=None, rollouts_used=total_budget,
            notes=f"abstained: {why}", seeds_used=list(allowed_seeds))

    report = episode.report
    picked = _match_candidate(report, decision, gt)
    estimate, ci90 = _recommended_ci_and_estimate(report)
    _rec = _as_dict(_as_dict(report.get("primary_metric")).get("recommended"))
    measured_hash = _rec.get("config") or _rec.get("config_hash") or _rec.get("hash")
    note = f"agent episode, {getattr(episode, 'num_turns', 0)} turns"
    cost = getattr(episode, "cost_usd", None)
    if cost is not None:
        note += f", ${cost:.2f}"
    if picked is None:
        note += "; recommendation did not map to a candidate"
    return ArmDecision(
        arm=arm, scenario_id=decision.scenario_id, picked=picked,
        metric_estimate=estimate, ci90=ci90, rollouts_used=total_budget,
        notes=note, seeds_used=list(allowed_seeds), measured_hash=measured_hash)


# ── the agent arm: existing ClaudeAgentRunner, equal budget, GT seed prefix ────

def _default_run_episode(model: str | None, max_turns: int, timeout_s: float):
    """Real episode executor (Max-spending). Reuses the proven ClaudeAgentRunner.
    Imported lazily so `import eval.baselines` never pulls the Agent SDK.

    A WALL-CLOCK timeout bounds a hung Agent-SDK call (observed: the SDK stalling
    after a rejected report blocks the whole eval). On timeout the coroutine is
    cancelled (the runner's finally terminates its engine subprocess) and the
    arm raises TimeoutError → run_eval grades it as an abstention, so one stuck
    episode never stalls the table."""
    def run(**kw):
        import asyncio

        from agent.runner import ClaudeAgentRunner
        runner = ClaudeAgentRunner(model=model, max_turns=max_turns)
        return asyncio.run(asyncio.wait_for(runner.run(**kw), timeout=timeout_s))
    return run


def agent(decision: Decision, budget: int = 200, gt: dict | None = None, *,
          run_episode=None, model: str | None = None,
          max_turns: int = 80, timeout_s: float = 900.0) -> ArmDecision:
    """The Laplace agent under an EQUAL rollout budget (spec §5.4 H2).

    The agent is given exactly grid_search's spend — ``len(fair_seed_prefix) * n``
    rollouts — and ``seed_base=0`` so its run_rollouts seeds are the SAME GT prefix
    (engine ``seed_for(i)=seed_base+i``), keeping the comparison CRN-paired and
    equal-budget rather than a seed-count artifact. ``run_episode`` is the
    injectable seam: tests pass a stub; the default spawns the real runner (Max)."""
    if gt is None:
        raise ValueError("agent arm needs the cached GT dict (for grading + seeds)")
    n = len(decision.candidates)
    allowed = fair_seed_prefix(budget, n, gt["seeds"])
    total = len(allowed) * n
    run = run_episode or _default_run_episode(model, max_turns, timeout_s)
    # Pass the scenario's base config inline so the engine can instantiate HELD-OUT scenarios
    # (which the public store deliberately does not serve) for grading — the harness loads it
    # programmatically; held configs never enter the public store / GET /health (CLAUDE.md #4).
    # Guarded: missing file or a test stub decision falls back to the store (config=None).
    cfg = None
    try:
        bp = decision.base_path()
        if bp.exists():
            cfg = load_config(str(bp))
    except Exception:  # noqa: BLE001 — never let config-loading break the arm; engine falls back
        cfg = None
    episode = run(question=decision_space_text(decision),
                  scenario_id=decision.scenario_id,
                  budgets={"rollouts": total, "renders": 0, "tool_calls": 50},
                  seed_base=0, config=cfg)
    return report_to_arm_decision(decision, gt, episode,
                                  total_budget=total, allowed_seeds=allowed, arm="agent")


# ── the llm-alone arm: same scene summary, NO sim tools (spec §5.4, H1) ────────

LLM_ALONE_SYSTEM = (
    "You are an operations-research expert estimating warehouse performance "
    "WITHOUT running any simulation. You get a scene description and a discrete set "
    "of options; give your single best recommendation and a calibrated 90% "
    "confidence interval for its primary metric. Output ONLY the requested JSON.")


def _llm_alone_prompt(decision: Decision, scene: str) -> str:
    labels = ", ".join(c.label for c in decision.candidates)
    return (f"{scene}\n\n{decision_space_text(decision)}\n\n"
            f"You do NOT have a simulator — reason from the scene alone. Respond with "
            f"a single JSON object and nothing else:\n"
            f'{{"pick": "<one of: {labels}>", '
            f'"metric_estimate": <your estimate of {decision.metric_name}>, '
            f'"ci90": [<low>, <high>], "confidence": <0..1>}}')


def _parse_json_answer(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    import re

    m = re.search(r"\{.*\}", str(raw), re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, TypeError):
        return {}


def _llm_answer_to_arm_decision(decision: Decision, gt: dict, answer: dict,
                                arm: str = "llm_alone") -> ArmDecision:
    labels = {c.label for c in decision.candidates}
    pick = answer.get("pick") or answer.get("recommendation")
    estimate = answer.get("metric_estimate")
    estimate = float(estimate) if isinstance(estimate, (int, float)) else None
    picked = pick if pick in labels else _label_from_text(str(pick), decision, gt, estimate)
    ci = answer.get("ci90")
    ci90 = tuple(ci) if isinstance(ci, (list, tuple)) and len(ci) == 2 else None
    conf = answer.get("confidence")
    note = "llm-alone (no sim)"
    if conf is not None:
        note += f", stated confidence={conf}"
    if picked is None:
        note += "; answer did not map to a candidate"
    # llm_alone has no tools: it estimates the NAMED canonical option directly, so its
    # "measured" config IS that candidate. Set measured_hash to the canonical hash so the
    # identity detector grades its CI like-for-like (it never constructs a variant).
    canon_hash = next((c.get("config_hash") for c in gt["candidates"]
                       if c["label"] == picked), None) if picked else None
    return ArmDecision(
        arm=arm, scenario_id=decision.scenario_id, picked=picked,
        metric_estimate=estimate, ci90=ci90, rollouts_used=0,
        notes=note, seeds_used=[], measured_hash=canon_hash)


def _default_llm_answer(prompt: str, model: str | None) -> str:
    """Real single-turn model call with NO tools (Max-spending). Lazy SDK import;
    exercised only on a `--live` run — the mapping above is what tests cover."""
    import asyncio

    from claude_agent_sdk import ClaudeAgentOptions, query

    from agent.runner import DISALLOWED_BUILTINS, _clean_env

    options = ClaudeAgentOptions(
        system_prompt=LLM_ALONE_SYSTEM, allowed_tools=[],
        disallowed_tools=DISALLOWED_BUILTINS, max_turns=1, model=model,
        env=_clean_env())

    async def collect() -> str:
        out: list[str] = []
        async for message in query(prompt=prompt, options=options):
            for block in getattr(message, "content", None) or []:
                if getattr(block, "text", None):
                    out.append(block.text)
        return "\n".join(out)

    return asyncio.run(collect())


def llm_alone(decision: Decision, budget: int = 200, gt: dict | None = None, *,
              answer_fn=None, model: str | None = None) -> ArmDecision:
    """Same model + the SAME scene summary the agent's get_scene_summary returns,
    but NO sim tools — must still output a recommendation + 90% CI. Quantifies H1,
    the grounding gap; spends zero rollouts. ``answer_fn`` is the injectable seam
    (tests pass a stub returning the model's raw JSON text); the default makes the
    real Max call. ``budget`` is accepted only for one calling convention."""
    if gt is None:
        raise ValueError("llm_alone arm needs the cached GT dict for grading")
    from engine.summary import scene_summary_text

    base = fill_defaults(load_config(decision.base_path()))
    prompt = _llm_alone_prompt(decision, scene_summary_text(base))
    raw = answer_fn(prompt) if answer_fn else _default_llm_answer(prompt, model)
    return _llm_answer_to_arm_decision(decision, gt, _parse_json_answer(raw), arm="llm_alone")
