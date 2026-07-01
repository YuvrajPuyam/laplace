"""Eval metrics (spec §5.3): grade an ArmDecision against the cached ground truth.

- decision accuracy: did the arm pick the GT-optimal candidate?
- regret: normalized score gap, 0 = picked the optimum, 1 = picked the worst.
  (Generalizes the spec's throughput-only formula to mixed objectives such as the
  fleet SLA, where "best raw metric" and "best decision" differ.)
- calibration: does the picked config's TRUE metric fall inside the arm's stated
  90% CI? Coverage over the suite vs the 90% target (only for arms that state a
  CI — the LLM-alone / agent arms; grid_search states none).
- cost: rollouts used (for the equal-budget H2 comparison).
"""

from __future__ import annotations

from eval.baselines import ArmDecision

TARGET_COVERAGE = 0.90


def interval_score(lo: float, hi: float, y: float, alpha: float = 0.10) -> float:
    """Winkler interval score for a central (1-alpha) interval: width + miss penalty.
    Lower is better; rewards SHARP intervals and penalizes misses ~proportionally, so coverage
    cannot be gamed by widening the CI (sharpness-subject-to-calibration; Gneiting/Raftery)."""
    w = hi - lo
    if y < lo:
        return w + (2.0 / alpha) * (lo - y)
    if y > hi:
        return w + (2.0 / alpha) * (y - hi)
    return w


def _candidate(gt: dict, label: str) -> dict:
    return next(c for c in gt["candidates"] if c["label"] == label)


def grade(arm: ArmDecision, gt: dict) -> dict:
    scores = {c["label"]: c["mean_score"] for c in gt["candidates"]}
    opt = gt["gt_optimum"]
    best, worst = scores[opt], min(scores.values())
    rng = best - worst
    gt_metric = _candidate(gt, opt)["mean_metric"]

    # An arm that abstained or recommended something outside the decision space
    # (e.g. a failed/rejected episode) is graded as a wrong decision at worst-case
    # regret — not a crash. It states no usable CI, so it does not enter coverage.
    if arm.picked not in scores:
        return {
            "arm": arm.arm, "scenario_id": arm.scenario_id,
            "picked": arm.picked, "gt_optimum": opt, "correct": False,
            "regret": 1.0, "picked_metric": None, "gt_metric": gt_metric,
            "ci_covered": None, "identity_mismatch": None, "interval_score": None,
            "rollouts": arm.rollouts_used,
        }

    regret = 0.0 if rng <= 0 else (best - scores[arm.picked]) / rng
    picked_metric = _candidate(gt, arm.picked)["mean_metric"]
    covered = None
    identity_mismatch = None
    iscore = None
    if arm.ci90 is not None:
        lo, hi = arm.ci90
        canon_hash = _candidate(gt, arm.picked).get("config_hash")
        # Only score coverage when the arm MEASURED the same config it is graded against. If
        # the report cites no config_hash, or a different one, its CI bounds a DIFFERENT object
        # — a config-identity confound, not (mis)calibration — so exclude it from coverage and
        # flag it. interval_score is still recorded (it characterizes the stated interval).
        identity_mismatch = bool(arm.measured_hash is None or arm.measured_hash != canon_hash)
        iscore = round(interval_score(lo, hi, picked_metric), 4)
        covered = None if identity_mismatch else bool(lo <= picked_metric <= hi)

    return {
        "arm": arm.arm, "scenario_id": arm.scenario_id,
        "picked": arm.picked, "gt_optimum": opt, "correct": arm.picked == opt,
        "regret": round(regret, 4),
        "picked_metric": picked_metric, "gt_metric": gt_metric,
        "ci_covered": covered, "identity_mismatch": identity_mismatch,
        "interval_score": iscore, "rollouts": arm.rollouts_used,
    }


def aggregate(graded: list[dict]) -> dict:
    n = len(graded)
    if n == 0:
        return {"decision_accuracy": 0.0, "mean_regret": 0.0,
                "total_rollouts": 0, "ci_coverage": None, "ece_gap": None, "n": 0}
    cov = [g["ci_covered"] for g in graded if g["ci_covered"] is not None]
    coverage = sum(cov) / len(cov) if cov else None
    iscores = [g.get("interval_score") for g in graded if g.get("interval_score") is not None]
    mismatches = sum(1 for g in graded if g.get("identity_mismatch"))
    return {
        "decision_accuracy": round(sum(g["correct"] for g in graded) / n, 4),
        "mean_regret": round(sum(g["regret"] for g in graded) / n, 4),
        "total_rollouts": sum(g["rollouts"] for g in graded),
        "ci_coverage": round(coverage, 4) if coverage is not None else None,
        "n_coverage_scored": len(cov),          # rows that were like-for-like (identity-matched)
        "n_identity_mismatch": mismatches,       # rows excluded from coverage (config not verified)
        "mean_interval_score": round(sum(iscores) / len(iscores), 4) if iscores else None,
        "ece_gap": round(abs(coverage - TARGET_COVERAGE), 4)
        if coverage is not None else None,
        "n": n,
    }
