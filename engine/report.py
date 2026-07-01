"""submit_report validation (tools.md §8, report shape from spec §6.4).

Hard rules implemented:
- required fields + types, confidence in [0, 1];
- every number in primary_metric must be traceable to engine data: it must
  match (rel 0.5% / abs 0.05 — the agent may round for prose) either a field
  of a recorded compare_configs call, or the engine's own mean/CI90 computed
  from stored rollout results. This is the engine-side enforcement of
  "recommending a config never rolled out is a rejection": numbers for a
  never-rolled-out config cannot trace to anything.
- evidence uris must be completed render jobs;
- confidence vs CIs (v1 heuristic): claiming > 0.9 while the recommended and
  rejected-alternative CI90s overlap is a violation.

NOTE (flagged, not silently changed): spec §6.4 reports per-config ci90, but
compare_configs only returns CIs of the paired DIFF. The engine therefore
recomputes per-config mean/CI90 from stored results for the traceability
check. A cleaner contract would surface per-config CIs in a tool response —
Yuv's call.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

REQUIRED = ("question", "recommendation", "primary_metric", "mechanism",
            "confidence", "evidence", "experiments", "caveats")


def _close(x: float, y: float) -> bool:
    return math.isclose(x, y, rel_tol=5e-3, abs_tol=0.05)


def _engine_stats(episode, metric: str) -> list[tuple[float, float, float]]:
    """(mean, ci90_lo, ci90_hi) per config with >= 2 results, engine-computed."""
    out = []
    by_hash: dict[str, list[float]] = {}
    for (h, _s), res in episode.results.items():
        v = res["metrics"].get(metric)
        if isinstance(v, (int, float)):
            by_hash.setdefault(h, []).append(float(v))
    for vals in by_hash.values():
        if len(vals) >= 2:
            a = np.asarray(vals)
            se = a.std(ddof=1) / math.sqrt(len(a))
            t = sps.t.ppf(0.95, len(a) - 1)
            out.append((float(a.mean()), float(a.mean() - t * se), float(a.mean() + t * se)))
    return out


def _traceable_numbers(episode, metric: str) -> list[float]:
    nums: list[float] = []
    for c in episode.compare_calls:
        if c["metric"] == metric:
            nums += [c["mean_a"], c["mean_b"], c["diff_mean"],
                     *c["ci95_diff"], *c["ci90_diff"]]
    for mean, lo, hi in _engine_stats(episode, metric):
        nums += [mean, lo, hi]
    return nums


def validate_report(report: dict, episode) -> list[str]:
    v: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a JSON object"]
    for key in REQUIRED:
        if key not in report:
            v.append(f"missing required field: {key}")
    if v:
        return v

    conf = report["confidence"]
    if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
        v.append("confidence must be a number in [0, 1]")

    pm = report["primary_metric"]
    if not isinstance(pm, dict) or "name" not in pm:
        v.append("primary_metric must be an object with a 'name'")
        return v
    metric = pm["name"]
    pool = _traceable_numbers(episode, metric)

    def check_block(label: str, block) -> tuple[float, float] | None:
        if not isinstance(block, dict):
            v.append(f"primary_metric.{label} must be an object")
            return None
        mean = block.get("mean")
        ci = block.get("ci90")
        if not isinstance(mean, (int, float)) or not (isinstance(ci, list) and len(ci) == 2):
            v.append(f"primary_metric.{label} needs mean and ci90=[lo, hi]")
            return None
        for x in (mean, ci[0], ci[1]):
            if not any(_close(x, y) for y in pool):
                v.append(
                    f"primary_metric.{label}: {x} does not trace to any "
                    f"compare_configs call or engine-computed statistic for "
                    f"'{metric}' — numbers must come from experiments")
        return float(ci[0]), float(ci[1])

    rec_ci = check_block("baseline", pm.get("baseline")) and \
        check_block("recommended", pm.get("recommended"))
    rej = pm.get("rejected_alternative")
    rej_ci = None
    if rej is not None:
        rej_ci = check_block("rejected_alternative", rej)

    if rec_ci and rej_ci and isinstance(conf, (int, float)) and conf > 0.9:
        lo1, hi1 = rec_ci
        lo2, hi2 = rej_ci
        if max(lo1, lo2) < min(hi1, hi2):  # CI90s overlap
            v.append("confidence > 0.9 but recommended and rejected_alternative "
                     "CI90s overlap — confidence must come from CIs")

    done_uris = {j.get("uri") for j in episode.render_jobs if j["status"] == "done"}
    evidence = report["evidence"]
    if not isinstance(evidence, list):
        v.append("evidence must be a list of {type, uri} objects (use [] if you "
                 "ran no renders) — not prose")
    else:
        for i, e in enumerate(evidence):
            uri = e.get("uri") if isinstance(e, dict) else None
            if uri not in done_uris:
                v.append(f"evidence[{i}].uri is not a completed render job")

    if not report["experiments"]:
        v.append("experiments must cite at least one experiment "
                 "(configs + seeds actually run)")
    if not episode.results:
        v.append("no rollouts were run this episode — a recommendation "
                 "cannot be grounded")
    return v
