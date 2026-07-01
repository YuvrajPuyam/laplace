"""Trace -> staged decision-twin events (the streaming product, North Star S0-S3).

A finished or in-flight episode trace (engine/episode.py `ep.trace(...)` records:
`episode_created` + per-tool `{tool, params, result_summary}`) is mapped to an
ordered list of STAGE EVENTS that the viewer streams over SSE as a progressively
disclosed report:

  plan        S0  the question + scene orientation
  experiment  S1  each propose / rollout / comparison as the agent works
  preliminary S1  synthesized marker at the first significant finding
  refined     S2  power check / follow-up comparisons
  report      S3  the final submitted recommendation

This is a PURE function over trace records, so it is identical for a live agent
trace and a recorded one (replay) — and unit-testable without spending the Max
window. Every number it surfaces comes from the trace (a compare_configs result),
never invented (CLAUDE.md non-negotiable #5).
"""

from __future__ import annotations

import re
from typing import Any

_HASH_RE = re.compile(r"\b[0-9a-f]{12}\b")   # config_hash tokens cited in the recommendation

# tools that are low-signal narration; collapsed out of the stream
_SKIP_TOOLS = {"get_budget"}


def _fix_text(s: Any) -> Any:
    """Repair mojibake (UTF-8 mis-decoded as cp1252/latin1 — e.g. an em-dash stored
    as 'â€"'). Only touches strings showing the tell-tale markers, then re-encodes
    and decodes as UTF-8; returns the input unchanged on clean text or on failure."""
    if not isinstance(s, str) or ("â€" not in s and "Ã" not in s):
        return s
    for enc in ("cp1252", "latin1"):
        try:
            return s.encode(enc).decode("utf-8")
        except (UnicodeError, ValueError):
            continue
    return s


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) else None


def _ci(x: Any) -> list[float] | None:
    """A [lo, hi] CI as two floats, or None if absent/malformed (so the viewer
    can guard on it). Never fabricated — passes through only real engine numbers."""
    if (isinstance(x, (list, tuple)) and len(x) == 2
            and all(isinstance(v, (int, float)) for v in x)):
        return [float(x[0]), float(x[1])]
    return None


def _significant(p: float | None) -> bool:
    return p is not None and p < 0.05


def trace_to_stages(records: list[dict]) -> list[dict]:
    """Map engine trace records to ordered stage events for the streaming twin."""
    stages: list[dict] = []
    seen_significant = False
    last_patch = None        # most recent proposed config patch (legacy fallback)
    patch_by_hash: dict[str, dict] = {}   # config_hash -> the patch that produced it

    def emit(stage: str, kind: str, title: str, detail: Any = None) -> None:
        stages.append({"stage": stage, "kind": kind, "title": title,
                       "detail": detail, "t": len(stages)})

    for r in records:
        if r.get("event") == "episode_created":
            emit("plan", "question", "Question", _fix_text(r.get("question", "")))
            continue
        tool = r.get("tool")
        if not tool or tool in _SKIP_TOOLS:
            continue
        params = r.get("params") or {}
        result = r.get("result_summary") or {}

        if tool == "get_scene_summary":
            emit("plan", "orient", "Reading the facility", None)

        elif tool == "propose_config":
            if isinstance(params.get("patch"), dict):
                last_patch = params["patch"]
                h = result.get("config_hash")
                if h:
                    patch_by_hash[h] = params["patch"]   # map each candidate to its config hash
            emit("experiment", "propose", "Proposed a candidate",
                 {"label": params.get("label"),
                  "diff": result.get("diff_summary"),
                  "config_hash": result.get("config_hash")})

        elif tool == "run_rollouts":
            seeds = result.get("seeds_used") or []
            emit("experiment", "rollouts",
                 f"Ran {len(seeds)} paired rollouts",
                 {"n_seeds": len(seeds),
                  "config_hashes": params.get("config_hashes")})

        elif tool == "compare_configs":
            diff = _num(result.get("diff_mean"))
            p = _num(result.get("p_value"))
            metric = params.get("metric")
            # 90% CI of the paired (CRN) difference, surfaced so the twin can show a
            # whisker next to diff_mean/p — already engine-computed (stats.paired_compare),
            # never invented (CLAUDE.md #5). None for legacy traces lacking the field.
            ci90 = _ci(result.get("ci90_diff"))
            emit("experiment", "comparison", f"Compared on {metric}",
                 {"metric": metric, "diff_mean": diff, "p_value": p, "ci90": ci90})
            if _significant(p) and not seen_significant:
                seen_significant = True
                emit("preliminary", "signal",
                     "Preliminary signal",
                     {"metric": metric, "diff_mean": diff, "p_value": p, "ci90": ci90})

        elif tool == "power_check":
            emit("refined", "power", "Checked statistical power",
                 {"n_pairs_required": result.get("n_pairs_required")})

        elif tool == "render_evidence":
            emit("experiment", "render", "Queued an evidence render",
                 {"job_id": result.get("job_id")})

        elif tool == "submit_report":
            rep = params if isinstance(params, dict) else {}
            caveats = rep.get("caveats")
            # Apply the RECOMMENDED candidate, not the last one proposed, so "add a lane" is
            # actually SEEN in the twin. Resolve its config_hash from the STRUCTURED report field
            # (primary_metric.recommended.config) first — reliable — then fall back to a hash cited
            # in the prose (legacy). Map that hash to the patch that produced it. If the recommended
            # config is a non-candidate (e.g. the baseline = keep as-is), apply nothing.
            rec_text = _fix_text(rep.get("recommendation")) or ""
            cited = set(_HASH_RE.findall(rec_text))
            pm = rep.get("primary_metric") if isinstance(rep.get("primary_metric"), dict) else {}
            recb = pm.get("recommended") if isinstance(pm.get("recommended"), dict) else {}
            rec_hash = recb.get("config") or recb.get("config_hash") or recb.get("hash")
            if isinstance(rec_hash, str):
                cited.add(rec_hash)                 # the recommendation's own config, structurally
            chosen = next((pt for h, pt in patch_by_hash.items() if h in cited), None)
            if chosen is not None:
                config_patch = chosen
            elif cited:
                config_patch = None                # report names a non-candidate config -> no change
            else:
                config_patch = last_patch
            # A report the engine REJECTED (accepted is explicitly False) is not a
            # recommendation — surface it as a NEUTRAL 'inconclusive' outcome so the
            # viewer never renders an unvalidated answer as if it passed. accepted None
            # (legacy traces) keeps the historical 'final' behaviour.
            accepted = (result or {}).get("accepted")
            is_final = accepted is not False
            emit("report", "final" if is_final else "inconclusive",
                 "Recommendation" if is_final else "No confident recommendation",
                 {"recommendation": rec_text,
                  "confidence": rep.get("confidence"),
                  "primary_metric": rep.get("primary_metric"),
                  "mechanism": _fix_text(rep.get("mechanism")),
                  "caveats": [_fix_text(c) for c in caveats] if isinstance(caveats, list)
                  else _fix_text(caveats),
                  "config_patch": config_patch if is_final else None,  # don't apply a rejected config
                  "accepted": accepted})

    return stages
