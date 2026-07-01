"""The eight tool handlers (schemas/tools.md), dispatched over an Episode.

dispatch() is the single entry point used by BOTH transports (FastAPI, MCP):
it charges the tool-call budget, executes, traces, and either returns the
result dict or raises ToolError (transports serialize the envelope).

NOTE (flagged): compare_configs responses include a call_id beyond the frozen
return schema — tools.md §8 requires reports to reference "a compare_configs
call id", so an id must be surfaced to the agent somewhere. Smallest possible
addition; needs Yuv's blessing in the tools.md wording.
"""

from __future__ import annotations

import json

from sim.config import config_hash  # noqa: F401  (re-export convenience)

from . import stats
from .episode import Episode
from .errors import ToolError
from .report import validate_report
from .summary import EDITABLE_BOUNDS, EDITABLE_ROOTS, scene_summary_text

CAMERAS = ("overview", "congestion_closeup", "follow_amr")
KINDS = ("clip", "still", "side_by_side")


def _require(params: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in params]
    if missing:
        raise ToolError("invalid_params", f"missing required params: {missing}")


# Params that tools.md declares as object/array/number but that some MCP
# clients deliver as JSON-encoded STRINGS (e.g. patch='{"fleet.amr_count": 5}').
# dispatch() decodes them so BOTH transports (FastAPI-direct + MCP-proxy) behave
# identically. This only WIDENS accepted input — the handler validation below
# still rejects anything that isn't the right shape. Without it a stringified
# `patch` fails the isinstance(dict) check and the agent cannot register a
# single config (see runs/ep_cb46b1e7d4: 18 wasted calls, zero rollouts).
_JSON_PARAMS = {
    "propose_config": ("patch",),
    "run_rollouts": ("config_hashes", "n_seeds", "horizon_minutes"),
    "power_check": ("observed_effect", "observed_sd_of_diff", "target_power"),
    "render_evidence": ("config_hashes", "seed", "t_range_min"),
    # submit_report's structured fields get stringified by the same clients
    # (observed: primary_metric as a JSON string, confidence as "0.95") — the
    # report is a free-form object so the MCP schema can't type its fields.
    "submit_report": ("primary_metric", "confidence", "evidence", "experiments",
                      "caveats"),
}


def _coerce_json_params(tool: str, params: dict) -> dict:
    out = dict(params)
    for key in _JSON_PARAMS.get(tool, ()):
        val = out.get(key)
        if isinstance(val, str):
            try:
                out[key] = json.loads(val)
            except json.JSONDecodeError:
                pass  # leave as-is; handler validation reports the bad shape
    return out


def get_scene_summary(ep: Episode, params: dict) -> dict:
    _require(params, "scenario_id")
    cfg = ep.store.get(params["scenario_id"])
    if cfg is None:
        raise ToolError("unknown_scenario", f"no scenario '{params['scenario_id']}'",
                        {"known": ep.store.ids()})
    h = ep._register(cfg, label=f"baseline:{params['scenario_id']}",
                     base=params["scenario_id"])
    return {"summary_text": scene_summary_text(cfg), "config": cfg,
            "config_hash": h, "editable_bounds": EDITABLE_BOUNDS}


def propose_config(ep: Episode, params: dict) -> dict:
    _require(params, "base", "patch", "label")
    patch = params["patch"]
    if not isinstance(patch, dict) or not patch:
        raise ToolError("invalid_params", "patch must be a non-empty object of dot-paths")
    for path in patch:
        root = path.split(".")[0]
        if root not in EDITABLE_ROOTS:
            raise ToolError("patch_path_not_editable",
                            f"'{path}' is not editable (identity and schema fields "
                            "are frozen)", {"editable_roots": list(EDITABLE_ROOTS)})
    h, base_cfg = ep.propose(params["base"], patch, params["label"])

    def old_at(path: str):
        node = base_cfg
        for k in path.split("."):
            if not isinstance(node, dict) or k not in node:
                return "<unset>"
            node = node[k]
        return node

    parts = []
    for path, new in patch.items():
        old = old_at(path)
        fmt = lambda x: json.dumps(x) if not isinstance(x, str) else x  # noqa: E731
        o, n = fmt(old), fmt(new)
        if len(o) > 60:
            o = o[:57] + "..."
        if len(n) > 60:
            n = n[:57] + "..."
        parts.append(f"{path}: {o} -> {n}")
    return {"config_hash": h, "diff_summary": "; ".join(parts)}


def run_rollouts(ep: Episode, params: dict) -> dict:
    _require(params, "config_hashes", "n_seeds")
    hashes = params["config_hashes"]
    n = params["n_seeds"]
    if not isinstance(hashes, list) or not 1 <= len(hashes) <= 8:
        raise ToolError("invalid_params", "config_hashes must list 1..8 configs")
    if len(set(hashes)) != len(hashes):
        raise ToolError("invalid_params", "config_hashes contains duplicates")
    if not isinstance(n, int) or not 1 <= n <= 100:
        raise ToolError("invalid_params", "n_seeds must be an integer in 1..100")
    horizon = params.get("horizon_minutes")
    if horizon is not None and (not isinstance(horizon, int) or not 60 <= horizon <= 1440):
        raise ToolError("invalid_params", "horizon_minutes must be an integer in 60..1440")
    return ep.run_rollouts(hashes, n, horizon)


def compare_configs(ep: Episode, params: dict) -> dict:
    _require(params, "hash_a", "hash_b", "metric")
    ha, hb, metric = params["hash_a"], params["hash_b"], params["metric"]
    for h in (ha, hb):
        if h not in ep.configs:
            raise ToolError("unknown_config", f"config '{h}' was never proposed",
                            {"known": list(ep.configs)})
    va, vb, seeds = ep.paired_metric(ha, hb, metric)
    if len(seeds) < 5:
        raise ToolError("insufficient_pairs",
                        f"need >= 5 common seeds, found {len(seeds)}",
                        {"common_seeds": seeds})
    out = stats.paired_compare(va, vb)
    warnings = [w for w in (ep.abandonment_warning(ha), ep.abandonment_warning(hb)) if w]
    out["warnings"] = warnings
    call_id = f"cmp_{len(ep.compare_calls):04d}"
    out["call_id"] = call_id
    ep.compare_calls.append({"call_id": call_id, "hash_a": ha, "hash_b": hb,
                             "metric": metric, "seeds": seeds, **out})
    return out


def power_check(ep: Episode, params: dict) -> dict:
    _require(params, "observed_effect", "observed_sd_of_diff")
    n_req = stats.power_n_pairs(float(params["observed_effect"]),
                                float(params["observed_sd_of_diff"]),
                                float(params.get("target_power", 0.8)))
    left = ep.budgets["rollouts"] - ep.rollouts_spent
    return {"n_pairs_required": n_req,
            "achievable_within_budget": 2 * n_req <= left}


def render_evidence(ep: Episode, params: dict) -> dict:
    _require(params, "kind", "config_hashes", "seed", "t_range_min", "camera")
    kind, hashes, seed = params["kind"], params["config_hashes"], params["seed"]
    t_range, camera = params["t_range_min"], params["camera"]
    if kind not in KINDS or camera not in CAMERAS:
        raise ToolError("invalid_params", f"kind in {KINDS}, camera in {CAMERAS}")
    want = 2 if kind == "side_by_side" else 1
    if not isinstance(hashes, list) or len(hashes) != want:
        raise ToolError("invalid_params", f"kind '{kind}' takes exactly {want} config(s)")
    if not (isinstance(t_range, list) and len(t_range) == 2 and t_range[0] < t_range[1]):
        raise ToolError("invalid_params", "t_range_min must be [start, end], start < end")
    if kind != "still" and t_range[1] - t_range[0] > 20:
        raise ToolError("invalid_params", "clips are limited to 20 sim-minutes")
    if ep.renders_spent >= ep.budgets["renders"]:
        raise ToolError("render_budget_exhausted",
                        f"render budget ({ep.budgets['renders']}) exhausted")
    for h in hashes:
        res = ep.results.get((h, seed))
        if res is None or not (ep.log_dir / f"{h}_s{seed}.events.parquet").exists():
            raise ToolError("log_unavailable",
                            f"no retained event log for ({h}, seed {seed}) — "
                            "render only seeds you rolled out")
    job = {"job_id": f"rnd_{len(ep.render_jobs):03d}", "status": "queued",
           "uri": None, "kind": kind, "config_hashes": hashes, "seed": seed,
           "t_range_min": t_range, "camera": camera}
    ep.render_jobs.append(job)
    ep.renders_spent += 1
    ep.render_backend.submit(job, ep)
    return {"job_id": job["job_id"]}


def get_budget(ep: Episode, params: dict) -> dict:  # noqa: ARG001
    state = ep.budget_state()
    return {**state["budget"], "render_jobs": state["render_jobs"]}


def submit_report(ep: Episode, params: dict) -> dict:
    violations = validate_report(params, ep)
    ep.report_attempts += 1
    if violations:
        if ep.report_attempts >= 2:
            ep.closed = True
        return {"accepted": False, "violations": violations}
    ep.report = params
    ep.closed = True
    with open(ep.dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return {"accepted": True, "violations": []}


HANDLERS = {
    "get_scene_summary": get_scene_summary,
    "propose_config": propose_config,
    "run_rollouts": run_rollouts,
    "compare_configs": compare_configs,
    "power_check": power_check,
    "render_evidence": render_evidence,
    "get_budget": get_budget,
    "submit_report": submit_report,
}


def dispatch(ep: Episode, tool: str, params: dict) -> dict:
    if tool not in HANDLERS:
        raise ToolError("unknown_tool", f"no tool '{tool}'", {"tools": list(HANDLERS)})
    if ep.closed:
        raise ToolError("episode_closed", "episode is closed (report submitted "
                        "or repair attempts exhausted)")
    ep.charge_tool_call(tool)
    params = _coerce_json_params(tool, params)
    try:
        result = HANDLERS[tool](ep, params)
    except ToolError as e:
        ep.trace({"tool": tool, "params": params, "error": e.envelope()["error"]})
        raise
    summary = {k: result[k] for k in ("config_hash", "diff_summary", "seeds_used",
                                      "p_value", "diff_mean", "ci90_diff", "job_id",
                                      "accepted", "n_pairs_required") if k in result}
    ep.trace({"tool": tool, "params": params, "result_summary": summary})
    return result
