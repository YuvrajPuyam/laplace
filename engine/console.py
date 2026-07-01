"""engine/console.py - testable backend for the Experiment Console (watch / edit / ask).

A pure CLIENT over the frozen contracts - it adds no new tool and changes no schema:
  - editable_objects(cfg): the tappable objects + the exact Contract A patch each edit writes.
  - build_patch(...): (object, op, value) -> a propose_config-style patch (validated, arrays
    replaced wholesale per Contract A); the only way a tap becomes a patch.
  - fast_preview(...): edit -> instant base-vs-patched estimate on a small shared CRN seed
    set, reusing the SAME rollout + paired-compare math as the validated path (so a "quick
    estimate" differs from a validated run only in seed count + labeling, never in math).
  - render_card(report): the structural FIREWALL - a Decision Card can be built ONLY from a
    validated submit_report; a quick_estimate can never populate primary_metric / CI.

Exposed levers are a NARROWER subset of the engine's EDITABLE_BOUNDS (structure + policy
SELECTION only) - never the objective (service_lognorm / arrival_rate), which is scenario-fixed.
"""

from __future__ import annotations

from engine.stats import paired_compare
from engine.summary import EDITABLE_BOUNDS
from sim.config import apply_patch, config_hash, validate_config
from sim.runner import run_many

PREVIEW_METRICS = ("throughput_orders_per_hr", "p50_order_latency_min",
                   "p95_order_latency_min", "amr_utilization_pct")

# Structure + policy SELECTION only. Deliberately omits service_lognorm / arrival_rate
# (the objective/scenario) and identity fields. A subset of EDITABLE_BOUNDS - never wider.
CONSOLE_EXPOSED_PATHS = {
    "fleet.amr_count", "fleet.dispatch", "fleet.congestion_penalty",
    "fleet.charge_threshold_pct", "stations.pick", "stations.pack",
    "stations.charge", "layout.extra_edges",
}
_SLOT_MIN, _SLOT_MAX = 1, 4          # service_station.slots bounds (Contract A $defs)


class EditError(ValueError):
    """An edit rejected before it became a patch (out of bounds / unknown op)."""


def _fleet(cfg: dict) -> dict:
    return cfg["fleet"]


def editable_objects(cfg: dict) -> list[dict]:
    """Project a loaded config into the console's tappable objects + affordances."""
    f = _fleet(cfg)
    objs: list[dict] = [{
        "object_id": "fleet", "kind": "fleet",
        "label": f"Fleet ({f['amr_count']} AMRs)", "anchor": {"node": None},
        "affordances": [{"edit_op": "amr_count_delta", "control": "stepper",
                         "path": "fleet.amr_count", "current": f["amr_count"],
                         "bounds": EDITABLE_BOUNDS["fleet.amr_count"], "step": 1}],
    }, {
        "object_id": "policy:dispatch", "kind": "policy", "label": "Dispatch policy",
        "anchor": {"node": None},
        "affordances": [{"edit_op": "set_dispatch", "control": "select",
                         "path": "fleet.dispatch",
                         "current": f.get("dispatch", "nearest_idle"),
                         "options": EDITABLE_BOUNDS["fleet.dispatch"]["enum"]}],
    }, {
        "object_id": "policy:congestion", "kind": "policy", "label": "Congestion penalty",
        "anchor": {"node": None},
        "affordances": [{"edit_op": "set_congestion_penalty", "control": "stepper",
                         "path": "fleet.congestion_penalty",
                         "current": f.get("congestion_penalty", 2.0),
                         "bounds": EDITABLE_BOUNDS["fleet.congestion_penalty"], "step": 0.5}],
    }, {
        "object_id": "policy:charge", "kind": "policy", "label": "Charge threshold",
        "anchor": {"node": None},
        "affordances": [{"edit_op": "set_charge_threshold", "control": "stepper",
                         "path": "fleet.charge_threshold_pct",
                         "current": f.get("charge_threshold_pct", 0.15),
                         "bounds": EDITABLE_BOUNDS["fleet.charge_threshold_pct"], "step": 0.05}],
    }]
    for kind in ("pick", "pack", "charge"):
        for s in cfg["stations"][kind]:
            objs.append({
                "object_id": f"station:{kind}:{s['id']}", "kind": "station",
                "label": f"{kind.title()} {s['id']} ({s['slots']} slots)",
                "anchor": {"node": s["node"]},
                "affordances": [{"edit_op": "slots_delta", "control": "stepper",
                                 "path": f"stations.{kind}", "current": s["slots"],
                                 "bounds": {"min": _SLOT_MIN, "max": _SLOT_MAX}, "step": 1}],
            })
    objs.append({
        "object_id": "extra_edges", "kind": "extra_edge", "label": "Cross-aisle shortcuts",
        "anchor": {"node": None},
        "affordances": [{"edit_op": "toggle_extra_edge", "control": "toggle",
                         "path": "layout.extra_edges",
                         "current": list(cfg["layout"].get("extra_edges", [])),
                         "bounds": EDITABLE_BOUNDS["layout.extra_edges"]}],
    })
    return objs


def build_patch(cfg: dict, object_id: str, edit_op: str, value) -> dict:
    """(object_id, edit_op, value) -> a propose_config-style patch dict. Pure; validates
    against the bounds; rebuilds whole arrays (Contract A replaces arrays). Raises EditError.
    Does NOT apply the patch - apply_patch + validate_config remain the authority."""
    f = _fleet(cfg)
    if object_id == "fleet" and edit_op == "amr_count_delta":
        b = EDITABLE_BOUNDS["fleet.amr_count"]
        nv = int(f["amr_count"]) + int(value)
        if not (b["min"] <= nv <= b["max"]):
            raise EditError(f"amr_count {nv} out of {b}")
        return {"fleet.amr_count": nv}
    if object_id == "policy:dispatch" and edit_op == "set_dispatch":
        if value not in EDITABLE_BOUNDS["fleet.dispatch"]["enum"]:
            raise EditError(f"dispatch {value!r} not allowed")
        return {"fleet.dispatch": value}
    if object_id == "policy:congestion" and edit_op == "set_congestion_penalty":
        b = EDITABLE_BOUNDS["fleet.congestion_penalty"]
        v = float(value)
        if not (b["min"] <= v <= b["max"]):
            raise EditError(f"congestion_penalty {v} out of {b}")
        return {"fleet.congestion_penalty": v}
    if object_id == "policy:charge" and edit_op == "set_charge_threshold":
        b = EDITABLE_BOUNDS["fleet.charge_threshold_pct"]
        v = float(value)
        if not (b["min"] <= v <= b["max"]):
            raise EditError(f"charge_threshold_pct {v} out of {b}")
        return {"fleet.charge_threshold_pct": v}
    if object_id.startswith("station:") and edit_op == "slots_delta":
        _, kind, sid = object_id.split(":")
        arr = [dict(s) for s in cfg["stations"][kind]]
        hit = next((s for s in arr if s["id"] == sid), None)
        if hit is None:
            raise EditError(f"no station {sid}")
        nv = int(hit["slots"]) + int(value)
        if not (_SLOT_MIN <= nv <= _SLOT_MAX):
            raise EditError(f"slots {nv} out of [{_SLOT_MIN},{_SLOT_MAX}]")
        hit["slots"] = nv
        return {f"stations.{kind}": arr}              # whole array replaced
    if object_id == "extra_edges" and edit_op == "toggle_extra_edge":
        cur = [dict(e) for e in cfg["layout"].get("extra_edges", [])]
        e = {"from": value["from"], "to": value["to"],
             "bidirectional": bool(value.get("bidirectional", True))}
        match = next((x for x in cur if x["from"] == e["from"] and x["to"] == e["to"]), None)
        if match is not None:
            cur.remove(match)                          # toggle off
        else:
            if len(cur) >= EDITABLE_BOUNDS["layout.extra_edges"]["max_items"]:
                raise EditError("too many extra_edges")
            cur.append(e)                              # toggle on
        return {"layout.extra_edges": cur}
    raise EditError(f"unknown edit ({object_id!r}, {edit_op!r})")


def fast_preview(base_cfg: dict, patch: dict, *, n_seeds: int = 8,
                 seed_base: int = 0) -> dict:
    """Instant base-vs-patched estimate on a small SHARED CRN seed set. Reuses run_many +
    paired_compare (same math as the validated path). Always labeled 'quick_estimate'."""
    patched = apply_patch(base_cfg, patch)
    validate_config(patched)
    seeds = list(range(seed_base, seed_base + n_seeds))
    ra = run_many(base_cfg, seeds, write_log=False)
    rb = run_many(patched, seeds, write_log=False)
    deltas = {}
    for m in PREVIEW_METRICS:
        cmp = paired_compare([r["metrics"][m] for r in ra],
                             [r["metrics"][m] for r in rb])
        deltas[m] = {"base_mean": round(cmp["mean_a"], 4),
                     "patched_mean": round(cmp["mean_b"], 4),
                     "diff_mean": round(cmp["diff_mean"], 4),
                     "ci90": cmp["ci90_diff"], "p_value": round(cmp["p_value"], 4),
                     "method": cmp["method"],
                     "direction": "up" if cmp["diff_mean"] > 0 else "down"}
    return {
        "fidelity": "quick_estimate", "n_seeds": n_seeds, "seeds_used": seeds,
        "base_hash": config_hash(base_cfg), "patched_hash": config_hash(patched),
        "deltas": deltas,
        "disclaimer": (f"Quick estimate on {n_seeds} seeds - not physics-validated. "
                       "Ask the agent to validate before deciding."),
    }


_CARD_REQUIRED = ("recommendation", "primary_metric", "mechanism", "confidence")


def render_card(report: dict) -> dict:
    """THE FIREWALL: build a Decision Card ONLY from a validated submit_report. A
    quick_estimate (or any non-report dict) cannot become a card - its numbers can never
    populate primary_metric / CI."""
    if not isinstance(report, dict) or report.get("fidelity") == "quick_estimate":
        raise EditError("a Decision Card cannot be built from a quick estimate")
    if not all(k in report for k in _CARD_REQUIRED):
        raise EditError("not a validated report (missing required fields)")
    return {
        "recommendation": report["recommendation"],
        "primary_metric": report["primary_metric"],   # verbatim; traces to compare_configs
        "mechanism": report["mechanism"],
        "confidence": report["confidence"],
        "fidelity": report.get("evidence_fidelity", "des-validated"),
        "show_the_work": report.get("experiments", []),
    }


class ConsoleSession:
    """Container for a human-in-the-loop session: human edits accumulate in working_cfg;
    fast-preview estimates live in `previews` (never a card); Decision Cards come ONLY from
    validated reports. Agent questions spawn budgeted child episodes (the `ask` seam)."""

    def __init__(self, base_cfg: dict):
        self.working_cfg = base_cfg
        self.previews: dict[str, dict] = {}
        self.cards: list[dict] = []

    def preview_edit(self, object_id: str, edit_op: str, value, *,
                     n_seeds: int = 8) -> dict:
        patch = build_patch(self.working_cfg, object_id, edit_op, value)
        est = fast_preview(self.working_cfg, patch, n_seeds=n_seeds)
        eid = f"edit_{len(self.previews):03d}"
        self.previews[eid] = {"patch": patch, **est}
        return {"edit_id": eid, **est}

    def apply_edit(self, edit_id: str) -> dict:
        patch = self.previews[edit_id]["patch"]
        self.working_cfg = apply_patch(self.working_cfg, patch)
        validate_config(self.working_cfg)
        return {"config_hash": config_hash(self.working_cfg)}

    def card_from_report(self, report: dict) -> dict:
        card = render_card(report)        # firewall; raises on a quick_estimate
        self.cards.append(card)
        return card
