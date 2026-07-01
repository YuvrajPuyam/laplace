"""engine/commands.py — deterministic command grammar for the live decision-twin.

Turns a typed operator COMMAND ("add a lane between A3_15 and A4_15", "use 12 robots")
into a config edit that can be applied to the twin INSTANTLY — no LLM round-trip — so the
common edits feel immediate and never misfire. Anything this grammar does not recognise
returns None, and the caller falls back to the LLM for open-ended phrasing or questions
(the hybrid path). Pure and side-effect free, so it is fully unit-testable.

Recognised commands (case-insensitive):
  - add a lane / cross-aisle / shortcut between <NODE> and <NODE>   -> layout.extra_edges
  - use/set the fleet to N robots   |   add/remove N robots         -> fleet.amr_count
  - set demand to N  |  N orders per minute|hour                    -> demand.arrival_rate_per_min
  - make the edge <NODE> to <NODE> one-way                          -> layout.edge_overrides
  - move <STATION_ID> to <NODE>                                     -> edits {id: node}

Returns {"kind", "summary", and EITHER "patch" (a dot-path config patch) OR "edits"
({station_id: node})}, or None when nothing matches. Node ids look like 'A3_15'.
"""

from __future__ import annotations

import re

_NODE = r"[A-Za-z]\d+_\d+"          # e.g. A3_15  (aisle 3, position 15 m)
_node_re = re.compile(_NODE)


def _nodes(text: str) -> list[str]:
    return [m.group(0).upper() for m in _node_re.finditer(text)]


def parse_command(text: str, config: dict) -> dict | None:
    """Parse one operator command against a config; None if unrecognised (LLM fallback)."""
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()
    nodes = _nodes(t)

    # --- move a station: "move P1 to A3_10" (check first so 'move' isn't eaten elsewhere) ---
    if re.search(r"\bmove\b", low):
        ids = {s["id"].lower(): s["id"]
               for kind in ("pick", "pack", "charge", "dock")
               for s in (config.get("stations", {}) or {}).get(kind, []) or []}
        for low_id, real_id in ids.items():
            if re.search(rf"\b{re.escape(low_id)}\b", low) and nodes:
                dest = nodes[-1]
                return {"kind": "station", "summary": f"Move {real_id} to {dest}",
                        "edits": {real_id: dest}}

    # --- one-way edge: "make the edge A3_15 to A4_15 one-way" ---
    if re.search(r"one[\s-]?way", low) and len(nodes) >= 2:
        edge = f"{nodes[0]}->{nodes[1]}"
        return {"kind": "oneway", "summary": f"Make {nodes[0]}->{nodes[1]} one-way",
                "patch": {"layout.edge_overrides": [{"edge": edge, "one_way": True}]}}

    # --- add a lane / cross-aisle / shortcut between two nodes ---
    if (re.search(r"\b(lane|cross[\s-]?aisle|shortcut|connection|link|edge)\b", low)
            and re.search(r"\b(add|open|create|build|connect|put|join)\b", low)
            and len(nodes) >= 2):
        return {"kind": "lane", "summary": f"Add a lane between {nodes[0]} and {nodes[1]}",
                "patch": {"layout.extra_edges":
                          [{"from": nodes[0], "to": nodes[1], "bidirectional": True}]}}

    # --- fleet: relative delta ("add 3 robots") then absolute ("use 12 robots") ---
    unit = r"(?:robots?|amrs?|units?|vehicles?)"
    m = re.search(rf"\b(add|remove|drop|cut)\s+(\d+)\s+(?:more\s+)?{unit}\b", low)
    if m:
        cur = int((config.get("fleet", {}) or {}).get("amr_count", 0))
        delta = int(m.group(2)) if m.group(1) == "add" else -int(m.group(2))
        val = max(1, cur + delta)
        return {"kind": "fleet", "summary": f"Set the fleet to {val} robots",
                "patch": {"fleet.amr_count": val}}
    if re.search(rf"\b(fleet|{unit})\b", low):
        m = re.search(rf"(?:fleet|{unit})\D{{0,15}}(\d+)|(\d+)\s+{unit}", low)
        if m:
            n = int(m.group(1) or m.group(2))
            if n >= 1:
                return {"kind": "fleet", "summary": f"Set the fleet to {n} robots",
                        "patch": {"fleet.amr_count": n}}

    # --- demand: "N orders per minute|hour" or "set demand to N" ---
    md = re.search(r"(\d+(?:\.\d+)?)\s*orders?\s*(?:per|/|a)\s*(min|minute|hr|hour)", low)
    if md:
        v = float(md.group(1))
        if md.group(2) in ("hr", "hour"):
            v = round(v / 60.0, 4)
        return {"kind": "demand", "summary": f"Set demand to {v} orders/min",
                "patch": {"demand.arrival_rate_per_min": v}}
    if "demand" in low or "arrival" in low:
        m = re.search(r"(\d+(?:\.\d+)?)", low)
        if m:
            v = float(m.group(1))
            return {"kind": "demand", "summary": f"Set demand to {v} orders/min",
                    "patch": {"demand.arrival_rate_per_min": v}}

    return None
