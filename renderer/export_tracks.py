"""export_tracks — MAIN env: rollout event log -> per-frame agent trajectories.

Runs one deterministic rollout, then uses the replay-sufficiency rule
(sim.replay.ReplayState) to sample every AMR's (x, y) and heading at a fixed
cadence over a sim-time window. The result is a tracks.json the Isaac builder
animates — same two-environment split as the scene export: the sim lives here,
Isaac never imports it.

  python -m renderer.export_tracks --scenario braess_dev `
      --patch renderer/scenes/shortcut_patch.json --seed 0 `
      --t0 45 --window 2.5 --n-frames 220 --out renderer/scenes/braess_tracks.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import json as _json

from engine.store import ScenarioStore
from sim import events as ev
from sim.config import apply_patch, fill_defaults
from sim.replay import ReplayState
from sim.runner import run_rollout


def _carrying(rows, config, agent_ids, times):
    """Per-AMR per-frame 'carrying an order' flag, derived from the event log.
    A robot is loaded from when it finishes a PICK service until it finishes
    the matching PACK service — the visible 'useful work' interval."""
    kind = {}
    for k in ("pick", "pack", "charge"):
        for s in config["stations"][k]:
            kind[s["id"]] = k
    changes: dict[str, list[tuple[float, bool]]] = {}
    for t, etype, aid, event, _loc, payload in rows:
        if etype != "amr" or event != ev.SERVICE_END:
            continue
        st = _json.loads(payload).get("station")
        if kind.get(st) == "pick":
            changes.setdefault(aid, []).append((t, True))
        elif kind.get(st) == "pack":
            changes.setdefault(aid, []).append((t, False))
    out = {}
    for aid in agent_ids:
        ch = changes.get(aid, [])
        flags, j, state = [], 0, False
        for t in times:
            while j < len(ch) and ch[j][0] <= t:
                state = ch[j][1]
                j += 1
            flags.append(state)
        out[aid] = flags
    return out


def _hud(rows, config, graph, times, agents, warmup):
    """Per-frame metrics + world-space callout anchors for the demo overlay.
    All numbers come from the event log so the HUD is honest."""
    import bisect

    import numpy as np

    # --- callout anchors (world xyz + label) -------------------------------
    anchors = []
    for ee in config["layout"].get("extra_edges", []):
        ux, uy = graph.node_xy[graph.node_index(ee["from"])]
        vx, vy = graph.node_xy[graph.node_index(ee["to"])]
        sc = ((ux + vx) / 2, (uy + vy) / 2)
        anchors.append({"xyz": [sc[0], sc[1], 0.6],
                        "label": "Capacity-1 shortcut", "kind": "alert"})
        break
    else:
        sc = None
    picks = config["stations"]["pick"]
    pk = picks[len(picks) // 2]
    px, py = graph.node_xy[graph.node_index(pk["node"])]
    anchors.append({"xyz": [px, py, 1.7], "label": "Pick station", "kind": "info"})
    pq = config["stations"]["pack"][0]
    qx, qy = graph.node_xy[graph.node_index(pq["node"])]
    anchors.append({"xyz": [qx, qy, 1.5], "label": "Pack station", "kind": "info"})

    # --- completed-order stream (for throughput + p95 latency) -------------
    comp = sorted((t, _json.loads(p).get("latency_min", 0.0))
                  for t, et, eid, ev_, loc, p in rows if ev_ == ev.ORDER_COMPLETE)
    ct = [c[0] for c in comp]
    cl = [c[1] for c in comp]
    lo = bisect.bisect_left(ct, warmup)  # exclude warmup completions

    frames = []
    for i, t in enumerate(times):
        carry = sum(int(agents[a]["carrying"][i]) for a in agents)
        occ = 0
        if sc:
            for a in agents:
                x, y = agents[a]["xy"][i]
                if abs(x - sc[0]) < 1.7 and abs(y - sc[1]) < 0.9:
                    occ += 1
        hi = bisect.bisect_right(ct, t)
        n_done = hi - lo
        hrs = max((t - warmup) / 60.0, 1e-6)
        thru = n_done / hrs
        lat = cl[lo:hi]
        p95 = float(np.percentile(lat, 95)) if len(lat) >= 5 else 0.0
        frames.append({"t": round(t, 3), "carrying": carry,
                       "shortcut_occ": occ,
                       "throughput": round(thru, 1), "p95_latency": round(p95, 1)})
    return {"anchors": anchors, "frames": frames}


# status-band colors by AMR state (emissive RGB), priority high->low
BAND = {
    "blocked":   [0.95, 0.12, 0.10],   # red — jammed on the capacity-1 edge
    "servicing": [0.10, 0.85, 0.95],   # cyan — picking / packing
    "charging":  [0.20, 0.45, 1.00],   # blue
    "carrying":  [0.25, 0.95, 0.35],   # green — loaded, doing useful work
    "moving":    [0.35, 0.70, 0.62],   # teal — deadheading empty
    "idle":      [0.18, 0.20, 0.24],   # dim — parked
}


def _spans(rows, amr, start_ev, end_ev):
    """Ordered [start,end) intervals of an event pair for one AMR."""
    out, open_t = [], None
    for t, et, eid, event, _loc, _p in rows:
        if et != "amr" or eid != amr:
            continue
        if event == start_ev:
            open_t = t
        elif event == end_ev and open_t is not None:
            out.append((open_t, t))
            open_t = None
    return out


def _state_bands(rows, config, graph, agent_ids, times, agents):
    """Per-AMR per-frame status-band color + per-frame shortcut occupancy."""
    sc = None
    for ee in config["layout"].get("extra_edges", []):
        ux, uy = graph.node_xy[graph.node_index(ee["from"])]
        vx, vy = graph.node_xy[graph.node_index(ee["to"])]
        sc = ((ux + vx) / 2, (uy + vy) / 2)
        break

    def in_band(x, y):
        return sc and abs(x - sc[0]) < 1.7 and abs(y - sc[1]) < 0.9

    bands = {}
    for aid in agent_ids:
        svc = _spans(rows, aid, ev.SERVICE_START, ev.SERVICE_END)
        chg = _spans(rows, aid, ev.CHARGE_START, ev.CHARGE_END)
        xy = agents[aid]["xy"]
        carry = agents[aid]["carrying"]
        cols = []
        for i, t in enumerate(times):
            x, y = xy[i]
            moving = i > 0 and ((x - xy[i - 1][0]) ** 2 + (y - xy[i - 1][1]) ** 2) > 9e-4
            servicing = any(s <= t < e for s, e in svc)
            charging = any(s <= t < e for s, e in chg)
            if in_band(x, y) and not moving:
                k = "blocked"
            elif servicing:
                k = "servicing"
            elif charging:
                k = "charging"
            elif carry[i]:
                k = "carrying"
            elif moving:
                k = "moving"
            else:
                k = "idle"
            cols.append(BAND[k])
        bands[aid] = cols

    occ = []
    for i in range(len(times)):
        occ.append(sum(1 for aid in agent_ids
                       if in_band(*agents[aid]["xy"][i])))
    return bands, occ


def _headings(xy: list[tuple[float, float]]) -> list[float]:
    """Travel-direction heading per frame (degrees), held through parked
    stretches so a stationary AMR doesn't spin."""
    headings: list[float] = []
    last = 0.0
    for i in range(len(xy)):
        if i + 1 < len(xy):
            dx = xy[i + 1][0] - xy[i][0]
            dy = xy[i + 1][1] - xy[i][1]
            if dx * dx + dy * dy > 1e-6:
                last = math.degrees(math.atan2(dy, dx))
        headings.append(last)
    return headings


def _edge_flow(config, g, rows, times):
    """Renderable navgraph LANES + per-frame occupancy, both derived from the
    event log (presentational; no contract/sim change). A lane is the meaningful
    unit a human reads: one corridor per aisle, one connector per cross-aisle
    segment, and each extra edge (the capacity-1 Braess shortcut) on its own.
    Occupancy = how many AMRs are traversing that lane's pool(s) at each frame,
    reconstructed from amr_depart_edge intervals (the replay-key event)."""
    import bisect

    grid = config["layout"]["grid"]
    n_aisles, length = grid["aisles"], grid["aisle_length_m"]
    cross = sorted(grid["cross_aisles"])

    def nm(a, p):
        return f"A{a}_{p:02d}"

    lanes, lane_pools = [], []   # parallel: lane dict + the pool indices it covers
    # aisle lanes: one corridor per aisle, covering all its 1 m segment pools
    for a in range(1, n_aisles + 1):
        x0, y0 = g.node_xy[g.node_index(nm(a, 0))]
        x1, y1 = g.node_xy[g.node_index(nm(a, length))]
        pset = {g.pool_index(g.node_index(nm(a, p)), g.node_index(nm(a, p + 1)))
                for p in range(length)}
        cap = max((g.pools[pi]["capacity"] for pi in pset), default=2)
        lanes.append({"x0": round(x0, 3), "y0": round(y0, 3),
                      "x1": round(x1, 3), "y1": round(y1, 3),
                      "cap": cap, "kind": "aisle"})
        lane_pools.append(pset)
    # cross-aisle lanes: one per adjacent-aisle connector at each cross position
    for c in cross:
        for a in range(1, n_aisles):
            u, v = g.node_index(nm(a, c)), g.node_index(nm(a + 1, c))
            pi = g.pool_index(u, v)
            x0, y0 = g.node_xy[u]
            x1, y1 = g.node_xy[v]
            lanes.append({"x0": round(x0, 3), "y0": round(y0, 3),
                          "x1": round(x1, 3), "y1": round(y1, 3),
                          "cap": g.pools[pi]["capacity"], "kind": "cross"})
            lane_pools.append({pi})
    # extra edges (the Braess shortcut): each on its own, labelled by capacity
    for ee in config["layout"].get("extra_edges", []):
        u, v = g.node_index(ee["from"]), g.node_index(ee["to"])
        pi = g.pool_index(u, v)
        x0, y0 = g.node_xy[u]
        x1, y1 = g.node_xy[v]
        lanes.append({"x0": round(x0, 3), "y0": round(y0, 3),
                      "x1": round(x1, 3), "y1": round(y1, 3),
                      "cap": g.pools[pi]["capacity"], "kind": "shortcut",
                      "label": f"capacity-{g.pools[pi]['capacity']} shortcut"})
        lane_pools.append({pi})

    # reverse map: pool index -> the lane that renders it (each pool belongs to exactly one lane)
    pool_lane = {}
    for li, pset in enumerate(lane_pools):
        for pi in pset:
            pool_lane[pi] = li

    # per-pool occupancy + per-AMR per-frame lane assignment, from amr_depart_edge intervals
    by_amr: dict[str, list] = {}
    for t, etype, aid, event, _loc, payload in rows:
        if etype == "amr":
            by_amr.setdefault(aid, []).append((t, event, payload))
    n_pools = len(g.pools)
    pool_occ = [[0] * len(times) for _ in range(n_pools)]
    on_lane = {aid: [-1] * len(times) for aid in by_amr}   # which lane each AMR is on per frame (-1 = parked/queued)
    for aid, evs in by_amr.items():
        for k, (t, event, payload) in enumerate(evs):
            if event != ev.AMR_DEPART_EDGE:
                continue
            p = _json.loads(payload)
            frm, to = p["edge"].split("->")
            pi = g.pool_index(g.node_index(frm), g.node_index(to))
            speed = p["speed_mps"] * 60.0
            t_arr = t + (g.pools[pi]["length"] / speed if speed > 0 else 0.0)
            t_end = min(t_arr, evs[k + 1][0]) if k + 1 < len(evs) else t_arr
            lo, hi = bisect.bisect_left(times, t), bisect.bisect_left(times, t_end)
            li = pool_lane.get(pi, -1)
            for fi in range(lo, hi):
                pool_occ[pi][fi] += 1
                on_lane[aid][fi] = li

    # a lane's congestion = its BUSIEST segment (max over its pools), not the sum:
    # 4 robots spread over a 30 m aisle is not a jam; 2 sharing one cap-2 segment is.
    edge_occ = [[max((pool_occ[pi][fi] for pi in pset), default=0) for fi in range(len(times))]
                for pset in lane_pools]
    return lanes, edge_occ, on_lane


# ── intent taxonomy ───────────────────────────────────────────────────────────
# What each AMR is TRYING to do this frame - drives the per-bot color and the live
# fleet-status panel, so a human can read WHY the floor is moving the way it is. All
# derived from the event log (honest): current service/charge/queue state, the carrying
# flag, and - for a moving bot - the KIND of the next station it is headed to.
INTENT_COLOR = {
    "carrying":  [0.25, 0.95, 0.35],   # green     - loaded, taking an order to pack
    "to_pick":   [0.32, 0.71, 0.97],   # blue      - empty, going to collect an order
    "servicing": [0.10, 0.85, 0.95],   # cyan      - at a pick/pack station, being served
    "to_charge": [1.00, 0.62, 0.16],   # amber     - battery low, heading to a charger
    "charging":  [0.22, 0.45, 1.00],   # deep blue - plugged in at a charger
    "queued":    [0.95, 0.16, 0.12],   # red       - blocked: waiting for a slot / busy edge
    "idle":      [0.32, 0.35, 0.40],   # dim       - parked, no task
}
INTENT_LABEL = {
    "carrying": "Carrying to pack", "to_pick": "Fetching order", "servicing": "At station",
    "to_charge": "Low battery", "charging": "Charging", "queued": "Waiting", "idle": "Idle",
}
INTENT_ORDER = ["carrying", "to_pick", "servicing", "to_charge", "charging", "queued", "idle"]


def _intents(rows, config, agent_ids, times, agents):
    """Per-AMR per-frame intent string + per-frame fleet counts, from the event log."""
    kind = {}
    for k in ("pick", "pack", "charge"):
        for s in config["stations"][k]:
            kind[s["id"]] = k
    per = {}
    for aid in agent_ids:
        svc = _spans(rows, aid, ev.SERVICE_START, ev.SERVICE_END)
        chg = _spans(rows, aid, ev.CHARGE_START, ev.CHARGE_END)
        que = _spans(rows, aid, ev.AMR_ENTER_QUEUE, ev.AMR_EXIT_QUEUE)
        anchors = []                                       # (t, dest-kind) for look-ahead
        for t, et, eid, event, _loc, p in rows:
            if et != "amr" or eid != aid:
                continue
            if event == ev.CHARGE_START:
                anchors.append((t, "charge"))
            elif event == ev.SERVICE_START:
                anchors.append((t, kind.get(_json.loads(p).get("station"), "pick")))
        anchors.sort()
        carry, xy = agents[aid]["carrying"], agents[aid]["xy"]
        seq = []
        for i, t in enumerate(times):
            moving = i > 0 and ((xy[i][0] - xy[i - 1][0]) ** 2
                                + (xy[i][1] - xy[i - 1][1]) ** 2) > 9e-4
            if any(s <= t < e for s, e in chg):
                it = "charging"
            elif any(s <= t < e for s, e in svc):
                it = "servicing"
            elif any(s <= t < e for s, e in que):
                it = "queued"
            elif carry[i]:
                it = "carrying"
            elif moving:
                nxt = next((ak for at, ak in anchors if at > t), None)
                it = "to_charge" if nxt == "charge" else "to_pick"
            else:
                it = "idle"
            seq.append(it)
        per[aid] = seq
    fleet = []
    for i in range(len(times)):
        c = {k: 0 for k in INTENT_ORDER}
        for aid in agent_ids:
            c[per[aid][i]] += 1
        fleet.append(c)
    return per, fleet


def _charge_occ(rows, config, times):
    """Per-charge-station per-frame count of AMRs charging + each station's slot capacity
    (the hard limit). Lets the viewer show 'charging / capacity' and which bays are busy."""
    cap = {s["id"]: s.get("slots", 1) for s in config["stations"]["charge"]}
    occ = {cid: [0] * len(times) for cid in cap}
    open_at, spans = {}, []
    for t, et, aid, event, _loc, p in rows:
        if et != "amr":
            continue
        if event == ev.CHARGE_START:
            open_at[aid] = (t, _json.loads(p).get("station"))
        elif event == ev.CHARGE_END and aid in open_at:
            t0, stn = open_at.pop(aid)
            spans.append((t0, t, stn))
    for t0, t1, stn in spans:
        if stn not in occ:
            continue
        for i, t in enumerate(times):
            if t0 <= t < t1:
                occ[stn][i] += 1
    return occ, cap


def _station_occ(rows, config, times):
    """Per-station per-frame count of AMRs being serviced/charged + slot capacity, for ALL
    pick/pack/charge stations - drives each station's 'busy' activity light in the viewer."""
    cap = {}
    for k in ("pick", "pack", "charge"):
        for s in config["stations"][k]:
            cap[s["id"]] = s.get("slots", 1)
    occ = {sid: [0] * len(times) for sid in cap}
    starts = {ev.SERVICE_START, ev.CHARGE_START}
    ends = {ev.SERVICE_END, ev.CHARGE_END}
    open_at, spans = {}, []
    for t, et, aid, event, _loc, p in rows:
        if et != "amr":
            continue
        if event in starts:
            open_at[aid] = (t, _json.loads(p).get("station"))
        elif event in ends and aid in open_at:
            t0, stn = open_at.pop(aid)
            spans.append((t0, t, stn))
    for t0, t1, stn in spans:
        if stn not in occ:
            continue
        for i, t in enumerate(times):
            if t0 <= t < t1:
                occ[stn][i] += 1
    return occ, cap


def _apply_lanes(agents, edges, spacing):
    """Bake a TWO-LANE system into the track xy: a robot traversing an aisle/cross-aisle is
    offset to ONE side of its travel direction by ~one lane, so opposing traffic uses opposite
    lanes and never overlaps. Done in the DATA (not the viewer) so BOTH the Three.js viewer and
    the Isaac render show clean lanes - the old viewer-only warp never reached the render."""
    off = min(0.95, max(0.4, spacing * 0.18))   # lane half-offset; fits the clear aisle width
    W = 3
    for a in agents.values():
        xy = a["xy"]
        on = a.get("on_lane") or [-1] * len(xy)
        N = len(xy)
        new = [list(p) for p in xy]
        for i in range(N):
            li = on[i]
            if li is None or li < 0 or li >= len(edges):
                continue
            e = edges[li]
            if e.get("kind") != "aisle":   # ONLY offset the long aisle corridors; cross-aisles sit
                continue                    # at the rack-bay boundary, so offsetting there clips racks
            ex, ez = e["x1"] - e["x0"], e["y1"] - e["y0"]
            el = math.hypot(ex, ez) or 1.0
            ex, ez = ex / el, ez / el
            j0, j1 = max(0, i - W), min(N - 1, i + W)
            sgn = 1.0 if ((xy[j1][0] - xy[j0][0]) * ex + (xy[j1][1] - xy[j0][1]) * ez) >= 0 else -1.0
            # taper the offset to 0 near the aisle ENDS so a turning bot merges to the centreline
            # at the intersection instead of cutting the corner into a rack
            s = (xy[i][0] - e["x0"]) * ex + (xy[i][1] - e["y0"]) * ez
            t = max(0.0, min(1.0, min(s, el - s) / 2.0))
            o = off * t
            new[i][0] = xy[i][0] + ez * sgn * o      # perpendicular offset, side by travel dir
            new[i][1] = xy[i][1] - ex * sgn * o
        a["xy"] = new


def _cross_corridor(agents, agent_ids, n_frames, edges, ylen, margin=1.6):
    """Route LATERAL cross-aisle travel through the real open corridor in FRONT of / BEHIND the
    rack block, not along its edge. The extracted sim clamps aisle length to the dense rack block,
    so its cross-aisles (sim y=0 and y=ylen) sit exactly on the rack block's front/back FACE - a
    bot changing aisles there ploughs through every rack row it crosses (verified against the real
    USD point cloud: 14/60 lateral samples land in rack material at sim y=0). Pushing the lateral
    move out by `margin` puts it in the clear floor in front of / behind the racks. The offset is
    smoothed over a few frames so the bot eases out of the aisle into the corridor and back, instead
    of teleporting sideways at the junction."""
    W = 4
    for aid in agent_ids:
        a = agents[aid]
        xy = a["xy"]
        on = a.get("on_lane") or [-1] * n_frames
        off = [0.0] * n_frames
        for i in range(n_frames):
            li = on[i]
            if li is None or li < 0 or li >= len(edges):
                continue
            e = edges[li]
            if e.get("kind") != "cross":      # only the lateral cross-aisle move clips; aisle
                continue                       # (vertical) travel rides the clear gap already
            ymid = 0.5 * (e["y0"] + e["y1"])
            # push the FRONT cross-aisle forward (negative y) and the BACK one further back
            off[i] = (-margin - ymid) if ymid < ylen * 0.5 else (ylen + margin - ymid)
        soff = [sum(off[max(0, i - W):min(n_frames, i + W + 1)])
                / (min(n_frames, i + W + 1) - max(0, i - W)) for i in range(n_frames)]
        for i in range(n_frames):
            xy[i][1] += soff[i]


def _station_layout(config, g, ylen, margin=1.6):
    """Each station's sim (x,y) + kind, with FRONT (y~=0) and BACK (y~=ylen) stations pulled into
    the clear corridor (the same offset _cross_corridor uses for lateral travel) so a station prop -
    and the bots working at it - sit on open floor, never on the rack-block face. Pick stations sit
    mid-aisle on the real rack rows (the warehouse USD's own racks ARE the pick faces) and stay put."""
    out = []
    for kind in ("pick", "pack", "charge", "dock"):
        for s in config["stations"].get(kind, []):
            xi, yi = g.node_xy[g.node_index(s["node"])]
            corridor = yi < 1.0 or yi > ylen - 1.0
            yc = -margin if yi < 1.0 else (ylen + margin if yi > ylen - 1.0 else yi)
            out.append({"id": s["id"], "x": float(xi), "y": float(yc),
                        "kind": kind, "corridor": corridor})
    return out


def _dock_parked(agents, agent_ids, n_frames, stations, sep=1.6):
    """Dock PARKED robots AT their station (so packing / picking / charging is VISIBLE), not on an
    anonymous aisle centreline. Each parked bot snaps to its nearest station; multiple bots at one
    station fan out single-file in a CLEAR direction - up the aisle for in-aisle stations, along the
    corridor for front/back stations - so a serviced bot is shown working at the station instead of
    being relocated away from it. (Replaced the centreline docker, which erased all station work.)"""
    if not stations:
        return
    for i in range(n_frames):
        groups = {}
        for aid in agent_ids:
            if (agents[aid].get("on_lane") or [-1] * n_frames)[i] < 0:
                x, y = agents[aid]["xy"][i]
                si = min(range(len(stations)),
                         key=lambda k: (stations[k]["x"] - x) ** 2 + (stations[k]["y"] - y) ** 2)
                groups.setdefault(si, []).append(aid)
        for si, members in groups.items():
            st = stations[si]
            sx, sy = st["x"], st["y"]
            # fan extras in a clear direction: along the corridor (x) for front/back stations,
            # up the aisle (y) for mid-aisle stations
            fx, fy = (1.0, 0.0) if st.get("corridor") else (0.0, 1.0)
            members.sort(key=lambda aid: (agents[aid]["xy"][i][0], agents[aid]["xy"][i][1]))
            for idx, aid in enumerate(members):
                agents[aid]["xy"][i] = [sx + fx * idx * sep, sy + fy * idx * sep]


def _separate(agents, agent_ids, n_frames, edges, sep=1.6):   # > one AMR length, so bots never merge
    """Resolve overlaps by sliding MOVING bots single-file ALONG their own travel edge - never
    sideways (sideways is what used to push bots into shelves/walls). Docked/parked bots are fixed
    by _dock_parked and are never moved here; a moving bot slides clear of them along its path."""
    def edir(li):
        e = edges[li]
        ex, ez = e["x1"] - e["x0"], e["y1"] - e["y0"]
        el = math.hypot(ex, ez) or 1.0
        return ex / el, ez / el
    n = len(agent_ids)
    for i in range(n_frames):
        onl = [(agents[aid].get("on_lane") or [-1] * n_frames)[i] for aid in agent_ids]
        for _ in range(8):
            for k in range(n):
                for j in range(k + 1, n):
                    pk, pj = onl[k] < 0, onl[j] < 0
                    if pk and pj:
                        continue
                    A, B = agents[agent_ids[k]]["xy"][i], agents[agent_ids[j]]["xy"][i]
                    dx, dz = A[0] - B[0], A[1] - B[1]
                    d = math.hypot(dx, dz)
                    if d < sep:
                        push = sep - d
                        if (not pk) and 0 <= onl[k] < len(edges):    # slide k along its edge, away from B
                            ex, ez = edir(onl[k])
                            s = 1.0 if (dx * ex + dz * ez) >= 0 else -1.0
                            m = push if pj else push / 2
                            A[0] += ex * s * m; A[1] += ez * s * m
                        if (not pj) and 0 <= onl[j] < len(edges):    # slide j along its edge, away from A
                            ex, ez = edir(onl[j])
                            s = 1.0 if (-dx * ex - dz * ez) >= 0 else -1.0
                            m = push if pk else push / 2
                            B[0] += ex * s * m; B[1] += ez * s * m


def sample_tracks(config: dict, seed: int = 0, t0: float = 45.0,
                  window: float = 2.5, n_frames: int = 220,
                  label: str | None = None) -> tuple[dict, dict]:
    """Run ONE rollout and sample per-frame AMR tracks + a HUD over a window.
    Returns (tracks, hud) in memory — the shared core used by this CLI and by
    the live twin endpoint (engine/api.py /twin/simulate)."""
    _, rows = run_rollout(config, seed, write_log=False)
    state = ReplayState(rows, config)
    dt = window / max(n_frames - 1, 1)
    times = [t0 + i * dt for i in range(n_frames)]
    agent_ids = sorted(state.tracks)
    carrying = _carrying(rows, config, agent_ids, times)
    agents = {}
    for amr_id in agent_ids:
        xy = [state.position(amr_id, t) for t in times]
        agents[amr_id] = {
            "xy": [[round(x, 4), round(y, 4)] for x, y in xy],
            "heading_deg": [round(h, 2) for h in _headings(xy)],
            "carrying": [bool(c) for c in carrying[amr_id]],
        }
    _, shortcut_occ = _state_bands(rows, config, state.g, agent_ids, times, agents)
    intents, fleet = _intents(rows, config, agent_ids, times, agents)
    for amr_id in agent_ids:
        agents[amr_id]["band"] = [INTENT_COLOR[it] for it in intents[amr_id]]   # bot color = intent
        agents[amr_id]["intent"] = intents[amr_id]
    charge_occ, charge_cap = _charge_occ(rows, config, times)
    station_occ, station_cap = _station_occ(rows, config, times)
    edges, edge_occ, on_lane = _edge_flow(config, state.g, rows, times)
    for amr_id in agent_ids:
        agents[amr_id]["on_lane"] = on_lane.get(amr_id, [-1] * n_frames)
    # bake lane discipline + station docking into the DATA so BOTH the viewer and the Isaac render
    # are clean AND show real work: bots stay on the aisle CENTRELINE (the real corridor is too
    # narrow for two lanes of full-size AMRs); PARKED bots dock AT their station (so packing /
    # picking / charging is visible); lateral cross-aisle moves route through the open corridor in
    # front of / behind the rack block; residual overlaps yield single-file ALONG the travel edge.
    ylen = float(config["layout"]["grid"]["aisle_length_m"])
    stations = _station_layout(config, state.g, ylen)
    _dock_parked(agents, agent_ids, n_frames, stations)
    _cross_corridor(agents, agent_ids, n_frames, edges, ylen)   # lateral moves -> real open corridor
    _separate(agents, agent_ids, n_frames, edges)
    for amr_id in agent_ids:
        agents[amr_id]["xy"] = [[round(p[0], 4), round(p[1], 4)] for p in agents[amr_id]["xy"]]
    out = {"scenario": config["scenario_id"], "seed": seed, "t0": t0,
           "window": window, "dt": dt, "n_frames": n_frames,
           "agents": agents, "shortcut_occ": shortcut_occ,
           "edges": edges, "edge_occ": edge_occ,
           "charge_occ": charge_occ, "charge_cap": charge_cap,
           "station_occ": station_occ, "station_cap": station_cap, "fleet": fleet,
           "stations": stations,   # id/x/y/kind/corridor — props placed HERE, where bots dock
           "intent_color": INTENT_COLOR, "intent_label": INTENT_LABEL,
           "intent_order": INTENT_ORDER}
    warmup = config["horizon"]["warmup_minutes"]
    hud = _hud(rows, config, state.g, times, agents, warmup)
    hud["label"] = label or config["scenario_id"]
    hud["n_frames"] = n_frames
    hud["fleet"] = fleet
    return out, hud


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="export_tracks")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--patch", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t0", type=float, default=45.0, help="window start (sim-min)")
    ap.add_argument("--window", type=float, default=2.5, help="window length (sim-min)")
    ap.add_argument("--n-frames", type=int, default=220)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hud-out", default=None,
                    help="per-frame metrics + callout anchors (default: <out>_hud.json)")
    ap.add_argument("--label", default=None, help="scenario label for the HUD")
    ap.add_argument("--scenario-dirs", nargs="*",
                    default=["examples", "eval/dev_scenarios"])
    args = ap.parse_args(argv)

    store = ScenarioStore(dirs=tuple(args.scenario_dirs))
    config = store.get(args.scenario)
    if config is None:
        raise SystemExit(f"unknown scenario '{args.scenario}'")
    if args.patch:
        text = Path(args.patch).read_text(encoding="utf-8") \
            if Path(args.patch).exists() else args.patch
        config = apply_patch(config, json.loads(text))
    config = fill_defaults(config)

    out, hud = sample_tracks(config, args.seed, args.t0, args.window,
                             args.n_frames, label=args.label)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    hud_path = args.hud_out or args.out.replace(".json", "_hud.json")
    Path(hud_path).write_text(json.dumps(hud), encoding="utf-8")
    moved = sum(1 for a in out["agents"].values()
                if any(abs(a["xy"][0][0] - p[0]) + abs(a["xy"][0][1] - p[1]) > 0.5
                       for p in a["xy"]))
    print(json.dumps({"out": args.out, "n_frames": args.n_frames,
                      "agents": len(out["agents"]), "agents_that_move": moved,
                      "window_sim_min": args.window, "dt_sim_min": round(out["dt"], 4)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
