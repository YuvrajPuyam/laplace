"""Contract B.1 metrics, computed FROM THE EVENT LOG (one format, four
consumers — metrics is consumer #1 and uses nothing the log doesn't contain,
plus the config-derived navgraph for edge lengths/capacities).

Measured window = [warmup_minutes, sim_minutes]. Conventions (documented in
sim/README.md):
- An order is MEASURED iff it arrived at t >= warmup_minutes.
- orders_completed / latency percentiles / throughput: measured orders that
  completed by sim end.
- orders_abandoned: measured orders not completed by sim end (includes
  in-flight; conservation: arrived = completed + abandoned).
- amr_utilization_pct: fraction of the window an AMR is tasked
  (task_assigned -> order_complete, clipped to the window), fleet mean.
- deadhead_pct: % of AMR travel distance while not carrying an order =
  legs to pick (before that order's pick service_start) + charge legs.
- charge_downtime_pct: charge_start -> charge_end time, clipped, fleet mean.
- station_wait: minutes between station-queue entry and service_start
  (0 when a slot was free on arrival); p95 per station over the window.
- edge_congestion: mean occupancy/capacity over the window per directed edge.
"""

from __future__ import annotations

import json

import numpy as np

from . import events as ev
from .navgraph import NavGraph


def station_wait_samples(rows: list[ev.Row], warmup: float = 0.0) -> dict[str, list[float]]:
    """All queue-wait samples per station (0.0 when a slot was free on
    arrival), for service_starts at t >= warmup. Used by the M/M/c-style
    queueing validation tests; the p95 metric uses the same definition."""
    enter: dict[tuple[str, str], float] = {}
    waits: dict[str, list[float]] = {}
    for t, etype, eid, event, loc, payload in rows:
        if event == ev.AMR_ENTER_QUEUE:
            p = json.loads(payload)
            if p["kind"] == "station":
                enter[(eid, p["at"])] = t
        elif event == ev.SERVICE_START:
            st = json.loads(payload)["station"]
            t0 = enter.pop((eid, st), None)
            if t >= warmup:
                waits.setdefault(st, []).append(t - t0 if t0 is not None else 0.0)
    return waits


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def compute_metrics(rows: list[ev.Row], config: dict,
                    graph: NavGraph | None = None) -> dict:
    g = graph if graph is not None else NavGraph(config)
    warmup = float(config["horizon"]["warmup_minutes"])
    sim_end = float(config["horizon"]["sim_minutes"])
    window = sim_end - warmup
    n_amrs = config["fleet"]["amr_count"]

    station_kind = {}
    for kind in ("pick", "pack"):
        for s in config["stations"][kind]:
            station_kind[s["id"]] = kind

    order_arrival: dict[str, float] = {}
    order_amr: dict[str, str] = {}
    completed: dict[str, tuple[float, float]] = {}   # order -> (t, latency)
    pick_service_start: dict[str, float] = {}        # order -> t
    assign_t: dict[str, float] = {}                  # order -> t

    amr_queue_enter: dict[tuple[str, str], float] = {}  # (amr, station) -> t
    station_waits: dict[str, list[float]] = {}
    edge_occ_time: dict[str, float] = {}
    travel_total = 0.0
    deadhead = 0.0
    charge_open: dict[str, float] = {}
    charge_time: dict[str, float] = {}

    for t, etype, eid, event, loc, payload in rows:
        if event == ev.AMR_DEPART_EDGE:
            p = json.loads(payload)
            u_name, v_name = p["edge"].split("->")
            u, v = g.node_index(u_name), g.node_index(v_name)
            pool = g.pools[g.pool_index(u, v)]
            dur = pool["length"] / (p["speed_mps"] * 60.0)
            overlap = min(t + dur, sim_end) - max(t, warmup)
            if overlap > 0:
                edge_occ_time[p["edge"]] = edge_occ_time.get(p["edge"], 0.0) + overlap
            if t >= warmup:
                travel_total += pool["length"]
                order = p["order"]
                if order is None:
                    deadhead += pool["length"]
                else:
                    started = pick_service_start.get(order)
                    if started is None or t < started:
                        deadhead += pool["length"]
        elif event == ev.ORDER_ARRIVED:
            order_arrival[eid] = t
        elif event == ev.TASK_ASSIGNED:
            order_amr[eid] = json.loads(payload)["amr"]
            assign_t[eid] = t
        elif event == ev.ORDER_COMPLETE:
            completed[eid] = (t, json.loads(payload)["latency_min"])
        elif event == ev.SERVICE_START:
            p = json.loads(payload)
            st = p["station"]
            if station_kind.get(st) == "pick":
                pick_service_start[p["order"]] = t
            if t >= warmup:
                entered = amr_queue_enter.pop((eid, st), None)
                wait = (t - entered) if entered is not None else 0.0
                station_waits.setdefault(st, []).append(wait)
        elif event == ev.AMR_ENTER_QUEUE:
            p = json.loads(payload)
            if p["kind"] == "station":
                amr_queue_enter[(eid, p["at"])] = t
        elif event == ev.CHARGE_START:
            charge_open[eid] = t
        elif event == ev.CHARGE_END:
            t0 = charge_open.pop(eid, None)
            if t0 is not None:
                overlap = min(t, sim_end) - max(t0, warmup)
                if overlap > 0:
                    charge_time[eid] = charge_time.get(eid, 0.0) + overlap

    # charges still open at sim end
    for amr_id, t0 in charge_open.items():
        overlap = sim_end - max(t0, warmup)
        if overlap > 0:
            charge_time[amr_id] = charge_time.get(amr_id, 0.0) + overlap

    measured = {o for o, ta in order_arrival.items() if ta >= warmup}
    done = [o for o in measured if o in completed]
    latencies = [completed[o][1] for o in done]

    # utilization: tasked intervals clipped to window
    tasked: dict[str, float] = {}
    for o, t0 in assign_t.items():
        amr_id = order_amr[o]
        t1 = completed[o][0] if o in completed else sim_end
        overlap = min(t1, sim_end) - max(t0, warmup)
        if overlap > 0:
            tasked[amr_id] = tasked.get(amr_id, 0.0) + overlap
    util = sum(tasked.values()) / (n_amrs * window) * 100.0 if window > 0 else 0.0

    top5 = sorted(
        ({"edge": e,
          "occupancy_pct": round(
              occ / (window * g.pools[g.pool_index(*map(g.node_index, e.split("->")))]["capacity"])
              * 100.0, 2)}
         for e, occ in edge_occ_time.items()),
        key=lambda d: (-d["occupancy_pct"], d["edge"]),
    )[:5]

    return {
        "throughput_orders_per_hr": round(len(done) / (window / 60.0), 4) if window > 0 else 0.0,
        "p50_order_latency_min": round(_pctl(latencies, 50), 4),
        "p95_order_latency_min": round(_pctl(latencies, 95), 4),
        "amr_utilization_pct": round(util, 4),
        "station_wait_p95_min": {
            st: round(_pctl(waits, 95), 4) for st, waits in sorted(station_waits.items())
        },
        "edge_congestion_top5": top5,
        "deadhead_pct": round(deadhead / travel_total * 100.0, 4) if travel_total > 0 else 0.0,
        "charge_downtime_pct": round(
            sum(charge_time.values()) / (n_amrs * window) * 100.0, 4) if window > 0 else 0.0,
        "orders_completed": len(done),
        "orders_abandoned": len(measured) - len(done),
    }
