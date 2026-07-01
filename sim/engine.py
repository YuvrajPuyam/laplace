"""Tier-0 discrete-event engine.

Semantics (see sim/README.md for the modeling-decision log):
- All times in sim minutes; distances in meters; speeds in m/s.
- Congestion: edges are capacity pools shared by both directions; an AMR
  entering a full pool queues FIFO at its tail node and holds no capacity
  while queued (no hold-and-wait => no deadlock).
- Stations: `slots` parallel servers, FIFO queue at the node.
- Service time at station S for order o = exp(mu_S + sigma_S * z_o) minutes,
  with z_o pre-drawn in the CRN order stream (paired across configs).
- Task allocation (frozen v1): pending orders FIFO; the oldest unassigned
  order gets the nearest idle AMR (static shortest-path distance, ties by
  AMR index).
- Battery: drains per meter at edge departure; when an AMR becomes idle with
  battery below 15% it heads to the nearest charge station, queues for a
  slot, charges to full in charge_minutes. Battery never interrupts a task
  in progress and idle AMRs do not drain.
- AMRs start at the first dock's node, idle, full battery.
"""

from __future__ import annotations

import heapq
import json
import math
from collections import deque

from . import events as ev
from .navgraph import NavGraph
from .stream import generate_order_stream

# heap event kinds (priority field keeps SIM_END last among same-t events)
_K_ARRIVAL = 0      # order arrival
_K_EDGE_DONE = 1    # AMR reaches the far node of an edge
_K_SERVICE_DONE = 2
_K_CHARGE_DONE = 3
_K_WARMUP = 4
_K_SIM_END = 5

# AMR purposes (what the current leg is for)
_TO_PICK = 0
_TO_PACK = 1
_TO_CHARGE = 2

_BATTERY_THRESHOLD = 0.15


class _AMR:
    __slots__ = ("idx", "id", "node", "battery_m", "idle", "order",
                 "path", "path_i", "purpose", "station")

    def __init__(self, idx: int, node: int, battery_m: float):
        self.idx = idx
        self.id = f"amr_{idx:02d}"
        self.node = node
        self.battery_m = battery_m
        self.idle = True
        self.order = None          # _Order while tasked
        self.path = []             # list of (u, v, pool_idx)
        self.path_i = 0
        self.purpose = -1
        self.station = None        # _Station while queued/in service/charging


class _Order:
    __slots__ = ("idx", "id", "t_arrival", "pick", "pack", "z_pick", "z_pack",
                 "assigned", "completed")

    def __init__(self, idx, t_arrival, pick, pack, z_pick, z_pack):
        self.idx = idx
        self.id = f"ord_{idx:06d}"
        self.t_arrival = t_arrival
        self.pick = pick           # _Station
        self.pack = pack           # _Station
        self.z_pick = z_pick
        self.z_pack = z_pack
        self.assigned = False
        self.completed = False


class _Station:
    __slots__ = ("id", "kind", "node", "slots", "mu", "sigma", "busy",
                 "free_slots", "queue")

    def __init__(self, sid, kind, node, slots, mu=0.0, sigma=0.0):
        self.id = sid
        self.kind = kind           # "pick" | "pack" | "charge"
        self.node = node
        self.slots = slots
        self.mu = mu
        self.sigma = sigma
        self.busy = 0
        self.free_slots = list(range(slots - 1, -1, -1))  # pop() -> lowest
        self.queue = deque()       # waiting _AMRs


class Engine:
    def __init__(self, config: dict, seed: int):
        self.cfg = config
        self.seed = seed
        self.graph = NavGraph(config)
        g = self.graph

        fleet = config["fleet"]
        self.speed = fleet["speed_mps"]
        self.battery_cap = float(fleet["battery_capacity_m"])
        self.charge_minutes = float(fleet["charge_minutes"])
        self.congestion_aware = fleet["routing"] == "congestion_aware"
        # OPTIONAL, agent-controllable levers. Read via fleet.get with the frozen-v1
        # fallback and NEVER added to fill_defaults, so a config that omits them is
        # byte-identical (hash-identical) to the pre-lever canonical form (the
        # aisle_spacing_m pattern; guarded by tests/test_gt_cache_hashes.py).
        self.dispatch = fleet.get("dispatch", "nearest_idle")
        self.congestion_penalty = float(fleet.get("congestion_penalty", 2.0))
        self.charge_threshold = float(fleet.get("charge_threshold_pct", 0.15))
        # Composite-rule parameters (engine constants for now; pre-register before
        # claiming a "tuned" baseline). No RNG -> dispatch stays CRN-safe.
        self.atc_k = 2.0
        self.covert_kc = 2.0
        self.covert_s0 = 0.0

        st = config["stations"]
        self.picks = [_Station(s["id"], "pick", g.node_index(s["node"]), s["slots"],
                               s["service_lognorm"][0], s["service_lognorm"][1])
                      for s in st["pick"]]
        self.packs = [_Station(s["id"], "pack", g.node_index(s["node"]), s["slots"],
                               s["service_lognorm"][0], s["service_lognorm"][1])
                      for s in st["pack"]]
        self.charges = [_Station(s["id"], "charge", g.node_index(s["node"]), s["slots"])
                        for s in st["charge"]]
        dock_node = g.node_index(st["dock"][0]["node"])

        self.amrs = [_AMR(i, dock_node, self.battery_cap)
                     for i in range(fleet["amr_count"])]
        self.idle_amrs: set[int] = set(range(len(self.amrs)))

        self.sim_minutes = float(config["horizon"]["sim_minutes"])
        self.warmup = float(config["horizon"]["warmup_minutes"])

        # CRN stream -> orders (pack assigned at arrival time by policy)
        self.pack_rr = 0
        self.pack_policy = config["demand"]["pack_assignment"]
        self.draws = generate_order_stream(
            seed, config["demand"]["arrival_rate_per_min"], self.sim_minutes)
        self.orders: list[_Order] = []
        self.pending: deque[_Order] = deque()

        # edge pools runtime state
        self.pool_occ = [0] * len(g.pools)
        self.pool_queue: list[deque] = [deque() for _ in g.pools]
        # while queued for an edge, the AMR's intended (u, v, pool) is at
        # path[path_i]; the queue stores the AMR itself.

        self.rows: list[ev.Row] = []
        self.heap: list = []
        self._seq = 0
        self.t = 0.0
        self.orders_arrived = 0
        self.orders_completed_total = 0

    # ----------------------------------------------------------- helpers
    def _push(self, t: float, kind: int, data) -> None:
        self._seq += 1
        heapq.heappush(self.heap, (t, kind == _K_SIM_END, self._seq, kind, data))

    def _emit(self, t, entity_type, entity_id, event, location, payload: dict) -> None:
        self.rows.append((t, entity_type, entity_id, event, location,
                          json.dumps(payload, separators=(",", ":"))))

    def _service_minutes(self, station: _Station, z: float) -> float:
        import math
        return math.exp(station.mu + station.sigma * z)

    # ----------------------------------------------------------- run
    def run(self) -> list[ev.Row]:
        for i, d in enumerate(self.draws):
            self._push(d.t_arrival, _K_ARRIVAL, i)
        self._push(self.warmup, _K_WARMUP, None)
        self._push(self.sim_minutes, _K_SIM_END, None)

        heap = self.heap
        while heap:
            t, _, _, kind, data = heapq.heappop(heap)
            self.t = t
            if kind == _K_EDGE_DONE:
                self._on_edge_done(t, data)
            elif kind == _K_ARRIVAL:
                self._on_arrival(t, data)
            elif kind == _K_SERVICE_DONE:
                self._on_service_done(t, data)
            elif kind == _K_CHARGE_DONE:
                self._on_charge_done(t, data)
            elif kind == _K_WARMUP:
                self._emit(t, "sim", "sim", ev.SIM_WARMUP_END, "", {})
            else:  # _K_SIM_END
                abandoned = self.orders_arrived - self.orders_completed_total
                self._emit(t, "sim", "sim", ev.SIM_END, "",
                           {"orders_abandoned": abandoned})
                break
        return self.rows

    # ----------------------------------------------------------- handlers
    def _on_arrival(self, t: float, draw_idx: int) -> None:
        d = self.draws[draw_idx]
        pick = self.picks[min(int(d.u_pick * len(self.picks)), len(self.picks) - 1)]
        if self.pack_policy == "round_robin":
            pack = self.packs[self.pack_rr % len(self.packs)]
            self.pack_rr += 1
        else:  # shortest_queue at arrival time
            pack = min(self.packs, key=lambda s: (s.busy + len(s.queue)))
        order = _Order(len(self.orders), t, pick, pack, d.z_pick, d.z_pack)
        self.orders.append(order)
        self.orders_arrived += 1
        self._emit(t, "order", order.id, ev.ORDER_ARRIVED, pick.id,
                   {"pick": pick.id, "pack": pack.id})
        self.pending.append(order)
        self._try_assign(t)

    def _try_assign(self, t: float) -> None:
        # Dispatch by the selected rule; absent/'nearest_idle' -> the frozen path,
        # whose body is moved verbatim into _assign_nearest_idle (byte-identical).
        if self.dispatch == "atc":
            self._assign_priority(t, self._idx_atc)
        elif self.dispatch == "covert":
            self._assign_priority(t, self._idx_covert)
        else:
            self._assign_nearest_idle(t)

    def _assign_nearest_idle(self, t: float) -> None:
        g = self.graph
        while self.pending and self.idle_amrs:
            order = self.pending[0]
            best = None
            best_key = None
            for ai in sorted(self.idle_amrs):
                amr = self.amrs[ai]
                key = (g.shortest_dist(amr.node, order.pick.node), ai)
                if best_key is None or key < best_key:
                    best_key = key
                    best = amr
            self.pending.popleft()
            order.assigned = True
            self.idle_amrs.discard(best.idx)
            best.idle = False
            best.order = order
            self._emit(t, "order", order.id, ev.TASK_ASSIGNED, order.pick.id,
                       {"amr": best.id})
            self._start_leg(t, best, order.pick.node, _TO_PICK)

    # --- composite dispatching rules (ATC / COVERT), deterministic, no RNG ------
    def _svc_mean(self, station: _Station) -> float:
        return math.exp(station.mu + 0.5 * station.sigma * station.sigma)

    def _work(self, order: _Order) -> float:
        """Expected total work for an order (min): pick->pack travel + both service
        means. Uses the distribution MEAN, never the realized CRN draw, so dispatch
        cannot peek at realized service times (keeps the rule honest + config-stable)."""
        travel = self.graph.shortest_dist(order.pick.node, order.pack.node) / (self.speed * 60.0)
        return travel + self._svc_mean(order.pick) + self._svc_mean(order.pack)

    def _p_bar(self) -> float:
        vals = [self._work(o) for o in self.pending]
        m = sum(vals) / len(vals) if vals else 1.0
        return m if m > 1e-9 else 1.0

    def _idx_atc(self, order, amr, t, p_bar, g) -> float:
        p = self._work(order)
        tau = g.shortest_dist(amr.node, order.pick.node) / (self.speed * 60.0)
        age = t - order.t_arrival
        return (1.0 / p) * math.exp(-tau / (self.atc_k * p_bar)) * (1.0 + age / p_bar)

    def _idx_covert(self, order, amr, t, p_bar, g) -> float:
        p = self._work(order)
        tau = g.shortest_dist(amr.node, order.pick.node) / (self.speed * 60.0)
        slack = self.covert_s0 - (t - order.t_arrival) - tau
        return (1.0 / p) * max(0.0, min(1.0, 1.0 - slack / (self.covert_kc * p)))

    def _assign_priority(self, t: float, index_fn) -> None:
        g = self.graph
        while self.pending and self.idle_amrs:
            p_bar = self._p_bar()
            best = None
            best_key = None
            for order in self.pending:                 # deterministic FIFO iteration
                for ai in sorted(self.idle_amrs):
                    amr = self.amrs[ai]
                    idx = index_fn(order, amr, t, p_bar, g)
                    # maximize; tie-break oldest order then lowest amr index (total order)
                    key = (idx, -(t - order.t_arrival), -order.idx, -ai)
                    if best_key is None or key > best_key:
                        best_key = key
                        best = (order, amr)
            order, amr = best
            self.pending.remove(order)
            order.assigned = True
            self.idle_amrs.discard(amr.idx)
            amr.idle = False
            amr.order = order
            self._emit(t, "order", order.id, ev.TASK_ASSIGNED, order.pick.id,
                       {"amr": amr.id})
            self._start_leg(t, amr, order.pick.node, _TO_PICK)

    def _start_leg(self, t: float, amr: _AMR, dest: int, purpose: int) -> None:
        amr.purpose = purpose
        if amr.node == dest:
            self._on_leg_complete(t, amr)
            return
        if self.congestion_aware:
            amr.path = self.graph.congestion_aware_path(
                amr.node, dest, self.pool_occ, self.congestion_penalty)
        else:
            amr.path = self.graph.shortest_path(amr.node, dest)
        amr.path_i = 0
        self._try_enter_edge(t, amr)

    def _try_enter_edge(self, t: float, amr: _AMR) -> None:
        u, v, pi = amr.path[amr.path_i]
        pool = self.graph.pools[pi]
        if self.pool_occ[pi] < pool["capacity"]:
            self._enter_edge(t, amr, u, v, pi)
        else:
            self.pool_queue[pi].append(amr)
            edge = self.graph.edge_id(u, v)
            self._emit(t, "amr", amr.id, ev.AMR_ENTER_QUEUE, edge,
                       {"at": edge, "kind": "edge", "pos": len(self.pool_queue[pi]) - 1})

    def _enter_edge(self, t: float, amr: _AMR, u: int, v: int, pi: int) -> None:
        pool = self.graph.pools[pi]
        self.pool_occ[pi] += 1
        speed = min(self.speed, pool["max_speed"])
        order = amr.order
        self._emit(t, "amr", amr.id, ev.AMR_DEPART_EDGE, self.graph.edge_id(u, v),
                   {"edge": self.graph.edge_id(u, v), "speed_mps": speed,
                    "order": order.id if (order is not None and amr.purpose != _TO_CHARGE) else None})
        amr.battery_m -= pool["length"]
        self._push(t + pool["length"] / (speed * 60.0), _K_EDGE_DONE, (amr, v, pi))

    def _on_edge_done(self, t: float, data) -> None:
        amr, v, pi = data
        amr.node = v
        self.pool_occ[pi] -= 1
        # release: admit queued AMRs while capacity allows
        q = self.pool_queue[pi]
        pool = self.graph.pools[pi]
        while q and self.pool_occ[pi] < pool["capacity"]:
            w = q.popleft()
            wu, wv, wpi = w.path[w.path_i]
            self._emit(t, "amr", w.id, ev.AMR_EXIT_QUEUE, self.graph.edge_id(wu, wv),
                       {"at": self.graph.edge_id(wu, wv)})
            self._enter_edge(t, w, wu, wv, wpi)

        amr.path_i += 1
        if amr.path_i < len(amr.path):
            self._try_enter_edge(t, amr)
        else:
            self._on_leg_complete(t, amr)

    def _on_leg_complete(self, t: float, amr: _AMR) -> None:
        if amr.purpose == _TO_PICK:
            self._station_arrive(t, amr, amr.order.pick)
        elif amr.purpose == _TO_PACK:
            self._station_arrive(t, amr, amr.order.pack)
        else:  # _TO_CHARGE
            self._station_arrive(t, amr, amr.station)

    def _station_arrive(self, t: float, amr: _AMR, station: _Station) -> None:
        amr.station = station
        if station.busy < station.slots:
            self._begin_service(t, amr, station)
        else:
            station.queue.append(amr)
            self._emit(t, "amr", amr.id, ev.AMR_ENTER_QUEUE, station.id,
                       {"at": station.id, "kind": "station",
                        "pos": len(station.queue) - 1})

    def _begin_service(self, t: float, amr: _AMR, station: _Station) -> None:
        station.busy += 1
        slot = station.free_slots.pop()
        if station.kind == "charge":
            pct = round(max(amr.battery_m, 0.0) / self.battery_cap * 100.0, 1)
            self._emit(t, "amr", amr.id, ev.CHARGE_START, station.id,
                       {"station": station.id, "battery_pct": pct})
            self._push(t + self.charge_minutes, _K_CHARGE_DONE, (amr, station, slot))
        else:
            order = amr.order
            z = order.z_pick if station.kind == "pick" else order.z_pack
            self._emit(t, "amr", amr.id, ev.SERVICE_START, station.id,
                       {"station": station.id, "order": order.id, "slot": slot})
            self._push(t + self._service_minutes(station, z),
                       _K_SERVICE_DONE, (amr, station, slot))

    def _release_slot(self, t: float, station: _Station, slot: int) -> None:
        station.busy -= 1
        station.free_slots.append(slot)
        if station.queue:
            nxt = station.queue.popleft()
            self._emit(t, "amr", nxt.id, ev.AMR_EXIT_QUEUE, station.id,
                       {"at": station.id})
            self._begin_service(t, nxt, station)

    def _on_service_done(self, t: float, data) -> None:
        amr, station, slot = data
        order = amr.order
        self._emit(t, "amr", amr.id, ev.SERVICE_END, station.id,
                   {"station": station.id, "order": order.id})
        self._release_slot(t, station, slot)
        amr.station = None
        if station.kind == "pick":
            self._start_leg(t, amr, order.pack.node, _TO_PACK)
        else:  # pack -> order complete
            order.completed = True
            self.orders_completed_total += 1
            self._emit(t, "order", order.id, ev.ORDER_COMPLETE, station.id,
                       {"latency_min": round(t - order.t_arrival, 4)})
            amr.order = None
            self._after_task(t, amr)

    def _on_charge_done(self, t: float, data) -> None:
        amr, station, slot = data
        amr.battery_m = self.battery_cap
        self._emit(t, "amr", amr.id, ev.CHARGE_END, station.id,
                   {"station": station.id})
        self._release_slot(t, station, slot)
        amr.station = None
        self._after_task(t, amr)

    def _after_task(self, t: float, amr: _AMR) -> None:
        if amr.battery_m < self.charge_threshold * self.battery_cap:
            g = self.graph
            station = min(self.charges,
                          key=lambda c: (g.shortest_dist(amr.node, c.node), c.id))
            amr.station = station
            self._start_leg(t, amr, station.node, _TO_CHARGE)
        else:
            amr.idle = True
            self.idle_amrs.add(amr.idx)
            self._try_assign(t)
