# sim/ — WS1 Tier-0 simulator

Implements the world model of spec §3.1 against frozen Contracts A and B.
This file is the **modeling-decision log**: choices the contracts left open,
made concrete here. Sim dynamics choices are Yuv's call (CLAUDE.md) — each
item below is a proposal that stands until overruled.

## Decisions the contracts left open

1. **Aisle spacing = 3.0 m** (`navgraph.AISLE_SPACING_M`). Contract A names
   nodes by position *along* an aisle but never states the distance *between*
   aisles. 3 m sets cross-aisle/extra-edge lengths and renderer geometry.
2. **Bidirectional edges are one capacity pool.** "Max simultaneous AMRs on
   the edge" is enforced across both directions jointly (an override on
   either direction id hits the pool). This is what makes the Braess
   capacity-1 shortcut contend. `one_way: true` restricts traversal to the
   stated from->to direction.
3. **Queued AMRs hold no capacity.** An AMR blocked on a full edge or
   station waits at the tail node, occupying nothing — so the network cannot
   deadlock (every edge drains in finite time). This matches the schema's
   "entrants beyond capacity queue at the tail node".
4. **Routing is computed per leg** (dispatch time), not per edge.
   `congestion_aware` snapshots pool occupancy at leg start with the
   contract's cost `length * (1 + 2.0 * occ/capacity)`. Mid-leg replanning:
   not in v1.
5. **Task allocation:** pending orders FIFO; oldest order gets the nearest
   idle AMR by *static* shortest-path distance (ties → lowest AMR index),
   under both routing policies.
6. **Battery:** drains per meter traveled; checked only when an AMR becomes
   idle (after pack or after charging). Below 15% → travel to nearest charge
   station, full recharge in `charge_minutes`. Battery never interrupts a
   task and idle AMRs don't drain. AMRs en route to / queued at / in charge
   are not assignable.
7. **AMRs idle in place** after completing a pack (no repositioning to dock).
8. **`shortest_queue` pack assignment** counts (in service + queued) at the
   station at arrival time; AMRs en route are not counted. Ties → first
   station in config order.
9. **Service-time pairing (CRN):** each order pre-draws `z_pick`/`z_pack`
   standard normals in the seed-only arrival stream; service at station S
   takes `exp(mu_S + sigma_S * z)` minutes whenever it happens. Arrival
   times, pick-station selection (via a uniform, robust to station-count
   changes), and service draws are therefore identical across configs at the
   same seed.

## Metric conventions (measured window = [warmup, sim_minutes])

- A **measured order** arrived at `t >= warmup`. `orders_completed`,
  latency percentiles, and throughput cover measured orders completed by sim
  end; `orders_abandoned` = measured − completed (includes in-flight at sim
  end, matching Contract B.1's "still incomplete"). Conservation:
  arrived = completed + abandoned, always.
- The `sim_end` payload's `orders_abandoned` counts ALL orders incomplete at
  sim end (warmup included) — it is informational; the metric is computed
  from order events over the window.
- **deadhead_pct** = % of travel distance while not *carrying*: legs toward
  a pick (before that order's pick `service_start`) plus charge legs.
  Travel to pick has `order` set in the depart payload; charge legs have
  `order: null`.
- **amr_utilization_pct** = fleet-mean fraction of the window between
  `task_assigned` and `order_complete` (clipped). Charging is downtime.
- **station_wait** samples are 0.0 for AMRs that found a free slot.
- **edge_congestion** occupancy integrates depart-event durations
  (`length / speed`, deterministic) against `window × pool_capacity`.

## Performance

No I/O in the hot loop; the parquet log is written once per rollout.
Rollouts parallelize across seeds via `runner.run_many` (ProcessPool).
Gate: ≥500× realtime per rollout on one core (`tests/test_perf.py`).
