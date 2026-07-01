# Event log format (Contract B.2), FROZEN

One parquet file per rollout. This single format has **four consumers**, metrics
computation, the Isaac/Three.js replay renderer, the live rollout board, and
side-by-side comparison, so it is never forked and never extended silently.

## Columns

| column | type | description |
|---|---|---|
| `t` | float64 | Sim time in minutes from rollout start (warmup included; consumers filter) |
| `entity_type` | category | `amr` \| `order` \| `station` \| `sim` |
| `entity_id` | string | `amr_03`, `ord_000142`, `P1`, `sim` |
| `event` | category | One of the closed enum below |
| `location` | string | Node id, edge id, or station id relevant to the event; empty if n/a |
| `payload` | string (JSON) | Event-specific fields per the enum table; compact JSON, no nesting beyond one level |

Rows are strictly ordered by `t`, ties broken by a monotonically increasing
implicit row index (stable sort, determinism requires it).

## Event enum (CLOSED, adding an event requires a schema_version bump)

| event | entity_type | required payload | emitted when |
|---|---|---|---|
| `order_arrived` | order | `{"pick": "P1", "pack": "K2"}` | Poisson arrival; pack chosen by policy at arrival time |
| `task_assigned` | order | `{"amr": "amr_03"}` | Nearest-idle-AMR allocation fires |
| `amr_depart_edge` | amr | `{"edge": "A3_15->A3_16", "speed_mps": 1.5, "order": "ord_000142" \| null}` | AMR begins traversing an edge. **Replay key event**, see sufficiency rule |
| `amr_enter_queue` | amr | `{"at": "P1", "kind": "station" \| "edge", "pos": 2}` | AMR blocked: station slots full or edge at capacity. `pos` = queue index (0-based) |
| `amr_exit_queue` | amr | `{"at": "P1"}` | AMR unblocked |
| `service_start` | amr | `{"station": "P1", "order": "ord_000142", "slot": 0}` | Slot acquired |
| `service_end` | amr | `{"station": "P1", "order": "ord_000142"}` | Lognormal service time elapsed |
| `order_complete` | order | `{"latency_min": 7.4}` | Pack service ends |
| `charge_start` | amr | `{"station": "C1", "battery_pct": 12.1}` | |
| `charge_end` | amr | `{"station": "C1"}` | |
| `sim_warmup_end` | sim | `{}` | Exactly once, at `t = warmup_minutes` |
| `sim_end` | sim | `{"orders_abandoned": 3}` | Exactly once, last row |

## Replay sufficiency rule (binding on WS1, consumed by WS5/WS6)

A renderer must be able to reconstruct every AMR's position at any time `t`
**from this log plus the config alone**:

- Between `amr_depart_edge` at `t0` and its next event, position = linear
  interpolation along the edge's geometry at `speed_mps` from `t0`, clamped at
  the edge's far node. Edge geometry derives deterministically from the config
  (grid generation rules in Contract A).
- While queued: parked at the queue anchor of `at`, offset by `pos` × 0.8 m
  along the approach edge.
- While in service / charging: parked at the station node, offset by `slot`.
- Before its first event: parked at the dock (AMRs start at dock D1, slot-offset
  by index).

WS1 must include a unit test that replays a log through this rule and asserts
no AMR ever teleports more than `speed_mps × Δt` between consecutive samples.

## Size discipline

Target < 5 MB per 480-min rollout at default scale. If exceeded, drop nothing -
reduce by coarsening only `amr_depart_edge` granularity (multi-edge legs with a
`path` payload) behind a schema_version bump. Do not silently sample.
