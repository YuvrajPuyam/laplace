"""Event log (Contract B.2): row construction, parquet writer/reader.

A row is the tuple (t, entity_type, entity_id, event, location, payload_json).
Rows are appended strictly in processing order; the writer never re-sorts —
the engine guarantees non-decreasing t, and ties keep insertion order (stable,
determinism requires it).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Closed event enum (events.schema.md). Adding one requires a schema bump.
ORDER_ARRIVED = "order_arrived"
TASK_ASSIGNED = "task_assigned"
AMR_DEPART_EDGE = "amr_depart_edge"
AMR_ENTER_QUEUE = "amr_enter_queue"
AMR_EXIT_QUEUE = "amr_exit_queue"
SERVICE_START = "service_start"
SERVICE_END = "service_end"
ORDER_COMPLETE = "order_complete"
CHARGE_START = "charge_start"
CHARGE_END = "charge_end"
SIM_WARMUP_END = "sim_warmup_end"
SIM_END = "sim_end"

EVENT_ENUM = [
    ORDER_ARRIVED, TASK_ASSIGNED, AMR_DEPART_EDGE, AMR_ENTER_QUEUE,
    AMR_EXIT_QUEUE, SERVICE_START, SERVICE_END, ORDER_COMPLETE,
    CHARGE_START, CHARGE_END, SIM_WARMUP_END, SIM_END,
]

ENTITY_TYPES = ["amr", "order", "station", "sim"]

_SCHEMA = pa.schema([
    ("t", pa.float64()),
    ("entity_type", pa.dictionary(pa.int8(), pa.string())),
    ("entity_id", pa.string()),
    ("event", pa.dictionary(pa.int8(), pa.string())),
    ("location", pa.string()),
    ("payload", pa.string()),
])

Row = tuple  # (t, entity_type, entity_id, event, location, payload)


def write_events(rows: list[Row], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(zip(*rows)) if rows else [[], [], [], [], [], []]
    table = pa.table(
        {
            "t": pa.array(cols[0], pa.float64()),
            "entity_type": pa.array(cols[1], pa.string()).dictionary_encode(),
            "entity_id": pa.array(cols[2], pa.string()),
            "event": pa.array(cols[3], pa.string()).dictionary_encode(),
            "location": pa.array(cols[4], pa.string()),
            "payload": pa.array(cols[5], pa.string()),
        },
        schema=_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")


def read_events(path: str | Path) -> list[Row]:
    table = pq.read_table(path)
    cols = [table.column(name).to_pylist() for name in
            ("t", "entity_type", "entity_id", "event", "location", "payload")]
    return list(zip(*cols))
