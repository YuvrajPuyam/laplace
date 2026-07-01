"""Rollout runner: one rollout, and multiprocessing sweeps across (config, seed).

run_rollout is the single authoritative path: simulate -> compute metrics from
the event rows -> write the parquet log -> validate the result against
Contract B.1 -> return it. The hot loop does no I/O; the parquet write happens
once at the end.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import events as ev
from .config import config_hash, fill_defaults, validate_config, validate_result
from .engine import Engine
from .metrics import compute_metrics


def run_rollout(config: dict, seed: int, log_dir: str | Path = "logs",
                write_log: bool = True) -> tuple[dict, list[ev.Row]]:
    """Run one (config, seed) rollout. Returns (result, event_rows).

    result["event_log_uri"] is relative ("logs/<hash>_s<seed>.events.parquet").
    """
    validate_config(config)
    cfg = fill_defaults(config)
    chash = config_hash(cfg)
    engine = Engine(cfg, seed)
    rows = engine.run()
    metrics = compute_metrics(rows, cfg, engine.graph)

    uri = f"{Path(log_dir).as_posix()}/{chash}_s{seed}.events.parquet"
    if write_log:
        ev.write_events(rows, uri)

    result = {
        "schema_version": "1.0",
        "config_hash": chash,
        "seed": seed,
        "metrics": metrics,
        "event_log_uri": uri,
    }
    validate_result(result)
    return result, rows


def _worker(args) -> dict:
    config, seed, log_dir, write_log = args
    result, _ = run_rollout(config, seed, log_dir, write_log)
    return result


def run_many(config: dict, seeds: list[int], log_dir: str | Path = "logs",
             write_log: bool = True, max_workers: int | None = None) -> list[dict]:
    """Run one config across many seeds in parallel. Order matches `seeds`."""
    jobs = [(config, s, str(log_dir), write_log) for s in seeds]
    if len(jobs) == 1:
        return [_worker(jobs[0])]
    if max_workers is None:                       # LAPLACE_MAX_WORKERS caps RAM on small boxes
        env = os.environ.get("LAPLACE_MAX_WORKERS")
        max_workers = int(env) if env else None
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_worker, jobs))
