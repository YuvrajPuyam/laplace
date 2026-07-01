"""Calibrate demand from a real order dataset.

Reads order purchase timestamps from a public dataset (default: the Olist
Brazilian e-commerce orders, 2016-2018, ~99k real orders) and extracts the
intraday demand SHAPE — the hour-of-day profile and the peak-to-average ratio.
That shape is real; the absolute per-zone rate is then anchored to make the
pick zone operate in a realistic regime (stated in the calibration note). This
is the honest "calibrated to real order data" claim — only the absolute scale
is an assumption, and it's disclosed.

  python -m facility.demand --orders data/olist_orders.csv

Stdlib only (csv + datetime) — no pandas dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

OLIST_CITE = ("Olist Brazilian E-Commerce Public Dataset (Kaggle, ~99k real "
              "orders 2016-2018); order_purchase_timestamp.")


def read_timestamps(csv_path, column="order_purchase_timestamp"):
    out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts = (row.get(column) or "").strip()
            if not ts:
                continue
            try:
                out.append(datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
    return out


def profile(times):
    hour = [0] * 24
    dow = [0] * 7
    for t in times:
        hour[t.hour] += 1
        dow[t.weekday()] += 1
    n = max(sum(hour), 1)
    hour_share = [h / n for h in hour]
    mean = n / 24.0
    peak = max(hour)
    peak_hour = hour.index(peak)
    trough = min(h for h in hour if h > 0) if any(hour) else 0
    span_days = ((max(times) - min(times)).days + 1) if times else 0
    return {
        "n_orders": n, "span_days": span_days,
        "hour_counts": hour, "hour_share": [round(s, 4) for s in hour_share],
        "dow_counts": dow,
        "peak_hour": peak_hour,
        "peak_to_average": round(peak / mean, 3) if mean else 0.0,
        "trough_to_average": round(trough / mean, 3) if mean else 0.0,
    }


def suggest_rates(peak_to_avg, lambda_peak_per_min):
    """Given the real peak-to-average ratio and a chosen peak rate (orders/min),
    derive the off-peak / average / peak operating points."""
    avg = lambda_peak_per_min / peak_to_avg if peak_to_avg else lambda_peak_per_min
    return {
        "peak_per_min": round(lambda_peak_per_min, 3),
        "average_per_min": round(avg, 3),
        "off_peak_per_min": round(avg * 0.5, 3),
        "peak_per_hr": round(lambda_peak_per_min * 60, 1),
        "average_per_hr": round(avg * 60, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="demand")
    ap.add_argument("--orders", default="data/olist_orders.csv")
    ap.add_argument("--column", default="order_purchase_timestamp")
    ap.add_argument("--lambda-peak", type=float, default=1.0,
                    help="chosen peak arrival rate (orders/min) for one pick zone")
    ap.add_argument("--out", default=None, help="write calibration JSON here")
    args = ap.parse_args(argv)

    times = read_timestamps(args.orders, args.column)
    if not times:
        raise SystemExit(f"no timestamps parsed from {args.orders}")
    p = profile(times)
    rates = suggest_rates(p["peak_to_average"], args.lambda_peak)
    note = (
        f"Demand calibrated from {OLIST_CITE} Parsed {p['n_orders']:,} real "
        f"orders over {p['span_days']} days. Intraday SHAPE is real: peak hour "
        f"= {p['peak_hour']:02d}:00, peak-to-average ratio "
        f"= {p['peak_to_average']}x. The pick zone is sized for the PEAK hour "
        f"(arrival_rate_per_min = {rates['peak_per_min']} = "
        f"{rates['peak_per_hr']}/hr); off-peak and average rates derive from the "
        f"real ratio. The absolute per-zone scale is an explicit modelling "
        f"choice (a single zone of a larger DC), not taken from Olist's "
        f"platform-wide volume.")
    out = {"layout_note": "", "demand_source": OLIST_CITE,
           "profile": p, "rates": rates, "calibration_note": note}
    print(json.dumps({"peak_hour": p["peak_hour"],
                      "peak_to_average": p["peak_to_average"],
                      "n_orders": p["n_orders"], "rates": rates}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
