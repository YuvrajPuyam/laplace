"""Named facility specs — real/literature-grounded footprints.

Dimensions and layout follow the multi-block parallel-aisle warehouse model of
Roodbergen & De Koster, the standard reference for AMR/picker pick zones. We
model a single AMR pick zone (Contract A caps the grid at 10 aisles x 60 m, so
a full DC is represented one zone at a time — stated honestly in provenance).
Arrival rates are calibrated from real order data in facility/demand.py.
"""

from __future__ import annotations

from .spec import FacilitySpec

ROODBERGEN = ("Roodbergen, K.J. & De Koster, R. (2001), 'Routing methods for "
              "warehouses with multiple cross aisles', Int. J. Production "
              "Research 39(9):1865-1883 — standard multi-block parallel-aisle "
              "layout. Aisle pitch 3.0 m (sim convention).")

OLIST_DEMAND = ("Olist Brazilian E-Commerce Public Dataset (~99,441 real orders, "
                "2016-2018). Intraday shape is real: peak hour 16:00, "
                "peak-to-average ratio 1.61 (see facility/demand.py, "
                "data/olist_calibration.json). Zone sized for the peak hour; "
                "absolute per-zone scale is a stated single-zone assumption.")

FACILITIES = {
    # A mid-size AMR pick zone: 9 aisles x 48 m, two blocks (front/mid/back
    # cross-aisles), 8 pick faces, 3 pack/ship at the front depot.
    "dc_pickzone_med": FacilitySpec(
        facility_id="dc_pickzone_med",
        name="Mid-size AMR pick zone (2-block, Roodbergen layout)",
        source=ROODBERGEN,
        notes="One pick zone of a larger DC. 9 parallel pick aisles x 48 m, "
              "3 cross-aisles (front/middle/back) = 2 storage blocks. ~24 x 48 m "
              "footprint. Pick faces distributed through the racks; pack/ship, "
              "chargers, and dock at the front depot. Contract A's 10-aisle / "
              "60 m cap means a full DC is modelled one zone at a time.",
        pick_aisles=9, aisle_length_m=48, cross_aisle_count=3,
        n_pick_stations=8, n_pack_stations=3, n_chargers=2,
        amr_count=6, speed_mps=1.5, battery_capacity_m=8000, charge_minutes=15,
        routing="shortest_path", pack_assignment="shortest_queue",
        arrival_rate_per_min=1.0, demand_source=OLIST_DEMAND,  # peak hour, calibrated
        sim_minutes=480, warmup_minutes=30),

    # A compact micro-fulfilment zone: 6 aisles x 30 m, single block.
    "mfc_compact": FacilitySpec(
        facility_id="mfc_compact",
        name="Compact micro-fulfilment zone (single-block, Roodbergen layout)",
        source=ROODBERGEN,
        notes="Urban micro-fulfilment centre pick zone. 6 aisles x 30 m, "
              "front/back cross-aisles only (single block). 5 pick faces, "
              "2 pack stations. ~15 x 30 m.",
        pick_aisles=6, aisle_length_m=30, cross_aisle_count=2,
        n_pick_stations=5, n_pack_stations=2, n_chargers=1,
        amr_count=4, speed_mps=1.5, battery_capacity_m=6000, charge_minutes=15,
        routing="shortest_path", pack_assignment="shortest_queue",
        arrival_rate_per_min=0.6, demand_source=OLIST_DEMAND,  # smaller zone, peak
        sim_minutes=480, warmup_minutes=30),
}
