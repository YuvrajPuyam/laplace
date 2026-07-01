"""facility/ — ground the twin in a real, citable facility footprint.

A FacilitySpec describes a real (or literature-standard) warehouse at the level
a logistics engineer would: number of pick aisles, aisle length in meters,
cross-aisle count, pick/pack/charge/dock placement, fleet, and a *cited source*.
`facility_to_config` lowers it to a frozen Contract A config (no schema change),
attaching a provenance record. This is what moves the project from a synthetic
6-aisle grid to "a deterministic decision twin of a real DC footprint."
"""

from .spec import FacilitySpec, assemble_config, facility_to_config  # noqa: F401
from .facilities import FACILITIES  # noqa: F401
from .usd_extractor import (  # noqa: F401
    ExtractedLayout,
    ExtractError,
    RackCloud,
    extract_layout,
    load_cloud,
    scenario_from_cloud,
    scenario_from_layout,
)
