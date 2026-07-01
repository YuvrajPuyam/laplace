"""Asset catalog: semantic type -> how to render it.

This is the second half of domain-extensibility. A TwinScene carries semantic
type strings ("amr", "pick_station", "bed", "gate", ...); the catalog maps
each to concrete geometry. v1 uses clean primitives with PBR-ish colors —
fast to render on an 8 GB laptop and zero asset downloads. To upgrade fidelity
or add a domain, point a type at a `usd` reference instead of a `prim`; the
Isaac builder resolves `usd` by referencing the asset, `prim` by drawing a
shape. Nothing else changes.

A domain provides a catalog dict (TYPE -> AssetSpec). build_stage merges the
domain catalog over DEFAULT_CATALOG, so domains only specify what differs.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class AssetSpec:
    # render mode
    kind: str                       # "prim" | "usd"
    # prim mode:
    shape: str = "box"              # box | cylinder | cone | sphere
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)  # meters (x, y, z)
    color: tuple[float, float, float] = (0.7, 0.7, 0.7)  # linear RGB 0..1
    metallic: float = 0.0
    roughness: float = 0.6
    # usd mode:
    usd_path: str = ""              # asset reference (file or nucleus URL)
    usd_scale: float = 1.0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AssetSpec":
        return cls(kind=d.get("kind", "prim"), shape=d.get("shape", "box"),
                   size=tuple(d.get("size", (1.0, 1.0, 1.0))),
                   color=tuple(d.get("color", (0.7, 0.7, 0.7))),
                   metallic=d.get("metallic", 0.0), roughness=d.get("roughness", 0.6),
                   usd_path=d.get("usd_path", ""), usd_scale=d.get("usd_scale", 1.0))


def catalog_to_dict(catalog: dict[str, "AssetSpec"]) -> dict:
    """Serialize a domain catalog for embedding in a TwinScene."""
    return {k: v.as_dict() for k, v in catalog.items()}


def catalog_from_dict(raw: dict) -> dict[str, "AssetSpec"]:
    return {k: AssetSpec.from_dict(v) for k, v in (raw or {}).items()}


# Warehouse domain catalog. Colors chosen to read clearly under RTX:
# amrs warm orange, pick blue, pack green, charger amber, dock gray, shelf tan.
WAREHOUSE_CATALOG: dict[str, AssetSpec] = {
    "amr": AssetSpec("prim", "box", (0.7, 0.9, 0.35), (0.90, 0.45, 0.10),
                     metallic=0.1, roughness=0.5),
    "pick_station": AssetSpec("prim", "box", (1.0, 1.0, 1.6), (0.20, 0.50, 0.85),
                              roughness=0.55),
    "pack_station": AssetSpec("prim", "box", (1.2, 1.2, 1.4), (0.30, 0.60, 0.20),
                              roughness=0.55),
    "charger": AssetSpec("prim", "box", (0.6, 0.6, 1.0), (0.95, 0.65, 0.10),
                         metallic=0.2, roughness=0.4),
    "dock": AssetSpec("prim", "box", (1.4, 0.4, 0.6), (0.55, 0.55, 0.55),
                      roughness=0.7),
}

# Generic fallbacks so an unknown type still renders (never a silent gap).
DEFAULT_CATALOG: dict[str, AssetSpec] = {
    "_agent_default": AssetSpec("prim", "box", (0.6, 0.6, 0.4), (0.85, 0.45, 0.1)),
    "_prop_default": AssetSpec("prim", "box", (1.0, 1.0, 1.2), (0.5, 0.5, 0.55)),
}

# Realistic warehouse: real NVIDIA Isaac assets. Verified to exist in the
# Isaac 5.1 asset library (renderer/probe_assets.py). UNIT GUARD: Simple_
# Warehouse SM_* props are authored in centimetres -> usd_scale 0.01; robots
# and /Isaac/Props assets are metres -> usd_scale 1.0.
_PROPS = "/Isaac/Environments/Simple_Warehouse/Props"
IWHUB = "/Isaac/Robots/Idealworks/iwhub/iw_hub.usd"
KLT_TOTE = "/Isaac/Props/KLT_Bin/small_KLT_visual.usd"   # metres
RACK = f"{_PROPS}/SM_RackPile_06.usd"                    # metres (Isaac default)
WALL = f"{_PROPS}/SM_WallA_6M.usd"

WAREHOUSE_REALISTIC_CATALOG: dict[str, AssetSpec] = {
    **WAREHOUSE_CATALOG,
    "amr": AssetSpec("usd", usd_path=IWHUB, usd_scale=1.0),
    "payload": AssetSpec("usd", usd_path=KLT_TOTE, usd_scale=1.0),
    "pick_station": AssetSpec("usd", usd_path=RACK, usd_scale=1.0),
    # dressing rack VARIANTS (build_stage --dress cycles all keys starting
    # "_rack" to break uniformity); different fills/heights for variety.
    "_rack": AssetSpec("usd", usd_path=f"{_PROPS}/SM_RackPile_06.usd", usd_scale=1.0),
    "_rack2": AssetSpec("usd", usd_path=f"{_PROPS}/SM_RackPile_04.usd", usd_scale=1.0),
    "_rack3": AssetSpec("usd", usd_path=f"{_PROPS}/SM_RackPile_03.usd", usd_scale=1.0),
    "_rack4": AssetSpec("usd", usd_path=f"{_PROPS}/SM_RackShelf_01.usd", usd_scale=1.0),
    "_wall": AssetSpec("usd", usd_path=WALL, usd_scale=1.0),
}


def resolve(type_key: str, domain_catalog: dict[str, AssetSpec],
            is_agent: bool) -> AssetSpec:
    if type_key in domain_catalog:
        return domain_catalog[type_key]
    if type_key in DEFAULT_CATALOG:
        return DEFAULT_CATALOG[type_key]
    return DEFAULT_CATALOG["_agent_default" if is_agent else "_prop_default"]
