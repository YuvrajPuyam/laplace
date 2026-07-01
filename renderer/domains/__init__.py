"""Domain packs: each turns a domain's own config into a TwinScene.

A domain pack exposes `to_scene(config) -> TwinScene` and a `CATALOG` dict.
Register new domains here so tooling can discover them by name.
"""

from . import warehouse

REGISTRY = {
    "warehouse": warehouse,
}
