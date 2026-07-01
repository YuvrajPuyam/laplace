"""Import smoke test guarding the engine's lazy cross-package imports.

engine.live_feed (and the live-twin path generally) imports renderer.* and
ui.* modules. Several of these are pulled in inside functions / lazily; a future
module move could silently break the engine at runtime without any test
catching it. This asserts every such target module resolves at import time.
"""

from __future__ import annotations

import importlib

import pytest

_LAZY_IMPORT_TARGETS = [
    "renderer.physx_stream",
    "renderer.avoidance",
    "ui.export_web_scene",
    "engine.live_feed",
]


@pytest.mark.parametrize("module_name", _LAZY_IMPORT_TARGETS)
def test_lazy_cross_package_import_resolves(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None
