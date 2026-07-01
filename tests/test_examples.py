"""schemas/README.md: examples/ instances are validated in CI; failing
examples fail the build."""

import json
from pathlib import Path

from sim.config import validate_config, validate_result

ROOT = Path(__file__).resolve().parent.parent


def test_example_config_valid():
    with open(ROOT / "examples" / "baseline_small.config.json", encoding="utf-8") as f:
        validate_config(json.load(f))


def test_example_result_valid():
    with open(ROOT / "examples" / "rollout_result.json", encoding="utf-8") as f:
        validate_result(json.load(f))


def test_example_braess_patch_applies(baseline_config, braess_patch):
    from sim.config import apply_patch
    validate_config(apply_patch(baseline_config, braess_patch["patch"]))
