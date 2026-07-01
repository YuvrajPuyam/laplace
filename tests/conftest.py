import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def baseline_config() -> dict:
    with open(ROOT / "examples" / "baseline_small.config.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def braess_patch() -> dict:
    with open(ROOT / "examples" / "braess_patch.json", encoding="utf-8") as f:
        return json.load(f)
