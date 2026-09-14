from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_facilities(path: Path = CONFIG_DIR / "facilities.json") -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["facilities"]


def phase_facilities(phase: int = 1, path: Path = CONFIG_DIR / "facilities.json") -> list[str]:
    """Facility names included up to the given phase (phase 1 = the 8 most-inspected types)."""
    return [f["name"] for f in load_facilities(path) if f["phase"] <= phase]
