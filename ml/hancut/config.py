from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_items(path: Path = CONFIG_DIR / "items.json") -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def phase_items(phase: int = 1, path: Path = CONFIG_DIR / "items.json") -> list[str]:
    """Prohibited item names up to the given phase (phase 1 = the 5 shared by PIDray and SIXray)."""
    return [i["name"] for i in load_items(path) if i["phase"] <= phase]


def items_in(dataset: str, path: Path = CONFIG_DIR / "items.json") -> list[str]:
    """Item names annotated in the given dataset, e.g. "pidray"."""
    return [i["name"] for i in load_items(path) if dataset in i["datasets"]]
