from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

CHOICE_LABEL_TO_CODE = {"A": 1, "B": 2, "C": 3}
CHOICE_CODE_TO_NAME = {1: "TRAIN", 2: "SWISSMETRO", 3: "CAR"}
CHOICE_LABEL_TO_NAME = {"A": "TRAIN", "B": "SWISSMETRO", "C": "CAR"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def weighted_sample_one(frame: pd.DataFrame, rng, weight_col: str) -> pd.Series:
    weights = frame[weight_col].astype(float).to_numpy()
    weights = weights / weights.sum()
    sampled_index = rng.choice(frame.index.to_numpy(), p=weights)
    return frame.loc[sampled_index]


def reconstructed_value(baseline: float, multiplier: float, zero_if_nonpositive: bool = False) -> int:
    if zero_if_nonpositive and baseline <= 0:
        return 0
    return int(round(float(baseline) * float(multiplier)))
