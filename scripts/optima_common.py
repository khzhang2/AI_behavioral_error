from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT_DIR / "experiment_config.json").read_text(encoding="utf-8"))
DATA_DIR = ROOT_DIR / CONFIG["paths"]["data_dir"]
EXPERIMENT_DIR = ROOT_DIR / CONFIG["paths"]["archive_dir"]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
AI_COLLECTION_DIR = DATA_DIR / CONFIG["paths"].get("ai_collection_subdir", "ai_collection_qwen3.5_9b")

INDICATOR_NAMES = ["Envir01", "Mobil05", "LifSty07", "Envir05", "Mobil12", "LifSty01"]
INDICATOR_TEXT = {
    "Envir01": "Fuel prices should be increased to reduce congestion and air pollution.",
    "Mobil05": "I reconsider frequently my mode choice.",
    "LifSty07": "The pleasure of having something beautiful consists in showing it.",
    "Envir05": "I am concerned about global warming.",
    "Mobil12": "It is very important to have a beautiful car.",
    "LifSty01": "I always choose the best products, regardless of price.",
}
CHOICE_LABEL_TO_CODE = {"A": 0, "B": 1, "C": 2}
CHOICE_CODE_TO_NAME = {0: "PT", 1: "CAR", 2: "SLOW_MODES"}
CHOICE_LABEL_TO_NAME = {"A": "PT", "B": "CAR", "C": "SLOW_MODES"}
DRAW_NAMES = ["omega_car", "omega_env"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def archive_experiment_config(trial_dir: Path | None = None) -> Path:
    source = ROOT_DIR / "experiment_config.json"
    target_dir = ensure_dir(EXPERIMENT_DIR if trial_dir is None else trial_dir)
    target = target_dir / "experiment_config.json"
    if not target.exists():
        shutil.copy2(source, target)
        return target

    index = 2
    while True:
        candidate = target_dir / f"experiment_config_{index}.json"
        if not candidate.exists():
            shutil.copy2(source, candidate)
            return candidate
        index += 1


def infer_trial_dir_from_output_dir(output_dir: Path) -> Path | None:
    for parent in [output_dir, *output_dir.parents]:
        if parent.name == "outputs":
            return parent.parent
    return None


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_json_payload(text: str) -> dict[str, Any]:
    stripped = strip_code_fence(text)
    if not stripped.startswith("{"):
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_indicator_value(text: str) -> int:
    payload = parse_json_payload(text)
    for key in ("indicator_value", "value", "response"):
        if key in payload:
            try:
                value = int(payload[key])
            except (TypeError, ValueError):
                value = -1
            return value if 1 <= value <= 6 else -1
    match = re.search(r"\b([1-6])\b", text)
    return int(match.group(1)) if match else -1


def parse_choice_label(text: str) -> str:
    payload = parse_json_payload(text)
    for key in ("choice_label", "choice", "answer"):
        if key in payload:
            label = str(payload[key]).strip().upper()
            if label in CHOICE_LABEL_TO_CODE:
                return label
            if label in {"PT", "CAR", "SLOW_MODES"}:
                return {"PT": "A", "CAR": "B", "SLOW_MODES": "C"}[label]
    match = re.search(r"\b([A-C])\b", text.upper())
    if match:
        return match.group(1)
    for name, label in {"PT": "A", "CAR": "B", "SLOW_MODES": "C"}.items():
        if re.search(rf"\b{name}\b", text.upper()):
            return label
    return ""


def total_variation_distance(left: pd.Series, right: pd.Series) -> float:
    levels = sorted(set(left.index).union(set(right.index)))
    return 0.5 * sum(abs(float(left.get(level, 0.0)) - float(right.get(level, 0.0))) for level in levels)


def generate_shared_sobol_draws(n_rows: int, n_draws: int, n_dims: int, seed: int) -> np.ndarray:
    n_points = n_rows * n_draws
    m = math.ceil(math.log2(max(2, n_points)))
    sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
    uniforms = sampler.random_base2(m=m)[:n_points]
    uniforms = np.clip(uniforms, 1e-10, 1 - 1e-10)
    return norm.ppf(uniforms).reshape(n_rows, n_draws, n_dims)


def draw_generator_from_file(draw_path: Path, dim_index: int):
    draw_cache: dict[tuple[int, int], np.ndarray] = {}

    def generator(sample_size: int, number_of_draws: int) -> np.ndarray:
        key = (sample_size, number_of_draws)
        if key not in draw_cache:
            draws = np.load(draw_path)
            draw_cache[key] = draws[:sample_size, :number_of_draws, dim_index]
        return draw_cache[key]

    return generator


def likert_probability_numpy(observed: np.ndarray, index_value: np.ndarray, delta_1: float, delta_2: float) -> np.ndarray:
    tau_1 = delta_1
    tau_2 = delta_1 + delta_2
    thresholds = np.array([-tau_2, -tau_1, 0.0, tau_1, tau_2], dtype=float)
    upper = np.empty_like(index_value)
    lower = np.empty_like(index_value)
    observed_zero = observed.astype(int) - 1
    for category in range(6):
        mask = observed_zero == category
        if not np.any(mask):
            continue
        upper_cut = thresholds[category] if category < 5 else np.inf
        lower_cut = thresholds[category - 1] if category > 0 else -np.inf
        upper[mask] = np.where(np.isfinite(upper_cut), norm.cdf(upper_cut - index_value[mask]), 1.0)
        lower[mask] = np.where(np.isfinite(lower_cut), norm.cdf(lower_cut - index_value[mask]), 0.0)
    return np.clip(upper - lower, 1e-30, 1.0)
