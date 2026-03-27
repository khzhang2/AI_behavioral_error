from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class Alternative:
    alternative_id: str
    alternative_name: str
    default_label: str
    travel_time_min: float
    access_time_min: float
    egress_time_min: float
    parking_search_time_min: float
    availability_pct: float
    cost_eur: float
    scheme: str
    engine: str
    range_km: float


@dataclass
class ChoiceTask:
    task_id: str
    task_index: int
    trip_length_km: float
    alternatives: List[Alternative]


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _clean_numeric(value: object) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def load_choice_tasks(frame: pd.DataFrame) -> List[ChoiceTask]:
    tasks: List[ChoiceTask] = []
    grouped = frame.sort_values(["task_index", "default_label"]).groupby("task_id", sort=False)

    for task_id, task_frame in grouped:
        first_row = task_frame.iloc[0]
        alternatives: List[Alternative] = []
        for _, row in task_frame.iterrows():
            alternatives.append(
                Alternative(
                    alternative_id=str(row["alternative_id"]),
                    alternative_name=str(row["alternative_name"]),
                    default_label=str(row["default_label"]),
                    travel_time_min=_clean_numeric(row["travel_time_min"]),
                    access_time_min=_clean_numeric(row["access_time_min"]),
                    egress_time_min=_clean_numeric(row["egress_time_min"]),
                    parking_search_time_min=_clean_numeric(row["parking_search_time_min"]),
                    availability_pct=_clean_numeric(row["availability_pct"]),
                    cost_eur=_clean_numeric(row["cost_eur"]),
                    scheme=_clean_text(row["scheme"]),
                    engine=_clean_text(row["engine"]),
                    range_km=_clean_numeric(row["range_km"]),
                )
            )

        tasks.append(
            ChoiceTask(
                task_id=task_id,
                task_index=int(first_row["task_index"]),
                trip_length_km=float(first_row["trip_length_km"]),
                alternatives=alternatives,
            )
        )

    return tasks
