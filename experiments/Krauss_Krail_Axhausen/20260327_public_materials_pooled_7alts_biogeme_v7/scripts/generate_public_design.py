from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "pooled_choice_sets_public_materials.csv"

ALL_ALTERNATIVES = [
    "e_scooter",
    "bikesharing",
    "walking",
    "private_car",
    "carsharing",
    "ridepooling",
    "public_transport",
]

ALTERNATIVE_NAMES = {
    "e_scooter": "E-Scooter Sharing",
    "bikesharing": "Bike Sharing",
    "walking": "Walking",
    "private_car": "Private Car",
    "carsharing": "Car Sharing",
    "ridepooling": "Ridepooling",
    "public_transport": "Public Transport",
}

ACTIVE_ALTERNATIVES = {
    "SD": ["e_scooter", "bikesharing", "walking", "private_car"],
    "MD": ["carsharing", "ridepooling", "public_transport", "private_car"],
}

DISPLAY_LABELS = {
    "SD": {
        "e_scooter": "A",
        "bikesharing": "B",
        "walking": "C",
        "private_car": "D",
    },
    "MD": {
        "carsharing": "A",
        "ridepooling": "B",
        "public_transport": "C",
        "private_car": "D",
    },
}

SALT = {
    "e_scooter": 1,
    "bikesharing": 2,
    "walking": 3,
    "private_car": 4,
    "carsharing": 5,
    "ridepooling": 6,
    "public_transport": 7,
}

SD_LENGTH_BUCKETS = {
    0.5: [0, 1],
    1.0: [1, 2, 3],
    2.0: [2, 3, 4],
    4.0: [4, 5],
}

MD_LENGTH_BUCKETS = {
    2.0: [0, 1],
    5.0: [1, 2, 3],
    10.0: [2, 3, 4],
    20.0: [4, 5],
}

SD_COST_BUCKETS = {
    0.5: [0, 1],
    1.0: [1, 2],
    2.0: [2, 3],
    4.0: [4, 5],
}

MD_COST_BUCKETS = {
    2.0: [0, 1],
    5.0: [1, 2],
    10.0: [2, 3, 4],
    20.0: [4, 5],
}


def parse_level(value: str) -> object:
    text = str(value).strip()
    if text == "":
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    return int(numeric) if numeric.is_integer() else numeric


def load_levels(path: Path) -> dict[str, dict[str, list[object]]]:
    frame = pd.read_csv(path)
    nested: dict[str, dict[str, list[object]]] = {}
    for (alternative_id, attribute_name), subset in frame.groupby(["alternative_id", "attribute_name"], sort=False):
        nested.setdefault(alternative_id, {})[attribute_name] = [parse_level(value) for value in subset["level_value"]]
    return nested


def choose_cycled(levels: list[object], block_id: int, task_in_block: int, salt: int) -> object:
    index = (block_id * 5 + task_in_block * 3 + salt) % len(levels)
    return levels[index]


def choose_length_aligned(
    levels: list[object],
    block_id: int,
    task_in_block: int,
    salt: int,
    trip_length: float,
    buckets: dict[float, list[int]],
) -> object:
    candidates = buckets[float(trip_length)]
    index = candidates[(block_id + task_in_block + salt) % len(candidates)]
    return levels[index]


def availability_flags(subsample: str) -> dict[str, int]:
    active = set(ACTIVE_ALTERNATIVES[subsample])
    return {
        "avail_es": int("e_scooter" in active),
        "avail_bs": int("bikesharing" in active),
        "avail_walk": int("walking" in active),
        "avail_car": int("private_car" in active),
        "avail_cs": int("carsharing" in active),
        "avail_rp": int("ridepooling" in active),
        "avail_pt": int("public_transport" in active),
    }


def base_row(subsample: str, block_id: int, task_in_block: int, trip_length: float, alternative_id: str) -> dict:
    return {
        "subsample": subsample,
        "block_id": block_id,
        "task_in_block": task_in_block,
        "task_index_within_subsample": (block_id - 1) * 6 + task_in_block,
        "task_id": f"{subsample}_B{block_id:02d}_T{task_in_block:02d}",
        "alternative_id": alternative_id,
        "alternative_name": ALTERNATIVE_NAMES[alternative_id],
        "display_label": DISPLAY_LABELS[subsample].get(alternative_id, ""),
        "trip_length_km": trip_length,
        "is_available": int(alternative_id in ACTIVE_ALTERNATIVES[subsample]),
        "travel_time_min": None,
        "access_time_min": None,
        "waiting_time_min": None,
        "egress_time_min": None,
        "detour_time_min": None,
        "parking_search_time_min": None,
        "availability_pct": None,
        "cost_eur": None,
        "scheme": "",
        "engine": "",
        "range_km": None,
        "crowding_pct": None,
        "transfer_count": None,
        "scheme_free_floating": 0,
        "scheme_hybrid": 0,
        "pedelec": 0,
        "provenance": "inferred_from_public",
        "source_note": "Task-level combination inferred from public attribute levels; underlying level values are public exact.",
        **availability_flags(subsample),
    }


def build_sd_attributes(levels: dict[str, dict[str, list[object]]], block_id: int, task_in_block: int, trip_length: float) -> dict[str, dict]:
    rows: dict[str, dict] = {}

    es = {
        "travel_time_min": choose_length_aligned(
            levels["e_scooter"]["travel_time_min"], block_id, task_in_block, SALT["e_scooter"], trip_length, SD_LENGTH_BUCKETS
        ),
        "access_time_min": choose_cycled(levels["e_scooter"]["access_time_min"], block_id, task_in_block, 11),
        "egress_time_min": choose_cycled(levels["e_scooter"]["egress_time_min"], block_id, task_in_block, 13),
        "parking_search_time_min": choose_cycled(levels["e_scooter"]["parking_search_time_min"], block_id, task_in_block, 17),
        "availability_pct": choose_cycled(levels["e_scooter"]["availability_pct"], block_id, task_in_block, 19),
        "cost_eur": choose_length_aligned(
            levels["e_scooter"]["cost_eur"], block_id, task_in_block, 23, trip_length, SD_COST_BUCKETS
        ),
        "scheme": choose_cycled(levels["e_scooter"]["scheme"], block_id, task_in_block, 29),
        "engine": "",
        "range_km": choose_cycled(levels["e_scooter"]["range_km"], block_id, task_in_block, 31),
    }
    es["scheme_free_floating"] = int(es["scheme"] == "free-floating")
    rows["e_scooter"] = es

    bs_engine = choose_cycled(levels["bikesharing"]["engine"], block_id, task_in_block, 37)
    bs_range = choose_cycled(levels["bikesharing"]["range_km"], block_id, task_in_block, 41) if bs_engine == "pedelec" else None
    bs = {
        "travel_time_min": choose_length_aligned(
            levels["bikesharing"]["travel_time_min"], block_id, task_in_block, SALT["bikesharing"], trip_length, SD_LENGTH_BUCKETS
        ),
        "access_time_min": choose_cycled(levels["bikesharing"]["access_time_min"], block_id, task_in_block, 43),
        "egress_time_min": choose_cycled(levels["bikesharing"]["egress_time_min"], block_id, task_in_block, 47),
        "parking_search_time_min": choose_cycled(levels["bikesharing"]["parking_search_time_min"], block_id, task_in_block, 53),
        "availability_pct": choose_cycled(levels["bikesharing"]["availability_pct"], block_id, task_in_block, 59),
        "cost_eur": choose_length_aligned(
            levels["bikesharing"]["cost_eur"], block_id, task_in_block, 61, trip_length, SD_COST_BUCKETS
        ),
        "scheme": choose_cycled(levels["bikesharing"]["scheme"], block_id, task_in_block, 67),
        "engine": bs_engine,
        "range_km": bs_range,
    }
    bs["scheme_free_floating"] = int(bs["scheme"] == "free-floating")
    bs["pedelec"] = int(bs_engine == "pedelec")
    rows["bikesharing"] = bs

    rows["walking"] = {
        "travel_time_min": choose_length_aligned(
            levels["walking"]["travel_time_min"], block_id, task_in_block, SALT["walking"], trip_length, SD_LENGTH_BUCKETS
        ),
        "access_time_min": 0,
        "egress_time_min": 0,
        "parking_search_time_min": 0,
        "availability_pct": None,
        "cost_eur": None,
        "scheme": "",
        "engine": "",
        "range_km": None,
    }

    rows["private_car"] = {
        "travel_time_min": choose_length_aligned(
            levels["private_car"]["travel_time_min"], block_id, task_in_block, SALT["private_car"], trip_length, SD_LENGTH_BUCKETS
        ),
        "access_time_min": choose_cycled(levels["private_car"]["access_time_min"], block_id, task_in_block, 71),
        "egress_time_min": choose_cycled(levels["private_car"]["egress_time_min"], block_id, task_in_block, 73),
        "parking_search_time_min": choose_cycled(levels["private_car"]["parking_search_time_min"], block_id, task_in_block, 79),
        "availability_pct": None,
        "cost_eur": choose_length_aligned(
            levels["private_car"]["cost_eur"], block_id, task_in_block, 83, trip_length, SD_COST_BUCKETS
        ),
        "scheme": "",
        "engine": "",
        "range_km": None,
    }
    return rows


def build_md_attributes(levels: dict[str, dict[str, list[object]]], block_id: int, task_in_block: int, trip_length: float) -> dict[str, dict]:
    rows: dict[str, dict] = {}

    cs_scheme = choose_cycled(levels["carsharing"]["scheme"], block_id, task_in_block, 5)
    rows["carsharing"] = {
        "travel_time_min": choose_length_aligned(
            levels["carsharing"]["travel_time_min"], block_id, task_in_block, SALT["carsharing"], trip_length, MD_LENGTH_BUCKETS
        ),
        "access_time_min": choose_cycled(levels["carsharing"]["access_time_min"], block_id, task_in_block, 11),
        "waiting_time_min": None,
        "egress_time_min": choose_cycled(levels["carsharing"]["egress_time_min"], block_id, task_in_block, 13),
        "detour_time_min": None,
        "parking_search_time_min": choose_cycled(levels["carsharing"]["parking_search_time_min"], block_id, task_in_block, 17),
        "availability_pct": None,
        "cost_eur": choose_length_aligned(
            levels["carsharing"]["cost_eur"], block_id, task_in_block, 19, trip_length, MD_COST_BUCKETS
        ),
        "scheme": cs_scheme,
        "engine": "",
        "range_km": None,
        "crowding_pct": None,
        "transfer_count": None,
        "scheme_free_floating": int(cs_scheme == "free-floating"),
        "scheme_hybrid": int(cs_scheme == "hybrid"),
    }

    rows["ridepooling"] = {
        "travel_time_min": choose_length_aligned(
            levels["ridepooling"]["travel_time_min"], block_id, task_in_block, SALT["ridepooling"], trip_length, MD_LENGTH_BUCKETS
        ),
        "access_time_min": choose_cycled(levels["ridepooling"]["access_time_min"], block_id, task_in_block, 23),
        "waiting_time_min": choose_cycled(levels["ridepooling"]["waiting_time_min"], block_id, task_in_block, 29),
        "egress_time_min": choose_cycled(levels["ridepooling"]["egress_time_min"], block_id, task_in_block, 31),
        "detour_time_min": choose_cycled(levels["ridepooling"]["detour_time_min"], block_id, task_in_block, 37),
        "parking_search_time_min": None,
        "availability_pct": None,
        "cost_eur": choose_length_aligned(
            levels["ridepooling"]["cost_eur"], block_id, task_in_block, 41, trip_length, MD_COST_BUCKETS
        ),
        "scheme": "",
        "engine": "",
        "range_km": None,
        "crowding_pct": choose_cycled(levels["ridepooling"]["crowding_pct"], block_id, task_in_block, 43),
        "transfer_count": None,
    }

    rows["public_transport"] = {
        "travel_time_min": choose_length_aligned(
            levels["public_transport"]["travel_time_min"], block_id, task_in_block, SALT["public_transport"], trip_length, MD_LENGTH_BUCKETS
        ),
        "access_time_min": choose_cycled(levels["public_transport"]["access_time_min"], block_id, task_in_block, 47),
        "waiting_time_min": choose_cycled(levels["public_transport"]["waiting_time_min"], block_id, task_in_block, 53),
        "egress_time_min": choose_cycled(levels["public_transport"]["egress_time_min"], block_id, task_in_block, 59),
        "detour_time_min": None,
        "parking_search_time_min": None,
        "availability_pct": None,
        "cost_eur": choose_length_aligned(
            levels["public_transport"]["cost_eur"], block_id, task_in_block, 61, trip_length, MD_COST_BUCKETS
        ),
        "scheme": "",
        "engine": "",
        "range_km": None,
        "crowding_pct": choose_cycled(levels["public_transport"]["crowding_pct"], block_id, task_in_block, 67),
        "transfer_count": choose_cycled(levels["public_transport"]["transfer_count"], block_id, task_in_block, 71),
    }

    rows["private_car"] = {
        "travel_time_min": choose_length_aligned(
            levels["private_car"]["travel_time_min"], block_id, task_in_block, SALT["private_car"], trip_length, MD_LENGTH_BUCKETS
        ),
        "access_time_min": choose_cycled(levels["private_car"]["access_time_min"], block_id, task_in_block, 73),
        "waiting_time_min": None,
        "egress_time_min": choose_cycled(levels["private_car"]["egress_time_min"], block_id, task_in_block, 79),
        "detour_time_min": None,
        "parking_search_time_min": choose_cycled(levels["private_car"]["parking_search_time_min"], block_id, task_in_block, 83),
        "availability_pct": None,
        "cost_eur": choose_length_aligned(
            levels["private_car"]["cost_eur"], block_id, task_in_block, 89, trip_length, MD_COST_BUCKETS
        ),
        "scheme": "",
        "engine": "",
        "range_km": None,
        "crowding_pct": None,
        "transfer_count": None,
    }
    return rows


def build_rows_for_subsample(subsample: str, levels: dict[str, dict[str, list[object]]]) -> list[dict]:
    rows: list[dict] = []
    length_levels = [float(value) for value in levels["all"]["trip_length_km"]]
    builder = build_sd_attributes if subsample == "SD" else build_md_attributes

    for block_id in range(1, 9):
        for task_in_block in range(1, 7):
            trip_length = float(length_levels[(block_id + task_in_block - 2) % len(length_levels)])
            active_attributes = builder(levels, block_id, task_in_block, trip_length)

            for alternative_id in ALL_ALTERNATIVES:
                row = base_row(subsample, block_id, task_in_block, trip_length, alternative_id)
                if alternative_id in active_attributes:
                    row.update(active_attributes[alternative_id])
                rows.append(row)
    return rows


def main() -> None:
    sd_levels = load_levels(DATA_DIR / "public_attribute_levels_sd.csv")
    md_levels = load_levels(DATA_DIR / "public_attribute_levels_md.csv")
    config = json.loads((DATA_DIR / "experiment_config.json").read_text(encoding="utf-8"))

    rows = build_rows_for_subsample("SD", sd_levels) + build_rows_for_subsample("MD", md_levels)
    frame = pd.DataFrame(rows)
    frame = frame[
        [
            "subsample",
            "block_id",
            "task_in_block",
            "task_index_within_subsample",
            "task_id",
            "alternative_id",
            "alternative_name",
            "display_label",
            "trip_length_km",
            "is_available",
            "avail_es",
            "avail_bs",
            "avail_walk",
            "avail_car",
            "avail_cs",
            "avail_rp",
            "avail_pt",
            "travel_time_min",
            "access_time_min",
            "waiting_time_min",
            "egress_time_min",
            "detour_time_min",
            "parking_search_time_min",
            "availability_pct",
            "cost_eur",
            "scheme",
            "engine",
            "range_km",
            "crowding_pct",
            "transfer_count",
            "scheme_free_floating",
            "scheme_hybrid",
            "pedelec",
            "provenance",
            "source_note",
        ]
    ]
    frame.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "experiment_name": config["experiment_name"],
        "design_file": str(OUTPUT_PATH.relative_to(EXPERIMENT_DIR)).replace("\\", "/"),
        "replication_standard": config["replication_standard"],
        "rows": int(len(frame)),
        "subsamples": frame["subsample"].value_counts().sort_index().to_dict(),
        "tasks_per_subsample": frame.groupby("subsample")["task_id"].nunique().to_dict(),
        "alternatives_per_task": int(frame.groupby("task_id").size().iloc[0]),
        "note": "Rows are inferred_from_public at the choice-set-combination level; attribute levels are taken from the published SD and MD design tables.",
    }
    (DATA_DIR / "pooled_choice_sets_public_materials_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
