from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
DATA_DIR = EXPERIMENT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DOCS_DIR = EXPERIMENT_DIR / "docs"

RAW_SWISSMETRO_URL = "https://raw.githubusercontent.com/timothyb0912/pylogit/master/examples/data/swissmetro.dat"

CHOICE_CODE_TO_NAME = {
    1: "TRAIN",
    2: "SWISSMETRO",
    3: "CAR",
}

LABEL_TO_CHOICE_CODE = {
    "A": 1,
    "B": 2,
    "C": 3,
}

CHOICE_CODE_TO_LABEL = {
    1: "A",
    2: "B",
    3: "C",
}

AGE_TEXT = {
    0: "age code 0",
    1: "age code 1",
    2: "age code 2",
    3: "age code 3",
    4: "age code 4",
    5: "age code 5",
    6: "age code 6",
}

INCOME_TEXT = {
    0: "income code 0",
    1: "income code 1",
    2: "income code 2",
    3: "income code 3",
    4: "income code 4",
}

LUGGAGE_TEXT = {
    0: "no luggage",
    1: "luggage code 1",
    2: "luggage code 2",
    3: "luggage code 3",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def purpose_text(value: int) -> str:
    if value == 1:
        return "purpose code 1"
    if value == 3:
        return "purpose code 3"
    return f"purpose code {value}"


def first_class_text(value: int) -> str:
    return "first class" if int(value) == 1 else "second class"


def ticket_text(value: int) -> str:
    return f"ticket category code {int(value)}"


def payer_text(value: int) -> str:
    return f"payer code {int(value)}"


def luggage_text(value: int) -> str:
    return LUGGAGE_TEXT.get(int(value), f"luggage code {int(value)}")


def age_text(value: int) -> str:
    return AGE_TEXT.get(int(value), f"age code {int(value)}")


def sex_text(value: int) -> str:
    return "male" if int(value) == 1 else "female"


def income_text(value: int) -> str:
    return INCOME_TEXT.get(int(value), f"income code {int(value)}")


def ga_text(value: int) -> str:
    return "has a GA travelcard" if int(value) == 1 else "does not have a GA travelcard"


def origin_text(value: int) -> str:
    return f"origin zone code {int(value)}"


def destination_text(value: int) -> str:
    return f"destination zone code {int(value)}"


def survey_text(value: int) -> str:
    return f"survey stratum {int(value)}"


def car_availability_text(value: int) -> str:
    return "car is available for this trip" if int(value) == 1 else "car is not available for this trip"


def safe_ratio(value: float, baseline: float, digits: int = 3) -> float:
    if baseline == 0:
        return 0.0
    return round(float(value) / float(baseline), digits)


def reconstruct_from_ratio(baseline: float, ratio: float) -> int:
    if baseline == 0 or ratio == 0:
        return 0
    return int(round(float(baseline) * float(ratio)))
