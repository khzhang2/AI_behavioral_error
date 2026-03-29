from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from common import DATA_DIR, RAW_DIR, RAW_SWISSMETRO_URL, ensure_dir, sha256_file, utc_timestamp, write_json


RAW_PATH = RAW_DIR / "swissmetro.dat"


def main() -> None:
    ensure_dir(RAW_DIR)

    with urllib.request.urlopen(RAW_SWISSMETRO_URL, timeout=120) as response:
        RAW_PATH.write_bytes(response.read())

    raw_frame = pd.read_table(RAW_PATH, sep="\t")

    provenance = {
        "source_url": RAW_SWISSMETRO_URL,
        "fetched_at_utc": utc_timestamp(),
        "sha256": sha256_file(RAW_PATH),
        "raw_path": str(RAW_PATH.relative_to(DATA_DIR.parent)),
        "n_rows": int(raw_frame.shape[0]),
        "n_columns": int(raw_frame.shape[1]),
        "columns": raw_frame.columns.tolist(),
        "notes": [
            "This file is downloaded from the public pylogit Swissmetro example repository.",
            "The public reverse-engineering line freezes the raw tab-delimited source locally before any cleaning or reconstruction."
        ],
    }
    write_json(DATA_DIR / "data_provenance.json", provenance)

    benchmark_targets = {
        "source": "pylogit Swissmetro benchmark notebook",
        "cleaning_rules": {
            "purpose_in": [1, 3],
            "choice_not_equal": 0
        },
        "expected_cleaned_observations": 6768,
        "expected_cleaned_respondents": 752,
        "expected_final_loglikelihood": -5331.252,
        "expected_coefficients": {
            "ASC_CAR": -0.154632,
            "ASC_TRAIN": -0.701187,
            "B_COST": -1.083791,
            "B_TIME": -1.277860
        }
    }
    write_json(DATA_DIR / "pylogit_benchmark_targets.json", benchmark_targets)

    print(f"[fetch] wrote raw data to {RAW_PATH}")
    print(f"[fetch] raw shape = {raw_frame.shape}")


if __name__ == "__main__":
    main()
