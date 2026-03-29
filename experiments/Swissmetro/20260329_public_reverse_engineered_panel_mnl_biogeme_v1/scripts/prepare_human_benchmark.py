from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import (
    CHOICE_CODE_TO_NAME,
    DATA_DIR,
    RAW_DIR,
    age_text,
    car_availability_text,
    destination_text,
    ensure_dir,
    first_class_text,
    ga_text,
    income_text,
    luggage_text,
    origin_text,
    payer_text,
    purpose_text,
    sex_text,
    survey_text,
    ticket_text,
    write_json,
)


RAW_PATH = RAW_DIR / "swissmetro.dat"


def convert_wide_to_long(clean_frame: pd.DataFrame) -> pd.DataFrame:
    long_rows: list[dict] = []
    for row in clean_frame.itertuples(index=False):
        for mode_id, mode_name in CHOICE_CODE_TO_NAME.items():
            if mode_id == 1:
                availability = int(row.TRAIN_AV)
                travel_time = int(row.TRAIN_TT)
                travel_cost = int(row.TRAIN_CO)
                headway = int(row.TRAIN_HE)
                seat_configuration = 0
            elif mode_id == 2:
                availability = int(row.SM_AV)
                travel_time = int(row.SM_TT)
                travel_cost = int(row.SM_CO)
                headway = int(row.SM_HE)
                seat_configuration = int(row.SM_SEATS)
            else:
                availability = int(row.CAR_AV)
                travel_time = int(row.CAR_TT)
                travel_cost = int(row.CAR_CO)
                headway = 0
                seat_configuration = 0

            travel_cost_hundredth = travel_cost / 100.0
            if int(row.GA) == 1 and mode_id in (1, 2):
                travel_cost_hundredth = 0.0

            long_rows.append(
                {
                    "custom_id": int(row.custom_id),
                    "ID": int(row.ID),
                    "mode_id": mode_id,
                    "alternative_name": mode_name,
                    "CHOICE": int(row.CHOICE),
                    "choice": int(row.CHOICE == mode_id),
                    "availability": availability,
                    "travel_time": travel_time,
                    "travel_cost": travel_cost,
                    "headway": headway,
                    "seat_configuration": seat_configuration,
                    "travel_time_hundredth": travel_time / 100.0,
                    "travel_cost_hundredth": travel_cost_hundredth,
                    "GROUP": int(row.GROUP),
                    "SURVEY": int(row.SURVEY),
                    "SP": int(row.SP),
                    "PURPOSE": int(row.PURPOSE),
                    "FIRST": int(row.FIRST),
                    "TICKET": int(row.TICKET),
                    "WHO": int(row.WHO),
                    "LUGGAGE": int(row.LUGGAGE),
                    "AGE": int(row.AGE),
                    "MALE": int(row.MALE),
                    "INCOME": int(row.INCOME),
                    "GA": int(row.GA),
                    "ORIGIN": int(row.ORIGIN),
                    "DEST": int(row.DEST),
                    "CAR_AV": int(row.CAR_AV),
                }
            )
    return pd.DataFrame(long_rows)


def build_profile_frame(clean_frame: pd.DataFrame) -> pd.DataFrame:
    profile_columns = [
        "ID",
        "GROUP",
        "SURVEY",
        "PURPOSE",
        "FIRST",
        "TICKET",
        "WHO",
        "LUGGAGE",
        "AGE",
        "MALE",
        "INCOME",
        "GA",
        "ORIGIN",
        "DEST",
        "CAR_AV",
    ]
    profiles = clean_frame.groupby("ID", as_index=False)[profile_columns[1:]].first()
    profiles["task_count"] = clean_frame.groupby("ID").size().values
    profiles["persona_source"] = "empirical_cleaned_profile"
    profiles["purpose_text"] = profiles["PURPOSE"].map(purpose_text)
    profiles["first_class_text"] = profiles["FIRST"].map(first_class_text)
    profiles["ticket_text"] = profiles["TICKET"].map(ticket_text)
    profiles["who_text"] = profiles["WHO"].map(payer_text)
    profiles["luggage_text"] = profiles["LUGGAGE"].map(luggage_text)
    profiles["age_text"] = profiles["AGE"].map(age_text)
    profiles["sex_text"] = profiles["MALE"].map(sex_text)
    profiles["income_text"] = profiles["INCOME"].map(income_text)
    profiles["ga_text"] = profiles["GA"].map(ga_text)
    profiles["origin_text"] = profiles["ORIGIN"].map(origin_text)
    profiles["dest_text"] = profiles["DEST"].map(destination_text)
    profiles["survey_text"] = profiles["SURVEY"].map(survey_text)
    profiles["car_av_text"] = profiles["CAR_AV"].map(car_availability_text)
    return profiles


def main() -> None:
    ensure_dir(DATA_DIR)
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    raw_frame = pd.read_table(RAW_PATH, sep="\t")
    clean_frame = raw_frame.loc[raw_frame["PURPOSE"].isin([1, 3]) & (raw_frame["CHOICE"] != 0)].copy()
    clean_frame["custom_id"] = range(1, len(clean_frame) + 1)
    clean_frame["task_position"] = clean_frame.groupby("ID").cumcount() + 1
    clean_frame["task_id"] = clean_frame["ID"].map(lambda value: f"H{int(value):04d}") + "_T" + clean_frame["task_position"].map(lambda value: f"{int(value):02d}")
    clean_frame["TRAIN_COST_SCALED"] = clean_frame["TRAIN_CO"] / 100.0
    clean_frame["SM_COST_SCALED"] = clean_frame["SM_CO"] / 100.0
    clean_frame["CAR_COST_SCALED"] = clean_frame["CAR_CO"] / 100.0
    ga_mask = clean_frame["GA"] == 1
    clean_frame.loc[ga_mask, "TRAIN_COST_SCALED"] = 0.0
    clean_frame.loc[ga_mask, "SM_COST_SCALED"] = 0.0
    clean_frame["TRAIN_TIME_SCALED"] = clean_frame["TRAIN_TT"] / 100.0
    clean_frame["SM_TIME_SCALED"] = clean_frame["SM_TT"] / 100.0
    clean_frame["CAR_TIME_SCALED"] = clean_frame["CAR_TT"] / 100.0
    clean_frame["chosen_alternative_name"] = clean_frame["CHOICE"].map(CHOICE_CODE_TO_NAME)

    long_frame = convert_wide_to_long(clean_frame)
    profile_frame = build_profile_frame(clean_frame)

    clean_frame.to_csv(DATA_DIR / "human_cleaned_wide.csv", index=False)
    long_frame.to_csv(DATA_DIR / "human_cleaned_long.csv", index=False)
    profile_frame.to_csv(DATA_DIR / "human_respondent_profiles.csv", index=False)

    summary = {
        "raw_shape": [int(raw_frame.shape[0]), int(raw_frame.shape[1])],
        "cleaned_shape": [int(clean_frame.shape[0]), int(clean_frame.shape[1])],
        "n_cleaned_respondents": int(clean_frame["ID"].nunique()),
        "rows_per_respondent_distribution": clean_frame.groupby("ID").size().value_counts().sort_index().to_dict(),
        "choice_counts": clean_frame["chosen_alternative_name"].value_counts().to_dict(),
    }
    write_json(DATA_DIR / "human_benchmark_sample_summary.json", summary)

    print(f"[prepare] cleaned wide shape = {clean_frame.shape}")
    print(f"[prepare] cleaned respondents = {clean_frame['ID'].nunique()}")


if __name__ == "__main__":
    main()
