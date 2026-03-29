from __future__ import annotations

from pathlib import Path

import pandas as pd

from optima_common import CONFIG, DATA_DIR, DRAW_NAMES, INDICATOR_NAMES, generate_shared_sobol_draws, write_json


def age_text(age_value: float) -> str:
    age_value = int(age_value)
    if age_value <= 30:
        return "30 or younger"
    if age_value >= 65:
        return "65 or older"
    return f"{age_value} years old"


def income_text(value: float) -> str:
    if value <= 0:
        return "unknown income"
    return f"around CHF {int(round(value))} per month"


def build_choice_long(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        rows.extend(
            [
                {
                    "respondent_id": row["respondent_id"],
                    "human_id": row["human_id"],
                    "Choice": row["Choice"],
                    "alternative_code": 0,
                    "alternative_name": "PT",
                    "availability": 1,
                    "travel_time": row["TimePT"],
                    "waiting_time": row["WaitingTimePT"],
                    "travel_cost": row["MarginalCostPT"],
                    "distance_km": row["distance_km"],
                    "scaled_time": row["TimePT_scaled"],
                    "scaled_waiting": row["WaitingTimePT_scaled"],
                    "scaled_cost": row["MarginalCostPT_scaled"],
                    "scaled_distance": row["distance_km_scaled"],
                },
                {
                    "respondent_id": row["respondent_id"],
                    "human_id": row["human_id"],
                    "Choice": row["Choice"],
                    "alternative_code": 1,
                    "alternative_name": "CAR",
                    "availability": row["CAR_AVAILABLE"],
                    "travel_time": row["TimeCar"],
                    "waiting_time": 0,
                    "travel_cost": row["CostCarCHF"],
                    "distance_km": row["distance_km"],
                    "scaled_time": row["TimeCar_scaled"],
                    "scaled_waiting": 0,
                    "scaled_cost": row["CostCarCHF_scaled"],
                    "scaled_distance": row["distance_km_scaled"],
                },
                {
                    "respondent_id": row["respondent_id"],
                    "human_id": row["human_id"],
                    "Choice": row["Choice"],
                    "alternative_code": 2,
                    "alternative_name": "SLOW_MODES",
                    "availability": 1,
                    "travel_time": 0,
                    "waiting_time": 0,
                    "travel_cost": 0,
                    "distance_km": row["distance_km"],
                    "scaled_time": 0,
                    "scaled_waiting": 0,
                    "scaled_cost": 0,
                    "scaled_distance": row["distance_km_scaled"],
                },
            ]
        )
    return pd.DataFrame(rows)


def main() -> None:
    raw_path = DATA_DIR / "raw" / "optima.dat"
    frame = pd.read_csv(raw_path, sep="\t")

    frame = frame.loc[frame["Choice"] != -1].copy()
    frame = frame.loc[~((frame["Choice"] == 1) & (frame["CarAvail"] == 3))].copy()
    frame = frame.loc[frame["TripPurpose"] != 3].copy()
    frame = frame.loc[frame["NbTrajects"] != 1].copy()
    frame = frame.loc[frame["TimePT"] != 0].copy()
    frame = frame.loc[frame["TimeCar"] != 0].copy()
    frame = frame.loc[frame["distance_km"] != 0].copy()
    for indicator_name in INDICATOR_NAMES:
        frame = frame.loc[frame[indicator_name].between(1, 6)].copy()

    frame = frame.reset_index(drop=True)
    frame["normalized_weight"] = frame["Weight"] * len(frame) / frame["Weight"].sum()
    frame["respondent_id"] = [f"H{index + 1:04d}" for index in range(len(frame))]
    frame["human_id"] = frame["ID"]
    frame["CAR_AVAILABLE"] = (frame["CarAvail"] != 3).astype(int)
    frame["PT_AVAILABLE"] = 1
    frame["SLOW_AVAILABLE"] = 1

    frame["ScaledIncome"] = frame["CalculatedIncome"] / 1000.0
    frame["TimePT_scaled"] = frame["TimePT"] / 200.0
    frame["TimeCar_scaled"] = frame["TimeCar"] / 200.0
    frame["WaitingTimePT_scaled"] = frame["WaitingTimePT"] / 60.0
    frame["MarginalCostPT_scaled"] = frame["MarginalCostPT"] / 10.0
    frame["CostCarCHF_scaled"] = frame["CostCarCHF"] / 10.0
    frame["distance_km_scaled"] = frame["distance_km"] / 5.0

    frame["high_education"] = (frame["Education"] >= 6).astype(int)
    frame["low_education"] = (frame["Education"] <= 3).astype(int)
    frame["top_manager"] = (frame["SocioProfCat"] == 1).astype(int)
    frame["employees"] = (frame["SocioProfCat"] == 6).astype(int)
    frame["artisans"] = (frame["SocioProfCat"] == 5).astype(int)
    frame["age_30_less"] = (frame["age"] <= 30).astype(int)
    frame["car_oriented_parents"] = (frame["FreqCarPar"] > frame["FreqTrainPar"]).astype(int)
    frame["city_center_as_kid"] = frame["ResidChild"].isin([1, 2]).astype(int)
    frame["childSuburb"] = frame["ResidChild"].isin([3, 4]).astype(int)
    frame["work_trip"] = (frame["TripPurpose"] == 1).astype(int)
    frame["other_trip"] = (frame["TripPurpose"] != 1).astype(int)

    frame["sex_text"] = frame["Gender"].map({1: "male", 2: "female"}).fillna("traveler")
    frame["age_text"] = frame["age"].apply(age_text)
    frame["income_text"] = frame["CalculatedIncome"].apply(income_text)
    frame["education_text"] = frame["high_education"].map({1: "high education", 0: "not high education"})
    frame["trip_purpose_text"] = frame["work_trip"].map({1: "a work trip", 0: "a non-work trip"})
    frame["car_availability_text"] = frame["CAR_AVAILABLE"].map(
        {1: "car is available for this trip", 0: "car is not available for this trip"}
    )

    choice_long = build_choice_long(frame)

    profile_columns = [
        "respondent_id",
        "human_id",
        "normalized_weight",
        "age",
        "sex_text",
        "age_text",
        "CalculatedIncome",
        "income_text",
        "high_education",
        "low_education",
        "top_manager",
        "employees",
        "artisans",
        "age_30_less",
        "ScaledIncome",
        "car_oriented_parents",
        "city_center_as_kid",
        "childSuburb",
        "NbCar",
        "NbBicy",
        "NbHousehold",
        "NbChild",
        "work_trip",
        "other_trip",
        "trip_purpose_text",
        "education_text",
        "CAR_AVAILABLE",
        "car_availability_text",
        *INDICATOR_NAMES,
        "TimePT",
        "WaitingTimePT",
        "MarginalCostPT",
        "TimeCar",
        "CostCarCHF",
        "distance_km",
        "TimePT_scaled",
        "TimeCar_scaled",
        "WaitingTimePT_scaled",
        "MarginalCostPT_scaled",
        "CostCarCHF_scaled",
        "distance_km_scaled",
        "Choice",
    ]
    profile_frame = frame[profile_columns].copy()

    frame.to_csv(DATA_DIR / "human_cleaned_wide.csv", index=False)
    choice_long.to_csv(DATA_DIR / "human_cleaned_long.csv", index=False)
    profile_frame.to_csv(DATA_DIR / "human_respondent_profiles.csv", index=False)

    n_rows = len(frame)
    draws_500 = generate_shared_sobol_draws(
        n_rows=n_rows,
        n_draws=int(CONFIG["n_monte_carlo_draws_torch_final"]),
        n_dims=len(DRAW_NAMES),
        seed=int(CONFIG["master_seed"]),
    )
    draws_32 = draws_500[:, : int(CONFIG["n_monte_carlo_draws_biogeme"]), :].copy()
    with (DATA_DIR / "shared_sobol_draws_500.npy").open("wb") as handle:
        import numpy as np

        np.save(handle, draws_500)
    with (DATA_DIR / "shared_sobol_draws_32.npy").open("wb") as handle:
        import numpy as np

        np.save(handle, draws_32)

    summary = {
        "n_rows_after_cleaning": int(n_rows),
        "n_selected_indicators": len(INDICATOR_NAMES),
        "selected_indicators": INDICATOR_NAMES,
        "choice_share": frame["Choice"].value_counts(normalize=True).sort_index().to_dict(),
    }
    write_json(DATA_DIR / "human_benchmark_sample_summary.json", summary)
    print(f"[prepare_optima_data] rows={n_rows} respondents={n_rows} draws32={draws_32.shape} draws500={draws_500.shape}")


if __name__ == "__main__":
    main()
