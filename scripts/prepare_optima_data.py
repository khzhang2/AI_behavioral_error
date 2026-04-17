from __future__ import annotations

from pathlib import Path

import pandas as pd

from optima_common import (
    COST_SCALE,
    DISTANCE_SCALE,
    INDICATOR_NAMES,
    SOURCE_DATA_DIR,
    SOURCE_OBSERVATION_COLUMN,
    TIME_SCALE,
    WAIT_SCALE,
    pt_non_wait_time,
)


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


def education_text(value: float) -> str:
    education = int(value)
    if education >= 7:
        return "university education"
    if education >= 6:
        return "applied higher education"
    if education >= 5:
        return "high school education"
    if education >= 3:
        return "vocational or secondary education"
    if education >= 1:
        return "compulsory education"
    return "education not reported"


def trip_purpose_text(value: float) -> str:
    trip_purpose = int(value)
    if trip_purpose == 1:
        return "a work-related trip"
    if trip_purpose == 2:
        return "a mixed work-and-leisure trip"
    if trip_purpose == 3:
        return "a leisure-related trip"
    return "a trip with unreported purpose"


def main() -> None:
    raw_path = SOURCE_DATA_DIR / "raw" / "optima.dat"
    frame = pd.read_csv(raw_path, sep="\t")
    frame[SOURCE_OBSERVATION_COLUMN] = frame.index.astype(int) + 1

    frame = frame.loc[frame["Choice"] != -1].copy().reset_index(drop=True)
    frame["normalized_weight"] = frame["Weight"] * len(frame) / frame["Weight"].sum()
    frame["respondent_id"] = [f"H{index + 1:04d}" for index in range(len(frame))]
    frame["human_id"] = frame["ID"]
    frame["CAR_AVAILABLE"] = (frame["CarAvail"] != 3).astype(int)
    frame["PT_AVAILABLE"] = 1
    frame["SLOW_AVAILABLE"] = 1

    frame["ScaledIncome"] = frame["CalculatedIncome"] / 1000.0
    frame["TimePT_non_wait"] = pt_non_wait_time(frame["TimePT"], frame["WaitingTimePT"])
    frame["TimePT_non_wait_scaled"] = frame["TimePT_non_wait"] / TIME_SCALE
    frame["TimePT_scaled"] = frame["TimePT"] / TIME_SCALE
    frame["TimeCar_scaled"] = frame["TimeCar"] / TIME_SCALE
    frame["WaitingTimePT_scaled"] = frame["WaitingTimePT"] / WAIT_SCALE
    frame["MarginalCostPT_scaled"] = frame["MarginalCostPT"] / COST_SCALE
    frame["CostCarCHF_scaled"] = frame["CostCarCHF"] / COST_SCALE
    frame["distance_km_scaled"] = frame["distance_km"] / DISTANCE_SCALE

    frame["high_education"] = (frame["Education"] >= 6).astype(int)
    frame["low_education"] = frame["Education"].isin([1, 2, 3]).astype(int)
    frame["top_manager"] = (frame["SocioProfCat"] == 1).astype(int)
    frame["employees"] = (frame["SocioProfCat"] == 6).astype(int)
    frame["artisans"] = (frame["SocioProfCat"] == 5).astype(int)
    frame["age_30_less"] = (frame["age"] <= 30).astype(int)
    frame["car_oriented_parents"] = (frame["FreqCarPar"] > frame["FreqTrainPar"]).astype(int)
    frame["city_center_as_kid"] = frame["ResidChild"].isin([1, 2]).astype(int)
    frame["childSuburb"] = frame["ResidChild"].isin([3, 4]).astype(int)
    frame["work_trip"] = frame["TripPurpose"].isin([1, 2]).astype(int)
    frame["other_trip"] = 1 - frame["work_trip"]

    frame["sex_text"] = frame["Gender"].map({1: "male", 2: "female"}).fillna("traveler")
    frame["age_text"] = frame["age"].apply(age_text)
    frame["income_text"] = frame["CalculatedIncome"].apply(income_text)
    frame["education_text"] = frame["Education"].apply(education_text)
    frame["trip_purpose_text"] = frame["TripPurpose"].apply(trip_purpose_text)
    frame["car_availability_text"] = frame["CAR_AVAILABLE"].map(
        {1: "car is available for this trip", 0: "car is not available for this trip"}
    )

    profile_columns = [
        SOURCE_OBSERVATION_COLUMN,
        "respondent_id",
        "human_id",
        "normalized_weight",
        "Weight",
        "LangCode",
        "UrbRur",
        "OccupStat",
        "TripPurpose",
        "Education",
        "Region",
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
        "TimePT_non_wait",
        "WaitingTimePT",
        "MarginalCostPT",
        "TimeCar",
        "CostCarCHF",
        "distance_km",
        "TimePT_scaled",
        "TimePT_non_wait_scaled",
        "TimeCar_scaled",
        "WaitingTimePT_scaled",
        "MarginalCostPT_scaled",
        "CostCarCHF_scaled",
        "distance_km_scaled",
        "Choice",
    ]
    profile_frame = frame[profile_columns].copy()

    frame.to_csv(SOURCE_DATA_DIR / "human_cleaned_wide.csv", index=False)
    profile_frame.to_csv(SOURCE_DATA_DIR / "human_respondent_profiles.csv", index=False)

    n_rows = len(frame)
    print(f"[prepare_optima_data] rows={n_rows} respondents={n_rows}")


if __name__ == "__main__":
    main()
