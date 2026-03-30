from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd
from biogeme import database as db
from biogeme import models
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Variable

from optima_common import AI_COLLECTION_DIR, DATA_DIR, EXPERIMENT_DIR, INDICATOR_NAMES, OUTPUT_DIR, archive_experiment_config, ensure_dir, infer_trial_dir_from_output_dir, write_json


SHARED_USER_COLUMNS = [
    "age_30_less",
    "high_education",
    "low_education",
    "top_manager",
    "employees",
    "artisans",
    "ScaledIncome",
    "car_oriented_parents",
    "city_center_as_kid",
    "childSuburb",
    "NbCar",
    "NbBicy",
    "NbHousehold",
    "NbChild",
    "work_trip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["human", "ai"], required=True)
    parser.add_argument("--specification", choices=["basic", "user_characteristics_no_origin_dest"], default="basic")
    parser.add_argument("--output-subdir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def load_dataset(dataset: str, max_rows: int | None) -> pd.DataFrame:
    if dataset == "human":
        frame = pd.read_csv(DATA_DIR / "human_cleaned_wide.csv")
    else:
        frame = pd.read_csv(AI_COLLECTION_DIR / "ai_cleaned_wide.csv")
        valid_mask = frame["Choice"].isin([0, 1, 2])
        for indicator_name in INDICATOR_NAMES:
            valid_mask = valid_mask & frame[indicator_name].isin([1, 2, 3, 4, 5, 6])
        frame = frame.loc[valid_mask].copy()
    frame = frame.copy().sort_values("respondent_id").reset_index(drop=True)
    if "PT_AVAILABLE" not in frame.columns:
        frame["PT_AVAILABLE"] = 1
    if "SLOW_AVAILABLE" not in frame.columns:
        frame["SLOW_AVAILABLE"] = 1
    if max_rows is not None:
        frame = frame.head(int(max_rows)).copy()
    return frame


def standardize_estimates(estimates: pd.DataFrame) -> pd.DataFrame:
    frame = estimates.reset_index().rename(columns={estimates.index.name or "index": "parameter_name"})
    rename_map = {
        "Value": "estimate",
        "Std err": "std_error",
        "t-test": "z_value",
        "p-value": "p_value",
        "Rob. Std err": "robust_std_error",
        "Rob. t-test": "robust_z_value",
        "Rob. p-value": "robust_p_value",
    }
    frame = frame.rename(columns={old: new for old, new in rename_map.items() if old in frame.columns})
    for column in ["estimate", "std_error", "z_value", "p_value", "robust_std_error", "robust_z_value", "robust_p_value"]:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[["parameter_name", "estimate", "std_error", "z_value", "p_value", "robust_std_error", "robust_z_value", "robust_p_value"]]


def build_model(frame: pd.DataFrame, specification: str) -> tuple[BIOGEME, list[str], int]:
    used_columns = [
        "Choice",
        "PT_AVAILABLE",
        "CAR_AVAILABLE",
        "SLOW_AVAILABLE",
        "MarginalCostPT_scaled",
        "CostCarCHF_scaled",
        "TimePT_scaled",
        "WaitingTimePT_scaled",
        "TimeCar_scaled",
    ]
    user_columns: list[str] = []
    if specification == "user_characteristics_no_origin_dest":
        user_columns = [column for column in SHARED_USER_COLUMNS if column in frame.columns]
        used_columns.extend(user_columns)

    database = db.Database("optima_mnl", frame[used_columns].copy())

    Choice = Variable("Choice")
    PT_AVAILABLE = Variable("PT_AVAILABLE")
    CAR_AVAILABLE = Variable("CAR_AVAILABLE")
    SLOW_AVAILABLE = Variable("SLOW_AVAILABLE")

    ASC_PT = Beta("ASC_PT", 0, None, None, 0)
    ASC_CAR = Beta("ASC_CAR", 0, None, None, 0)
    B_COST = Beta("B_COST", 0, None, None, 0)
    B_TIME = Beta("B_TIME", 0, None, None, 0)

    utility_pt = ASC_PT + B_COST * Variable("MarginalCostPT_scaled") + B_TIME * (
        Variable("TimePT_scaled") + Variable("WaitingTimePT_scaled")
    )
    utility_car = ASC_CAR + B_COST * Variable("CostCarCHF_scaled") + B_TIME * Variable("TimeCar_scaled")
    utility_slow = 0

    for column in user_columns:
        utility_pt += Beta(f"B_PT_{column}", 0, None, None, 0) * Variable(column)
        utility_car += Beta(f"B_CAR_{column}", 0, None, None, 0) * Variable(column)

    utilities = {0: utility_pt, 1: utility_car, 2: utility_slow}
    availability = {0: PT_AVAILABLE, 1: CAR_AVAILABLE, 2: SLOW_AVAILABLE}
    logprob = models.loglogit(utilities, availability, Choice)

    biogeme = BIOGEME(database, logprob)
    biogeme.generate_html = False
    biogeme.generate_pickle = False
    threads = max(1, (os.cpu_count() or 1) - 1)
    for attribute_name in ("number_of_threads", "numberOfThreads"):
        if hasattr(biogeme, attribute_name):
            setattr(biogeme, attribute_name, threads)
    return biogeme, user_columns, threads
def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        output_dir = ensure_dir(Path(args.output_dir))
    elif args.output_subdir is not None:
        output_dir = ensure_dir(OUTPUT_DIR / args.output_subdir)
    else:
        output_dir = ensure_dir(OUTPUT_DIR / f"{args.dataset}_mnl_{args.specification}")
    trial_dir = infer_trial_dir_from_output_dir(output_dir) or EXPERIMENT_DIR
    archive_experiment_config(trial_dir)
    frame = load_dataset(args.dataset, args.max_rows)
    biogeme, user_columns, threads = build_model(frame, args.specification)
    model_name = f"optima_{args.dataset}_{args.specification}_mnl"
    biogeme.modelName = model_name
    results = biogeme.estimate()

    estimates = results.get_estimated_parameters() if hasattr(results, "get_estimated_parameters") else results.getEstimatedParameters()
    standardized = standardize_estimates(estimates)
    standardized.to_csv(output_dir / "biogeme_mnl_estimates.csv", index=False)
    frame.to_csv(output_dir / "estimation_input.csv", index=False)

    null_loglikelihood = None
    if all(column in frame.columns for column in ["PT_AVAILABLE", "CAR_AVAILABLE", "SLOW_AVAILABLE"]):
        available = frame[["PT_AVAILABLE", "CAR_AVAILABLE", "SLOW_AVAILABLE"]].sum(axis=1)
        null_loglikelihood = float(sum(0.0 if value <= 0 else math.log(1.0 / value) for value in available))

    summary = {
        "dataset": args.dataset,
        "specification": args.specification,
        "n_rows": int(len(frame)),
        "final_loglikelihood": float(results.data.logLike),
        "null_loglikelihood": null_loglikelihood,
        "number_of_parameters": int(results.data.nparam),
        "number_of_threads": int(threads),
        "n_user_characteristics": int(len(user_columns)),
        "user_characteristics": user_columns,
    }
    write_json(output_dir / "biogeme_mnl_summary.json", summary)
    print(
        f"[biogeme_mnl] dataset={args.dataset} specification={args.specification} "
        f"rows={len(frame)} loglik={results.data.logLike:.3f} threads={threads}"
    )


if __name__ == "__main__":
    main()
