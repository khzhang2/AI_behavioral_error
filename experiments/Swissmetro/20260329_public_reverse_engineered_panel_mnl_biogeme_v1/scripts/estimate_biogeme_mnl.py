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

from common import DATA_DIR, OUTPUT_DIR, ensure_dir, read_json, write_json


CONFIG = read_json(DATA_DIR / "experiment_config.json")
TARGETS = read_json(DATA_DIR / "pylogit_benchmark_targets.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["human", "ai"], required=True)
    return parser.parse_args()


def maybe_float(value):
    return None if value is None else float(value)


def standardize_estimates(estimates: pd.DataFrame) -> pd.DataFrame:
    frame = estimates.reset_index().rename(columns={estimates.index.name or "index": "parameter"})
    renamed = {}
    for column in frame.columns:
        label = column.lower().replace(".", " ").replace("_", " ").strip()
        if column == "parameter":
            continue
        if label == "value":
            renamed[column] = "value"
        elif "robust" in label and "std" in label:
            renamed[column] = "robust_std_err"
        elif "std" in label:
            renamed[column] = "std_err"
        elif "robust" in label and "p" in label:
            renamed[column] = "robust_p_value"
        elif label == "p value":
            renamed[column] = "p_value"
        elif "robust" in label and "t" in label:
            renamed[column] = "robust_t_stat"
        elif label == "t test":
            renamed[column] = "t_stat"
    frame = frame.rename(columns=renamed)
    keep = ["parameter", "value", "std_err", "t_stat", "p_value", "robust_std_err", "robust_t_stat", "robust_p_value"]
    for column in keep:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[keep]


def manual_null_loglikelihood(frame: pd.DataFrame) -> float:
    loglike = 0.0
    for _, row in frame.iterrows():
        available = int(row["TRAIN_AV"]) + int(row["SM_AV"]) + int(row["CAR_AV"])
        if available <= 0:
            continue
        loglike += math.log(1.0 / available)
    return float(loglike)


def prepare_human_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / "human_cleaned_wide.csv").copy()
    frame.loc[frame["GA"] == 1, ["TRAIN_CO", "SM_CO"]] = 0
    frame["TRAIN_COST_SCALED"] = frame["TRAIN_CO"] / 100.0
    frame["SM_COST_SCALED"] = frame["SM_CO"] / 100.0
    frame["CAR_COST_SCALED"] = frame["CAR_CO"] / 100.0
    frame["TRAIN_TIME_SCALED"] = frame["TRAIN_TT"] / 100.0
    frame["SM_TIME_SCALED"] = frame["SM_TT"] / 100.0
    frame["CAR_TIME_SCALED"] = frame["CAR_TT"] / 100.0
    return frame


def prepare_ai_dataset() -> pd.DataFrame:
    wide = pd.read_csv(OUTPUT_DIR / "reconstructed_panels_wide.csv").copy()
    choices = pd.read_csv(OUTPUT_DIR / "parsed_choices.csv").copy()
    choices = choices.loc[choices["is_valid_choice"] == 1, ["respondent_id", "task_id", "choice_code", "chosen_alternative_name"]]
    merged = wide.merge(choices, on=["respondent_id", "task_id"], how="inner")
    merged["CHOICE"] = merged["choice_code"]
    merged["chosen_alternative_name"] = merged["chosen_alternative_name_y"] if "chosen_alternative_name_y" in merged.columns else merged["chosen_alternative_name"]
    merged.loc[merged["GA"] == 1, ["TRAIN_CO", "SM_CO"]] = 0
    merged["TRAIN_COST_SCALED"] = merged["TRAIN_CO"] / 100.0
    merged["SM_COST_SCALED"] = merged["SM_CO"] / 100.0
    merged["CAR_COST_SCALED"] = merged["CAR_CO"] / 100.0
    merged["TRAIN_TIME_SCALED"] = merged["TRAIN_TT"] / 100.0
    merged["SM_TIME_SCALED"] = merged["SM_TT"] / 100.0
    merged["CAR_TIME_SCALED"] = merged["CAR_TT"] / 100.0
    return merged


def run_biogeme(frame: pd.DataFrame, model_name: str):
    model_columns = [
        "CHOICE",
        "TRAIN_AV",
        "SM_AV",
        "CAR_AV",
        "TRAIN_COST_SCALED",
        "SM_COST_SCALED",
        "CAR_COST_SCALED",
        "TRAIN_TIME_SCALED",
        "SM_TIME_SCALED",
        "CAR_TIME_SCALED",
    ]
    numeric_frame = frame[model_columns].copy()
    database = db.Database(model_name, numeric_frame)

    CHOICE = Variable("CHOICE")
    TRAIN_AV = Variable("TRAIN_AV")
    SM_AV = Variable("SM_AV")
    CAR_AV = Variable("CAR_AV")

    ASC_TRAIN = Beta("ASC_TRAIN", 0, None, None, 0)
    ASC_CAR = Beta("ASC_CAR", 0, None, None, 0)
    B_COST = Beta("B_COST", 0, None, None, 0)
    B_TIME = Beta("B_TIME", 0, None, None, 0)

    V = {
        1: ASC_TRAIN + B_COST * Variable("TRAIN_COST_SCALED") + B_TIME * Variable("TRAIN_TIME_SCALED"),
        2: B_COST * Variable("SM_COST_SCALED") + B_TIME * Variable("SM_TIME_SCALED"),
        3: ASC_CAR + B_COST * Variable("CAR_COST_SCALED") + B_TIME * Variable("CAR_TIME_SCALED"),
    }
    av = {1: TRAIN_AV, 2: SM_AV, 3: CAR_AV}
    logprob = models.loglogit(V, av, CHOICE)

    biogeme = BIOGEME(database, logprob)
    biogeme.modelName = model_name
    biogeme.generate_html = False
    biogeme.generate_pickle = False
    threads = max(1, (os.cpu_count() or 1) - 1)
    for attribute_name in ("number_of_threads", "numberOfThreads"):
        if hasattr(biogeme, attribute_name):
            setattr(biogeme, attribute_name, threads)
    results = biogeme.estimate()
    return results, threads


def build_summary(frame: pd.DataFrame, results, dataset_name: str, threads: int) -> dict:
    null_loglikelihood = manual_null_loglikelihood(frame)
    final_loglikelihood = maybe_float(getattr(results.data, "logLike", None))
    number_of_parameters = int(getattr(results.data, "nparam", 0))
    rho_square = None
    rho_bar_square = None
    if null_loglikelihood not in (None, 0) and final_loglikelihood is not None:
        rho_square = 1.0 - (final_loglikelihood / null_loglikelihood)
        rho_bar_square = 1.0 - ((final_loglikelihood - number_of_parameters) / null_loglikelihood)
    summary = {
        "dataset": dataset_name,
        "n_rows": int(len(frame)),
        "n_respondents": int(frame["ID"].nunique()) if "ID" in frame.columns else None,
        "null_loglikelihood": null_loglikelihood,
        "final_loglikelihood": final_loglikelihood,
        "rho_square": rho_square,
        "rho_bar_square": rho_bar_square,
        "number_of_parameters": number_of_parameters,
        "number_of_threads": int(threads),
    }
    if dataset_name == "human":
        summary["pylogit_targets"] = TARGETS
    return summary


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)

    if args.dataset == "human":
        frame = prepare_human_dataset()
        input_path = OUTPUT_DIR / "human_benchmark_biogeme_input.csv"
        estimate_path = OUTPUT_DIR / "human_benchmark_biogeme_mnl_estimates.csv"
        summary_path = OUTPUT_DIR / "human_benchmark_biogeme_mnl_summary.json"
        model_name = "swissmetro_human_benchmark_mnl"
    else:
        frame = prepare_ai_dataset()
        input_path = OUTPUT_DIR / "ai_biogeme_input.csv"
        estimate_path = OUTPUT_DIR / "ai_biogeme_mnl_estimates.csv"
        summary_path = OUTPUT_DIR / "ai_biogeme_mnl_summary.json"
        model_name = "swissmetro_ai_mnl"

    frame.to_csv(input_path, index=False)
    results, threads = run_biogeme(frame, model_name)
    estimates = results.get_estimated_parameters() if hasattr(results, "get_estimated_parameters") else results.getEstimatedParameters()
    standardized = standardize_estimates(estimates)
    standardized.to_csv(estimate_path, index=False)
    write_json(summary_path, build_summary(frame, results, args.dataset, threads))
    print(
        f"[biogeme] dataset={args.dataset} rows={len(frame)} "
        f"loglik={results.data.logLike:.3f} threads={threads}"
    )


if __name__ == "__main__":
    main()
