from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd

from biogeme import database as db
from biogeme import models
from biogeme.biogeme import BIOGEME
from biogeme.expressions import Beta, Variable

from common import CHOICE_CODE_TO_NAME, DATA_DIR, OUTPUT_DIR, ensure_dir, read_json, utc_timestamp


def maybe_float(value: object) -> float | None:
    return None if value is None else float(value)


def standardize_estimates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.index.name is not None or "Value" in result.columns:
        result = result.reset_index().rename(columns={"index": "parameter_name"})
    column_map = {
        "Value": "estimate",
        "Std err": "std_error",
        "Rob. Std err": "robust_std_error",
        "t-test": "z_value",
        "Rob. t-test": "robust_z_value",
        "p-value": "p_value",
        "Rob. p-value": "robust_p_value",
    }
    result = result.rename(columns=column_map)
    desired = [
        "parameter_name",
        "estimate",
        "std_error",
        "z_value",
        "p_value",
        "robust_std_error",
        "robust_z_value",
        "robust_p_value",
    ]
    for column in desired:
        if column not in result.columns:
            result[column] = None
    return result[desired].copy()


def compute_manual_null_loglikelihood(frame: pd.DataFrame) -> float:
    availability_count = frame[["TRAIN_AV", "SM_AV", "CAR_AV"]].sum(axis=1)
    chosen_prob = np.where(
        (frame["CHOICE"] == 1) & (frame["TRAIN_AV"] == 1),
        1.0 / availability_count,
        np.where(
            (frame["CHOICE"] == 2) & (frame["SM_AV"] == 1),
            1.0 / availability_count,
            np.where((frame["CHOICE"] == 3) & (frame["CAR_AV"] == 1), 1.0 / availability_count, np.nan),
        ),
    )
    return float(np.log(chosen_prob).sum())


def prepare_estimation_frame(dataset_role: str, run_id: int | None) -> tuple[pd.DataFrame, Path, str]:
    if dataset_role == "human":
        output_dir = OUTPUT_DIR / "human_benchmark"
        ensure_dir(output_dir)
        wide = pd.read_csv(DATA_DIR / "human_cleaned_wide.csv")
        frame = wide.copy()
        model_label = "human_benchmark"
    else:
        if run_id is None:
            raise ValueError("--run-id is required when dataset-role=ai")
        output_dir = OUTPUT_DIR / f"ai_run_{run_id:02d}"
        wide = pd.read_csv(output_dir / "reconstructed_panels_wide.csv")
        choices = pd.read_csv(output_dir / "parsed_choices.csv")
        frame = wide.merge(
            choices[["synthetic_respondent_id", "task_id", "choice_code", "is_valid_choice"]],
            on=["synthetic_respondent_id", "task_id"],
            how="left",
        )
        frame = frame.loc[frame["is_valid_choice"] == 1].copy()
        frame["CHOICE"] = frame["choice_code"].astype(int)
        model_label = f"ai_run_{run_id:02d}"

    frame["TRAIN_TIME_SCALED"] = frame["TRAIN_TT"] / 100.0
    frame["SM_TIME_SCALED"] = frame["SM_TT"] / 100.0
    frame["CAR_TIME_SCALED"] = frame["CAR_TT"] / 100.0
    frame["TRAIN_COST_SCALED"] = frame["TRAIN_CO"] / 100.0
    frame["SM_COST_SCALED"] = frame["SM_CO"] / 100.0
    frame["CAR_COST_SCALED"] = frame["CAR_CO"] / 100.0
    ga_mask = frame["GA"] == 1
    frame.loc[ga_mask, "TRAIN_COST_SCALED"] = 0.0
    frame.loc[ga_mask, "SM_COST_SCALED"] = 0.0

    if "custom_id" not in frame.columns:
        frame["custom_id"] = range(1, len(frame) + 1)

    return frame, output_dir, model_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-role", choices=["human", "ai"], required=True)
    parser.add_argument("--run-id", type=int, default=None)
    args = parser.parse_args()

    config = read_json(DATA_DIR / "experiment_config.json")
    frame, output_dir, model_label = prepare_estimation_frame(args.dataset_role, args.run_id)
    ensure_dir(output_dir)
    n_respondents = int(frame["ID"].nunique()) if "ID" in frame.columns else int(frame["synthetic_respondent_id"].nunique())
    numeric_frame = frame.select_dtypes(include=["number", "bool"]).copy()
    numeric_frame.to_csv(output_dir / "biogeme_estimation_wide.csv", index=False)

    database = db.Database(model_label, numeric_frame)

    CHOICE = Variable("CHOICE")
    TRAIN_AV = Variable("TRAIN_AV")
    SM_AV = Variable("SM_AV")
    CAR_AV = Variable("CAR_AV")
    TRAIN_COST_SCALED = Variable("TRAIN_COST_SCALED")
    SM_COST_SCALED = Variable("SM_COST_SCALED")
    CAR_COST_SCALED = Variable("CAR_COST_SCALED")
    TRAIN_TIME_SCALED = Variable("TRAIN_TIME_SCALED")
    SM_TIME_SCALED = Variable("SM_TIME_SCALED")
    CAR_TIME_SCALED = Variable("CAR_TIME_SCALED")

    ASC_CAR = Beta("ASC_CAR", 0, None, None, 0)
    ASC_TRAIN = Beta("ASC_TRAIN", 0, None, None, 0)
    B_COST = Beta("B_COST", 0, None, None, 0)
    B_TIME = Beta("B_TIME", 0, None, None, 0)

    utilities = {
        1: ASC_TRAIN + B_COST * TRAIN_COST_SCALED + B_TIME * TRAIN_TIME_SCALED,
        2: B_COST * SM_COST_SCALED + B_TIME * SM_TIME_SCALED,
        3: ASC_CAR + B_COST * CAR_COST_SCALED + B_TIME * CAR_TIME_SCALED,
    }
    availability = {
        1: TRAIN_AV,
        2: SM_AV,
        3: CAR_AV,
    }
    logprob = models.loglogit(utilities, availability, CHOICE)

    os.chdir(output_dir)
    biogeme = BIOGEME(database, logprob)
    biogeme.modelName = f"swissmetro_mnl_{model_label}"
    biogeme.generate_html = False
    biogeme.generate_pickle = False
    thread_budget = max(1, multiprocessing.cpu_count() - 1)
    biogeme.number_of_threads = thread_budget
    results = biogeme.estimate()

    if hasattr(results, "get_estimated_parameters"):
        estimates = results.get_estimated_parameters()
    else:
        estimates = results.getEstimatedParameters()
    standardized = standardize_estimates(estimates)
    standardized.to_csv(output_dir / "biogeme_mnl_estimates.csv", index=False)

    final_loglikelihood = maybe_float(getattr(results.data, "logLike", None))
    null_loglikelihood = maybe_float(getattr(results.data, "nullLogLike", None))
    if null_loglikelihood is None:
        null_loglikelihood = compute_manual_null_loglikelihood(numeric_frame)
    rho_square = None if final_loglikelihood is None else 1.0 - (final_loglikelihood / null_loglikelihood)
    rho_bar_square = None if final_loglikelihood is None else 1.0 - ((final_loglikelihood - results.data.nparam) / null_loglikelihood)

    summary = {
        "experiment_name": config["experiment_name"],
        "dataset_role": args.dataset_role,
        "run_id": args.run_id,
        "model_name": biogeme.modelName,
        "n_observations": int(len(numeric_frame)),
        "n_respondents": n_respondents,
        "null_loglikelihood": null_loglikelihood,
        "final_loglikelihood": final_loglikelihood,
        "rho_square": rho_square,
        "rho_bar_square": rho_bar_square,
        "n_parameters": int(results.data.nparam),
        "number_of_threads": int(thread_budget),
        "estimated_at_utc": utc_timestamp(),
        "choice_counts": {
            CHOICE_CODE_TO_NAME[int(code)]: int(count)
            for code, count in numeric_frame["CHOICE"].value_counts().sort_index().to_dict().items()
        },
    }

    if args.dataset_role == "human":
        benchmark = read_json(DATA_DIR / "pylogit_benchmark_targets.json")
        summary["pylogit_benchmark"] = benchmark
        summary["benchmark_deltas"] = {
            name: float(
                standardized.loc[standardized["parameter_name"] == name, "estimate"].iloc[0]
                - benchmark["expected_coefficients"][name]
            )
            for name in benchmark["expected_coefficients"]
        }
        summary["benchmark_loglikelihood_delta"] = float(summary["final_loglikelihood"] - benchmark["expected_final_loglikelihood"])

    (output_dir / "biogeme_mnl_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[estimate] {model_label}: final_ll={summary['final_loglikelihood']:.3f}")


if __name__ == "__main__":
    main()
