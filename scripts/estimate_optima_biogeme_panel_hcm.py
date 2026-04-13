from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import biogeme.biogeme as bio
from biogeme import database as db
from biogeme import models
from biogeme.expressions import Beta, MonteCarlo, Variable, bioDraws, bioNormalCdf, log

from optima_common import INDICATOR_NAMES, ROOT_DIR, draw_generator_from_file, ensure_dir, experiment_analysis_dir, write_json
from optima_hcm_model_spec import INDICATOR_SPECS, PARAMETER_ORDER, POSITIVE_PARAMETERS


RUNTIME_PARAMETER_FILE = ROOT_DIR / "biogeme_runtime.toml"
TIME_SCALE = 200.0
WAIT_SCALE = 60.0
COST_SCALE = 10.0
DISTANCE_SCALE = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--dataset", choices=["ai", "human", "both"], default="both")
    parser.add_argument("--n-draws", type=int, default=None)
    parser.add_argument("--max-respondents", type=int, default=None)
    return parser.parse_args()


def read_json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment_config(experiment_dir: Path) -> dict:
    config_path = experiment_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing experiment_config.json under {experiment_dir}")
    payload = read_json_file(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"experiment_config.json under {experiment_dir} must be a JSON object.")
    return payload


def source_data_dir(config: dict) -> Path:
    return ROOT_DIR / str(config["paths"]["source_data_dir"])


def n_draws_from_config(config: dict, override: int | None) -> int:
    if override is not None:
        return int(override)
    return int(config.get("n_monte_carlo_draws_biogeme", 32))


def draw_file_path(config: dict, n_draws: int) -> Path:
    return source_data_dir(config) / f"shared_sobol_draws_{int(n_draws)}.npy"


def optimization_algorithm(config: dict) -> str:
    return str(config.get("biogeme", {}).get("optimization_algorithm", "simple_bounds"))


def n_core_tasks(config: dict) -> int:
    return int(config.get("survey_design", {}).get("n_core_tasks", 0))


def write_parameter_file(n_draws: int, threads: int, algorithm: str) -> None:
    RUNTIME_PARAMETER_FILE.write_text(
        "\n".join(
            [
                "[MonteCarlo]",
                f"number_of_draws = {int(n_draws)}",
                "",
                "[MultiThreading]",
                f"number_of_threads = {int(threads)}",
                "",
                "[Estimation]",
                f'optimization_algorithm = "{algorithm}"',
                'save_iterations = "False"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def make_beta(name: str, lower: float | None = None, upper: float | None = None) -> Beta:
    start = 0.5 if name in {"SIGMA_CAR", "SIGMA_ENV", "DELTA_1", "DELTA_2"} else 0.0
    return Beta(name, start, lower, upper, 0)


def register_shared_draws(database: db.Database, draw_path: Path) -> None:
    generators = {
        "SOBOL_CAR": (draw_generator_from_file(draw_path, 0), "Shared Sobol normal draw for car latent disturbance"),
        "SOBOL_ENV": (draw_generator_from_file(draw_path, 1), "Shared Sobol normal draw for environmental latent disturbance"),
    }
    database.setRandomNumberGenerators(generators)


def ordered_probit_probability(observed: Variable, latent_index, delta_1, delta_2):
    tau_1 = delta_1
    tau_2 = delta_1 + delta_2
    thresholds = [-tau_2, -tau_1, 0, tau_1, tau_2]
    probabilities = []
    for category in range(1, 7):
        upper = 1.0 if category == 6 else bioNormalCdf(thresholds[category - 1] - latent_index)
        lower = 0.0 if category == 1 else bioNormalCdf(thresholds[category - 2] - latent_index)
        probabilities.append((observed == category) * (upper - lower))
    return sum(probabilities)


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


def null_loglikelihood(frame: pd.DataFrame) -> float:
    n_available = frame[["PT_AVAILABLE", "CAR_AVAILABLE", "SLOW_AVAILABLE"]].sum(axis=1).clip(lower=1.0)
    logprob = -np.log(n_available)
    respondent_ll = frame.groupby("respondent_id").agg(
        normalized_weight=("normalized_weight", "first"),
        logprob_sum=("null_logprob", "sum"),
    ) if "null_logprob" in frame.columns else None
    if respondent_ll is None:
        tmp = frame[["respondent_id", "normalized_weight"]].copy()
        tmp["null_logprob"] = logprob
        respondent_ll = tmp.groupby("respondent_id").agg(
            normalized_weight=("normalized_weight", "first"),
            logprob_sum=("null_logprob", "sum"),
        )
    return float((respondent_ll["normalized_weight"] * respondent_ll["logprob_sum"]).sum())


def write_summary(results, frame: pd.DataFrame, summary_path: Path, dataset: str, n_draws: int, threads: int, output_files: list[str]) -> None:
    write_json(
        summary_path,
        {
            "dataset": dataset,
            "n_rows": int(len(frame)),
            "n_respondents": int(frame["respondent_id"].nunique()),
            "n_tasks": int(len(frame)),
            "n_draws": int(n_draws),
            "final_loglikelihood": float(results.data.logLike),
            "null_loglikelihood": float(null_loglikelihood(frame)),
            "number_of_parameters": int(results.data.nparam),
            "number_of_threads": int(threads),
            "optimizer_success": True,
            "optimizer_message": "Biogeme estimate completed",
            "output_files": output_files,
        },
    )


def validate_indicator_columns(frame: pd.DataFrame) -> pd.Series:
    valid = pd.Series(True, index=frame.index)
    for indicator in INDICATOR_NAMES:
        valid = valid & frame[indicator].isin([1, 2, 3, 4, 5, 6])
    return valid


def ai_task_observation_frame(long_frame: pd.DataFrame) -> pd.DataFrame:
    core = long_frame.loc[(long_frame["task_role"] == "core") & (long_frame["is_valid_task_response"] == 1)].copy()
    if core.empty:
        return pd.DataFrame()
    keep_columns = [
        "respondent_id",
        "task_index",
        "alternative_code",
        "chosen",
        "alt_available",
        "alt_time",
        "alt_waiting",
        "alt_cost",
        "alt_distance",
    ]
    core = core[keep_columns].copy()
    grouped = []
    for (respondent_id, task_index), group in core.groupby(["respondent_id", "task_index"], sort=True):
        if set(group["alternative_code"].astype(int).tolist()) != {0, 1, 2}:
            continue
        group = group.sort_values("alternative_code")
        chosen_rows = group.loc[group["chosen"] == 1]
        if len(chosen_rows) != 1:
            continue
        if int(chosen_rows.iloc[0]["alt_available"]) != 1:
            continue
        row = {
            "respondent_id": respondent_id,
            "task_index": int(task_index),
            "Choice": int(chosen_rows.iloc[0]["alternative_code"]),
            "PT_AVAILABLE": int(group.loc[group["alternative_code"] == 0, "alt_available"].iloc[0]),
            "CAR_AVAILABLE": int(group.loc[group["alternative_code"] == 1, "alt_available"].iloc[0]),
            "SLOW_AVAILABLE": int(group.loc[group["alternative_code"] == 2, "alt_available"].iloc[0]),
            "TimePT": float(group.loc[group["alternative_code"] == 0, "alt_time"].iloc[0]),
            "TimeCar": float(group.loc[group["alternative_code"] == 1, "alt_time"].iloc[0]),
            "WaitingTimePT": float(group.loc[group["alternative_code"] == 0, "alt_waiting"].iloc[0]),
            "MarginalCostPT": float(group.loc[group["alternative_code"] == 0, "alt_cost"].iloc[0]),
            "CostCarCHF": float(group.loc[group["alternative_code"] == 1, "alt_cost"].iloc[0]),
            "distance_km": float(group.loc[group["alternative_code"] == 2, "alt_distance"].iloc[0]),
        }
        grouped.append(row)
    if not grouped:
        return pd.DataFrame()
    observations = pd.DataFrame(grouped).sort_values(["respondent_id", "task_index"]).reset_index(drop=True)
    observations["panel_row_index"] = observations.groupby("respondent_id").cumcount() + 1
    observations["TimePT_scaled"] = observations["TimePT"] / TIME_SCALE
    observations["TimeCar_scaled"] = observations["TimeCar"] / TIME_SCALE
    observations["WaitingTimePT_scaled"] = observations["WaitingTimePT"] / WAIT_SCALE
    observations["MarginalCostPT_scaled"] = observations["MarginalCostPT"] / COST_SCALE
    observations["CostCarCHF_scaled"] = observations["CostCarCHF"] / COST_SCALE
    observations["distance_km_scaled"] = observations["distance_km"] / DISTANCE_SCALE
    return observations


def prepare_ai_inputs(experiment_dir: Path, config: dict, max_respondents: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_path = experiment_dir / "ai_panel_block.csv"
    long_path = experiment_dir / "ai_panel_long.csv"
    if not block_path.exists() or not long_path.exists():
        raise FileNotFoundError(f"Missing ai_panel_block.csv or ai_panel_long.csv under {experiment_dir}")
    block_frame = pd.read_csv(block_path)
    long_frame = pd.read_csv(long_path)
    if block_frame.empty or long_frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    block_frame = block_frame.loc[validate_indicator_columns(block_frame)].copy()
    task_frame = ai_task_observation_frame(long_frame)
    if task_frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    expected_core_tasks = n_core_tasks(config)
    valid_counts = task_frame.groupby("respondent_id")["task_index"].nunique()
    valid_ids = valid_counts.loc[valid_counts == expected_core_tasks].index.astype(str).tolist()
    block_frame["respondent_id"] = block_frame["respondent_id"].astype(str)
    task_frame["respondent_id"] = task_frame["respondent_id"].astype(str)
    block_frame = block_frame.loc[block_frame["respondent_id"].isin(valid_ids)].copy()
    if max_respondents is not None:
        keep_ids = block_frame.sort_values("respondent_id").head(int(max_respondents))["respondent_id"].tolist()
        block_frame = block_frame.loc[block_frame["respondent_id"].isin(keep_ids)].copy()
    task_frame = task_frame.loc[task_frame["respondent_id"].isin(block_frame["respondent_id"])].copy()
    merged = task_frame.merge(
        block_frame[
            [
                "respondent_id",
                "normalized_weight",
                "age",
                "CalculatedIncome",
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
                "work_trip",
                "other_trip",
                *INDICATOR_NAMES,
            ]
        ],
        on="respondent_id",
        how="inner",
    )
    merged = merged.sort_values(["respondent_id", "task_index"]).reset_index(drop=True)
    block_out = block_frame.sort_values("respondent_id").reset_index(drop=True)
    return merged, block_out


def prepare_human_inputs(config: dict, max_respondents: int | None) -> pd.DataFrame:
    path = source_data_dir(config) / "human_cleaned_wide.csv"
    frame = pd.read_csv(path).copy().sort_values("respondent_id").reset_index(drop=True)
    frame = frame.loc[validate_indicator_columns(frame)].copy()
    keep_columns = [
        "respondent_id",
        "normalized_weight",
        "Choice",
        "CAR_AVAILABLE",
        "PT_AVAILABLE",
        "SLOW_AVAILABLE",
        "TimePT",
        "TimeCar",
        "WaitingTimePT",
        "MarginalCostPT",
        "CostCarCHF",
        "distance_km",
        "TimePT_scaled",
        "TimeCar_scaled",
        "WaitingTimePT_scaled",
        "MarginalCostPT_scaled",
        "CostCarCHF_scaled",
        "distance_km_scaled",
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
        "work_trip",
        "other_trip",
        *INDICATOR_NAMES,
    ]
    frame = frame[keep_columns].copy()
    frame["respondent_id"] = frame["respondent_id"].astype(str)
    if max_respondents is not None:
        frame = frame.head(int(max_respondents)).copy()
    frame["task_index"] = 1
    frame["panel_row_index"] = 1
    return frame.reset_index(drop=True)


def wide_estimation_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    task_ids = sorted(int(task_id) for task_id in frame["task_index"].drop_duplicates().tolist())
    rows = []
    for respondent_id, group in frame.groupby("respondent_id", sort=True):
        group = group.sort_values("task_index")
        base = {
            "respondent_id": respondent_id,
            "normalized_weight": float(group["normalized_weight"].iloc[0]),
            "high_education": float(group["high_education"].iloc[0]),
            "low_education": float(group["low_education"].iloc[0]),
            "top_manager": float(group["top_manager"].iloc[0]),
            "employees": float(group["employees"].iloc[0]),
            "age_30_less": float(group["age_30_less"].iloc[0]),
            "ScaledIncome": float(group["ScaledIncome"].iloc[0]),
            "car_oriented_parents": float(group["car_oriented_parents"].iloc[0]),
            "childSuburb": float(group["childSuburb"].iloc[0]),
            "city_center_as_kid": float(group["city_center_as_kid"].iloc[0]),
            "artisans": float(group["artisans"].iloc[0]),
        }
        for indicator in INDICATOR_NAMES:
            base[indicator] = int(group[indicator].iloc[0])
        task_map = {int(task_index): row for task_index, row in group.set_index("task_index").iterrows()}
        for task_id in task_ids:
            row = task_map[task_id]
            suffix = f"_t{int(task_id)}"
            base[f"Choice{suffix}"] = int(row["Choice"])
            base[f"CAR_AVAILABLE{suffix}"] = int(row["CAR_AVAILABLE"])
            base[f"TimePT_scaled{suffix}"] = float(row["TimePT_scaled"])
            base[f"TimeCar_scaled{suffix}"] = float(row["TimeCar_scaled"])
            base[f"WaitingTimePT_scaled{suffix}"] = float(row["WaitingTimePT_scaled"])
            base[f"MarginalCostPT_scaled{suffix}"] = float(row["MarginalCostPT_scaled"])
            base[f"CostCarCHF_scaled{suffix}"] = float(row["CostCarCHF_scaled"])
            base[f"distance_km_scaled{suffix}"] = float(row["distance_km_scaled"])
            base[f"work_trip{suffix}"] = float(row["work_trip"])
            base[f"other_trip{suffix}"] = float(row["other_trip"])
        rows.append(base)
    wide = pd.DataFrame(rows).reset_index(drop=True)
    return wide, task_ids


def create_model(database: db.Database, draw_path: Path, n_draws: int, algorithm: str, task_ids: list[int]):
    threads = max(1, (os.cpu_count() or 1) - 1)
    write_parameter_file(n_draws, threads, algorithm)
    register_shared_draws(database, draw_path)

    normalized_weight = Variable("normalized_weight")

    high_education = Variable("high_education")
    low_education = Variable("low_education")
    top_manager = Variable("top_manager")
    employees = Variable("employees")
    age_30_less = Variable("age_30_less")
    ScaledIncome = Variable("ScaledIncome")
    car_oriented_parents = Variable("car_oriented_parents")
    childSuburb = Variable("childSuburb")
    city_center_as_kid = Variable("city_center_as_kid")
    artisans = Variable("artisans")

    beta = {
        name: make_beta(name, lower=1e-6 if name in POSITIVE_PARAMETERS else None)
        for name in PARAMETER_ORDER
    }

    omega_car = bioDraws("omega_car", "SOBOL_CAR")
    omega_env = bioDraws("omega_env", "SOBOL_ENV")

    lv_car = (
        beta["LV_CAR_INTERCEPT"]
        + beta["LV_CAR_HIGH_EDU"] * high_education
        + beta["LV_CAR_TOP_MANAGER"] * top_manager
        + beta["LV_CAR_EMPLOYEES"] * employees
        + beta["LV_CAR_AGE_30_LESS"] * age_30_less
        + beta["LV_CAR_SCALED_INCOME"] * ScaledIncome
        + beta["LV_CAR_PARENTS"] * car_oriented_parents
        + beta["SIGMA_CAR"] * omega_car
    )

    lv_env = (
        beta["LV_ENV_INTERCEPT"]
        + beta["LV_ENV_CHILD_SUBURB"] * childSuburb
        + beta["LV_ENV_SCALED_INCOME"] * ScaledIncome
        + beta["LV_ENV_CITY_CENTER_KID"] * city_center_as_kid
        + beta["LV_ENV_ARTISANS"] * artisans
        + beta["LV_ENV_HIGH_EDU"] * high_education
        + beta["LV_ENV_LOW_EDU"] * low_education
        + beta["SIGMA_ENV"] * omega_env
    )

    choice_probability = 1
    for task_id in task_ids:
        suffix = f"_t{int(task_id)}"
        Choice = Variable(f"Choice{suffix}")
        CAR_AVAILABLE = Variable(f"CAR_AVAILABLE{suffix}")
        TimePT_scaled = Variable(f"TimePT_scaled{suffix}")
        TimeCar_scaled = Variable(f"TimeCar_scaled{suffix}")
        WaitingTimePT_scaled = Variable(f"WaitingTimePT_scaled{suffix}")
        MarginalCostPT_scaled = Variable(f"MarginalCostPT_scaled{suffix}")
        CostCarCHF_scaled = Variable(f"CostCarCHF_scaled{suffix}")
        distance_km_scaled = Variable(f"distance_km_scaled{suffix}")
        work_trip = Variable(f"work_trip{suffix}")
        other_trip = Variable(f"other_trip{suffix}")

        V = {
            0: beta["ASC_PT"]
            + beta["B_COST"] * MarginalCostPT_scaled
            + beta["B_TIME_PT"] * TimePT_scaled
            + beta["B_WAIT_WORK"] * WaitingTimePT_scaled * work_trip
            + beta["B_WAIT_OTHER"] * WaitingTimePT_scaled * other_trip
            + beta["B_LV_CAR_TO_PT"] * lv_car
            + beta["B_LV_ENV_TO_PT"] * lv_env,
            1: beta["ASC_CAR"]
            + beta["B_COST"] * CostCarCHF_scaled
            + beta["B_TIME_CAR"] * TimeCar_scaled
            + beta["B_LV_CAR_TO_CAR"] * lv_car
            + beta["B_LV_ENV_TO_CAR"] * lv_env,
            2: beta["B_DIST_WORK"] * distance_km_scaled * work_trip + beta["B_DIST_OTHER"] * distance_km_scaled * other_trip,
        }
        av = {0: 1, 1: CAR_AVAILABLE, 2: 1}
        choice_probability = choice_probability * models.logit(V, av, Choice)

    measurement_probability = 1
    for indicator_name in INDICATOR_NAMES:
        observed = Variable(indicator_name)
        spec = INDICATOR_SPECS[indicator_name]
        intercept = beta[spec["intercept"]]
        if spec["latent"] == "car":
            loading = spec["loading"] if spec["normalized"] else beta[str(spec["loading"])]
            latent_index = intercept + loading * lv_car
        else:
            loading = spec["loading"] if spec["normalized"] else beta[str(spec["loading"])]
            latent_index = intercept + loading * lv_env
        measurement_probability = measurement_probability * ordered_probit_probability(
            observed, latent_index, beta["DELTA_1"], beta["DELTA_2"]
        )

    conditional_probability = choice_probability * measurement_probability
    logprob = normalized_weight * log(MonteCarlo(conditional_probability))
    biogeme = bio.BIOGEME(database, logprob, parameter_file=str(RUNTIME_PARAMETER_FILE))
    biogeme.modelName = "optima_panel_hcm"
    biogeme.generate_html = False
    biogeme.generate_pickle = False
    for attribute in ("number_of_threads", "numberOfThreads"):
        if hasattr(biogeme, attribute):
            setattr(biogeme, attribute, threads)
    return biogeme, threads


def estimate_dataset(frame: pd.DataFrame, draw_path: Path, n_draws: int, algorithm: str):
    wide_frame, task_ids = wide_estimation_frame(frame)
    work = wide_frame.copy()
    if not pd.api.types.is_numeric_dtype(work["respondent_id"]):
        work["respondent_id"] = pd.factorize(work["respondent_id"])[0] + 1
    database = db.Database("optima_panel_hcm", work)
    biogeme, threads = create_model(database, draw_path, n_draws, algorithm, task_ids)
    results = biogeme.estimate()
    estimates = standardize_estimates(results.getEstimatedParameters(onlyRobust=False))
    return results, estimates, threads


def run_ai_estimation(experiment_dir: Path, config: dict, n_draws: int, max_respondents: int | None) -> None:
    long_frame, block_frame = prepare_ai_inputs(experiment_dir, config, max_respondents)
    if long_frame.empty or block_frame.empty:
        raise RuntimeError(f"No valid AI HCM estimation sample found under {experiment_dir}")
    output_dir = experiment_analysis_dir(experiment_dir, "hcm", "ai")
    long_path = output_dir / "ai_biogeme_hcm_estimation_input_long.csv"
    block_path = output_dir / "ai_biogeme_hcm_estimation_input_block.csv"
    estimate_path = output_dir / "ai_biogeme_hcm_estimates.csv"
    summary_path = output_dir / "ai_biogeme_hcm_summary.json"
    long_frame.to_csv(long_path, index=False)
    block_frame.to_csv(block_path, index=False)
    results, estimates, threads = estimate_dataset(long_frame, draw_file_path(config, n_draws), n_draws, optimization_algorithm(config))
    estimates.to_csv(estimate_path, index=False)
    write_summary(
        results,
        long_frame,
        summary_path,
        "ai",
        n_draws,
        threads,
        [long_path.name, block_path.name, estimate_path.name, summary_path.name],
    )


def run_human_estimation(experiment_dir: Path, config: dict, n_draws: int, max_respondents: int | None) -> None:
    long_frame = prepare_human_inputs(config, max_respondents)
    if long_frame.empty:
        raise RuntimeError("No valid human HCM estimation sample found.")
    output_dir = experiment_analysis_dir(experiment_dir, "hcm", "human")
    long_path = output_dir / "human_baseline_biogeme_hcm_estimation_input_long.csv"
    estimate_path = output_dir / "human_baseline_biogeme_hcm_estimates.csv"
    summary_path = output_dir / "human_baseline_biogeme_hcm_summary.json"
    long_frame.to_csv(long_path, index=False)
    results, estimates, threads = estimate_dataset(long_frame, draw_file_path(config, n_draws), n_draws, optimization_algorithm(config))
    estimates.to_csv(estimate_path, index=False)
    write_summary(
        results,
        long_frame,
        summary_path,
        "human",
        n_draws,
        threads,
        [long_path.name, estimate_path.name, summary_path.name],
    )


def main() -> None:
    args = parse_args()
    experiment_dir = ensure_dir(Path(args.experiment_dir) if Path(args.experiment_dir).is_absolute() else ROOT_DIR / args.experiment_dir)
    config = load_experiment_config(experiment_dir)
    n_draws = n_draws_from_config(config, args.n_draws)
    if args.dataset in {"ai", "both"}:
        run_ai_estimation(experiment_dir, config, n_draws, args.max_respondents)
    if args.dataset in {"human", "both"}:
        run_human_estimation(experiment_dir, config, n_draws, args.max_respondents)


if __name__ == "__main__":
    main()
