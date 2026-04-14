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

from optima_common import AI_COLLECTION_DIR, CONFIG, DATA_DIR, DRAW_NAMES, EXPERIMENT_DIR, INDICATOR_NAMES, OUTPUT_DIR, archive_experiment_config, draw_generator_from_file, ensure_dir, ensure_pt_non_wait_columns, write_json
from optima_hcm_model_spec import INDICATOR_SPECS, PARAMETER_ORDER, POSITIVE_PARAMETERS

RUNTIME_PARAMETER_FILE = Path(__file__).resolve().parents[1] / "biogeme_runtime.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["human", "ai"], required=True)
    parser.add_argument("--n-draws", type=int, default=int(CONFIG["n_monte_carlo_draws_biogeme"]))
    parser.add_argument("--output-subdir", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def output_dir_for(dataset: str, n_draws: int, output_subdir: str | None) -> Path:
    if output_subdir:
        return ensure_dir(OUTPUT_DIR / output_subdir)
    return ensure_dir(OUTPUT_DIR / f"{dataset}_biogeme_{n_draws}")


def load_dataset(dataset: str, max_rows: int | None = None) -> pd.DataFrame:
    if dataset == "human":
        frame = pd.read_csv(DATA_DIR / "human_cleaned_wide.csv")
    else:
        frame = pd.read_csv(AI_COLLECTION_DIR / "ai_cleaned_wide.csv")
        valid_mask = frame["Choice"].isin([0, 1, 2])
        for indicator_name in INDICATOR_NAMES:
            valid_mask = valid_mask & frame[indicator_name].isin([1, 2, 3, 4, 5, 6])
        frame = frame.loc[valid_mask].copy()
    frame = ensure_pt_non_wait_columns(frame)
    if "TimePT_non_wait_scaled" in frame.columns:
        frame["TimePT_scaled"] = frame["TimePT_non_wait_scaled"]
    frame = frame.copy().sort_values("respondent_id").reset_index(drop=True)
    numeric_frame = frame.select_dtypes(include=["number", "bool"]).copy()
    if max_rows is not None:
        numeric_frame = numeric_frame.head(int(max_rows)).copy()
    return numeric_frame


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


def write_parameter_file(n_draws: int, threads: int) -> None:
    RUNTIME_PARAMETER_FILE.write_text(
        "\n".join(
            [
                "[MonteCarlo]",
                f'number_of_draws = {int(n_draws)}',
                "",
                "[MultiThreading]",
                f'number_of_threads = {int(threads)}',
                "",
                "[Estimation]",
                f'optimization_algorithm = "{CONFIG["biogeme"]["optimization_algorithm"]}"',
                'save_iterations = "False"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def create_model(database: db.Database, draw_path: Path, n_draws: int):
    threads = max(1, (os.cpu_count() or 1) - 1)
    write_parameter_file(n_draws, threads)
    register_shared_draws(database, draw_path)

    normalized_weight = Variable("normalized_weight")
    Choice = Variable("Choice")
    CAR_AVAILABLE = Variable("CAR_AVAILABLE")
    TimePT_scaled = Variable("TimePT_scaled")
    TimeCar_scaled = Variable("TimeCar_scaled")
    WaitingTimePT_scaled = Variable("WaitingTimePT_scaled")
    MarginalCostPT_scaled = Variable("MarginalCostPT_scaled")
    CostCarCHF_scaled = Variable("CostCarCHF_scaled")
    distance_km_scaled = Variable("distance_km_scaled")
    work_trip = Variable("work_trip")
    other_trip = Variable("other_trip")

    high_education = Variable("high_education")
    top_manager = Variable("top_manager")
    employees = Variable("employees")
    age_30_less = Variable("age_30_less")
    ScaledIncome = Variable("ScaledIncome")
    car_oriented_parents = Variable("car_oriented_parents")

    childSuburb = Variable("childSuburb")
    city_center_as_kid = Variable("city_center_as_kid")
    artisans = Variable("artisans")
    low_education = Variable("low_education")

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
    choice_probability = models.logit(V, av, Choice)

    indicator_probabilities = []
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
        indicator_probabilities.append(
            ordered_probit_probability(observed, latent_index, beta["DELTA_1"], beta["DELTA_2"])
        )

    conditional_probability = choice_probability
    for probability in indicator_probabilities:
        conditional_probability = conditional_probability * probability

    logprob = normalized_weight * log(MonteCarlo(conditional_probability))
    biogeme = bio.BIOGEME(database, logprob, parameter_file=str(RUNTIME_PARAMETER_FILE))
    biogeme.modelName = "optima_reduced_official_style_hcm"
    biogeme.generate_html = False
    biogeme.generate_pickle = False
    for attribute in ("number_of_threads", "numberOfThreads"):
        if hasattr(biogeme, attribute):
            setattr(biogeme, attribute, threads)
    return biogeme, PARAMETER_ORDER, threads


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


def evaluate_loglikelihood(frame: pd.DataFrame, beta_values: dict[str, float], draw_path: Path, n_draws: int) -> float:
    database = db.Database("optima_eval", frame)
    biogeme, order, _ = create_model(database, draw_path, n_draws)
    vector = [float(beta_values[name]) for name in order]
    return float(biogeme.calculateLikelihood(vector, scaled=False))


def write_summary(results, frame: pd.DataFrame, output_dir: Path, dataset: str, n_draws: int, threads: int) -> dict:
    summary = {
        "dataset": dataset,
        "n_rows": int(len(frame)),
        "n_draws": int(n_draws),
        "final_loglikelihood": float(results.data.logLike),
        "number_of_parameters": int(results.data.nparam),
        "number_of_threads": int(threads),
    }
    write_json(output_dir / "biogeme_hcm_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    archive_experiment_config(EXPERIMENT_DIR)
    output_dir = output_dir_for(args.dataset, args.n_draws, args.output_subdir)
    frame = load_dataset(args.dataset, args.max_rows)
    draw_path = DATA_DIR / f"shared_sobol_draws_{int(args.n_draws)}.npy"
    database = db.Database(f"optima_{args.dataset}", frame)
    biogeme, _, threads = create_model(database, draw_path, int(args.n_draws))
    results = biogeme.estimate()
    estimates = results.getEstimatedParameters(onlyRobust=False)
    standardized = standardize_estimates(estimates)
    standardized.to_csv(output_dir / "biogeme_hcm_estimates.csv", index=False)
    frame.to_csv(output_dir / "estimation_input.csv", index=False)
    write_summary(results, frame, output_dir, args.dataset, args.n_draws, threads)
    print(f"[biogeme_hcm] dataset={args.dataset} rows={len(frame)} draws={args.n_draws} loglik={results.data.logLike:.3f} threads={threads}")


if __name__ == "__main__":
    main()
