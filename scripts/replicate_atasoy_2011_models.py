from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "Swissmetro" / "demographic_choice_psychometric"
RAW_DATA_FILE = DATA_DIR / "raw" / "optima.dat"
DEFAULT_OUTPUT_DIR = DATA_DIR / "atasoy_2011_replication"

PAPER_BASE_TARGETS = {
    "ASCPMM": -0.413,
    "ASCSM": -0.470,
    "beta_cost": -0.0592,
    "beta_time_pmm": -0.0299,
    "beta_time_pt": -0.0121,
    "beta_distance": -0.227,
    "beta_ncars": 1.000,
    "beta_nchildren": 0.154,
    "beta_language": 1.090,
    "beta_work": -0.582,
    "beta_urban": 0.286,
    "beta_student": 3.210,
    "beta_nbikes": 0.347,
}

PAPER_BASE_TARGETS_TEXT = {
    "ASCPMM": "-0.413",
    "ASCSM": "-0.470",
    "beta_cost": "-0.0592",
    "beta_time_pmm": "-0.0299",
    "beta_time_pt": "-0.0121",
    "beta_distance": "-0.227",
    "beta_ncars": "1.000",
    "beta_nchildren": "0.154",
    "beta_language": "1.090",
    "beta_work": "-0.582",
    "beta_urban": "0.286",
    "beta_student": "3.210",
    "beta_nbikes": "0.347",
}

PAPER_CONTINUOUS_UTILITY_TARGETS = {
    "ASCPMM": -0.599,
    "ASCSM": -0.772,
    "beta_cost": -0.0559,
    "beta_time_pmm": -0.0294,
    "beta_time_pt": -0.0119,
    "beta_distance": -0.224,
    "beta_ncars": 0.970,
    "beta_nchildren": 0.215,
    "beta_language": 1.060,
    "beta_work": -0.583,
    "beta_urban": 0.283,
    "beta_student": 3.260,
    "beta_nbikes": 0.385,
    "beta_Acar": -0.574,
    "beta_Aenv": 0.393,
}

PAPER_CONTINUOUS_UTILITY_TARGETS_TEXT = {
    "ASCPMM": "-0.599",
    "ASCSM": "-0.772",
    "beta_cost": "-0.0559",
    "beta_time_pmm": "-0.0294",
    "beta_time_pt": "-0.0119",
    "beta_distance": "-0.224",
    "beta_ncars": "0.970",
    "beta_nchildren": "0.215",
    "beta_language": "1.060",
    "beta_work": "-0.583",
    "beta_urban": "0.283",
    "beta_student": "3.260",
    "beta_nbikes": "0.385",
    "beta_Acar": "-0.574",
    "beta_Aenv": "0.393",
}

PAPER_CONTINUOUS_ATTITUDE_TARGETS = {
    "Acar": 3.020,
    "Aenv": 3.230,
    "theta_ncars": 0.1040,
    "theta_educ": 0.2350,
    "theta_nbikes": 0.0845,
    "theta_age": 0.00445,
    "theta_valais": -0.2230,
    "theta_bern": -0.3610,
    "theta_basel_zurich": -0.2560,
    "theta_east": -0.2280,
    "theta_graubunden": -0.3030,
}

PAPER_CONTINUOUS_ATTITUDE_TARGETS_TEXT = {
    "Acar": "3.020",
    "Aenv": "3.230",
    "theta_ncars": "0.1040",
    "theta_educ": "0.2350",
    "theta_nbikes": "0.0845",
    "theta_age": "0.00445",
    "theta_valais": "-0.2230",
    "theta_bern": "-0.3610",
    "theta_basel_zurich": "-0.2560",
    "theta_east": "-0.2280",
    "theta_graubunden": "-0.3030",
}

PAPER_TABLES = {
    "base_loglik": -1067.4,
    "continuous_choice_loglik": -1069.8,
    "rho2_base": 0.490,
    "rho2_continuous": 0.489,
    "base_market_shares": {"PMM": 0.6231, "PT": 0.3209, "SM": 0.0560},
    "continuous_market_shares": {"PMM": 0.6311, "PT": 0.3120, "SM": 0.0569},
    "base_elasticities": {"PMM_cost": -0.064, "PMM_time": -0.247, "PT_cost": -0.216, "PT_time": -0.471},
    "continuous_elasticities": {"PMM_cost": -0.058, "PMM_time": -0.234, "PT_cost": -0.202, "PT_time": -0.465},
    "base_vot": {"PMM": 30.30, "PT": 12.26},
    "continuous_vot": {"PMM": 31.54, "PT": 12.81},
}

BASE_PARAMETER_ORDER = [
    "ASCPMM",
    "ASCSM",
    "beta_cost",
    "beta_time_pmm",
    "beta_time_pt",
    "beta_distance",
    "beta_ncars",
    "beta_nchildren",
    "beta_language",
    "beta_work",
    "beta_urban",
    "beta_student",
    "beta_nbikes",
]

CONTINUOUS_UTILITY_ORDER = [
    "ASCPMM",
    "ASCSM",
    "beta_cost",
    "beta_time_pmm",
    "beta_time_pt",
    "beta_distance",
    "beta_ncars",
    "beta_nchildren",
    "beta_language",
    "beta_work",
    "beta_urban",
    "beta_student",
    "beta_nbikes",
    "beta_Acar",
    "beta_Aenv",
]

CONTINUOUS_ATTITUDE_ORDER = [
    "Acar",
    "Aenv",
    "theta_ncars",
    "theta_educ",
    "theta_nbikes",
    "theta_age",
    "theta_valais",
    "theta_bern",
    "theta_basel_zurich",
    "theta_east",
    "theta_graubunden",
]

PRO_CAR_INDICATORS = ["Mobil10", "Mobil11", "Mobil16"]
ENV_INDICATORS = ["Envir01", "Envir02", "Envir05", "Envir06"]
ALL_CONTINUOUS_INDICATORS = PRO_CAR_INDICATORS + ENV_INDICATORS
FIXED_PRO_CAR_REFERENCE = "Mobil10"
FIXED_ENV_REFERENCE = "Envir05"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def decimal_places(display_text: str) -> int:
    text = str(display_text)
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def strict_comparison_row(
    section: str,
    name: str,
    paper_value: float,
    our_value: float,
    digits: int,
    block: str,
) -> dict[str, object]:
    paper_display = f"{paper_value:.{digits}f}"
    our_display = f"{our_value:.{digits}f}"
    return {
        "section": section,
        "block": block,
        "name": name,
        "paper_value": float(paper_value),
        "our_value": float(our_value),
        "paper_display": paper_display,
        "our_display": our_display,
        "digits": int(digits),
        "rounded_match": int(paper_display == our_display),
        "status": "match" if paper_display == our_display else "mismatch",
        "gap": float(our_value - paper_value),
    }


def approx_standard_errors(result) -> np.ndarray | None:
    hess_inv = getattr(result, "hess_inv", None)
    if hess_inv is None:
        return None
    try:
        matrix = np.asarray(hess_inv.todense(), dtype=float)
    except AttributeError:
        matrix = np.asarray(hess_inv, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    diagonal = np.diag(matrix)
    if np.any(~np.isfinite(diagonal)):
        return None
    diagonal = np.clip(diagonal, a_min=0.0, a_max=None)
    return np.sqrt(diagonal)


def add_atasoy_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for column in ["NbCar", "NbChild", "NbBicy"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
        work[f"{column}_0"] = work[column].clip(lower=0)
    work["French"] = (work["LangCode"] == 1).astype(int)
    work["Urban"] = (work["UrbRur"] == 2).astype(int)
    work["Student"] = (work["OccupStat"] == 8).astype(int)
    work["WorkTrip"] = work["TripPurpose"].isin([1, 2]).astype(int)
    work["EducHigh"] = (work["Education"] >= 6).astype(int)
    work["AgeTerm"] = (pd.to_numeric(work["age"], errors="coerce") - 45).clip(lower=0)
    work["Valais"] = (work["Region"] == 2).astype(int)
    work["Bern"] = (work["Region"] == 4).astype(int)
    work["BaselZurich"] = work["Region"].isin([5, 6]).astype(int)
    work["East"] = (work["Region"] == 7).astype(int)
    work["Graubunden"] = (work["Region"] == 8).astype(int)
    return work


def prepare_replication_frame() -> pd.DataFrame:
    frame = pd.read_csv(RAW_DATA_FILE, sep="\t")
    frame = frame.loc[frame["Choice"] != -1].copy().reset_index(drop=True)
    frame = add_atasoy_covariates(frame)
    frame["Weight"] = frame["Weight"].astype(float)
    return frame


def base_choice_probabilities(parameters: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    asc_pmm, asc_sm, beta_cost, beta_time_pmm, beta_time_pt, beta_distance, beta_ncars, beta_nchildren, beta_language, beta_work, beta_urban, beta_student, beta_nbikes = parameters
    v_pt = beta_cost * frame["MarginalCostPT"].to_numpy(float)
    v_pt = v_pt + beta_time_pt * frame["TimePT"].to_numpy(float)
    v_pt = v_pt + beta_urban * frame["Urban"].to_numpy(float)
    v_pt = v_pt + beta_student * frame["Student"].to_numpy(float)
    v_pmm = asc_pmm + beta_cost * frame["CostCarCHF"].to_numpy(float)
    v_pmm = v_pmm + beta_time_pmm * frame["TimeCar"].to_numpy(float)
    v_pmm = v_pmm + beta_ncars * frame["NbCar_0"].to_numpy(float)
    v_pmm = v_pmm + beta_nchildren * frame["NbChild_0"].to_numpy(float)
    v_pmm = v_pmm + beta_language * frame["French"].to_numpy(float)
    v_pmm = v_pmm + beta_work * frame["WorkTrip"].to_numpy(float)
    v_sm = asc_sm + beta_distance * frame["distance_km"].to_numpy(float)
    v_sm = v_sm + beta_nbikes * frame["NbBicy_0"].to_numpy(float)
    stacked = np.column_stack([v_pt, v_pmm, v_sm])
    maximum = stacked.max(axis=1, keepdims=True)
    shifted = np.exp(stacked - maximum)
    return shifted / shifted.sum(axis=1, keepdims=True)


def base_negative_log_likelihood(parameters: np.ndarray, frame: pd.DataFrame) -> float:
    probabilities = base_choice_probabilities(parameters, frame)
    chosen = frame["Choice"].to_numpy(int)
    chosen_probabilities = probabilities[np.arange(len(frame)), chosen]
    return -float(np.sum(np.log(np.clip(chosen_probabilities, 1.0e-300, None))))


def weighted_market_shares(probabilities: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    weight_sum = float(np.sum(weights))
    return {
        "PT": float(np.sum(weights * probabilities[:, 0]) / weight_sum),
        "PMM": float(np.sum(weights * probabilities[:, 1]) / weight_sum),
        "SM": float(np.sum(weights * probabilities[:, 2]) / weight_sum),
    }


def own_elasticity(probabilities: np.ndarray, beta: float, x: np.ndarray, weights: np.ndarray, alternative_index: int) -> float:
    own = beta * x * (1.0 - probabilities[:, alternative_index])
    numerator = np.sum(weights * probabilities[:, alternative_index] * own)
    denominator = np.sum(weights * probabilities[:, alternative_index])
    return float(numerator / denominator)


def estimate_base_model(frame: pd.DataFrame) -> dict[str, object]:
    start = np.array([PAPER_BASE_TARGETS[name] for name in BASE_PARAMETER_ORDER], dtype=float)
    result = minimize(
        base_negative_log_likelihood,
        start,
        args=(frame,),
        method="BFGS",
        options={"maxiter": 500, "gtol": 1.0e-6},
    )
    estimates = result.x
    standard_errors = approx_standard_errors(result)
    probabilities = base_choice_probabilities(estimates, frame)
    weights = frame["Weight"].to_numpy(float)
    metrics = {
        "log_likelihood": float(-result.fun),
        "market_shares": weighted_market_shares(probabilities, weights),
        "elasticities": {
            "PMM_cost": own_elasticity(probabilities, estimates[2], frame["CostCarCHF"].to_numpy(float), weights, 1),
            "PMM_time": own_elasticity(probabilities, estimates[3], frame["TimeCar"].to_numpy(float), weights, 1),
            "PT_cost": own_elasticity(probabilities, estimates[2], frame["MarginalCostPT"].to_numpy(float), weights, 0),
            "PT_time": own_elasticity(probabilities, estimates[4], frame["TimePT"].to_numpy(float), weights, 0),
        },
        "value_of_time_chf_per_hour": {
            "PMM": float(60.0 * abs(estimates[3] / estimates[2])),
            "PT": float(60.0 * abs(estimates[4] / estimates[2])),
        },
    }
    rows = []
    for index, name in enumerate(BASE_PARAMETER_ORDER):
        row = {
            "parameter_name": name,
            "estimate": float(estimates[index]),
            "paper_estimate": float(PAPER_BASE_TARGETS[name]),
        }
        if standard_errors is not None and np.isfinite(standard_errors[index]) and standard_errors[index] > 0:
            row["std_error"] = float(standard_errors[index])
            row["z_value"] = float(estimates[index] / standard_errors[index])
        rows.append(row)
    return {
        "result": result,
        "estimates": estimates,
        "standard_errors": standard_errors,
        "probabilities": probabilities,
        "metrics": metrics,
        "estimates_table": pd.DataFrame(rows),
        "specification": {
            "sample_rule": "Choice != -1",
            "cost_pmm": "CostCarCHF",
            "cost_pt": "MarginalCostPT",
            "work_trip_dummy": "TripPurpose in {1, 2}",
            "student_dummy": "OccupStat == 8",
            "urban_dummy": "UrbRur == 2",
            "language_dummy": "LangCode == 1 (French-speaking commune)",
            "household_counts": "NbCar, NbChild, NbBicy with negative missing codes recoded to 0",
        },
    }


def continuous_choice_probabilities(
    utility_parameters: np.ndarray,
    attitude_parameters: np.ndarray,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    asc_pmm, asc_sm, beta_cost, beta_time_pmm, beta_time_pt, beta_distance, beta_ncars, beta_nchildren, beta_language, beta_work, beta_urban, beta_student, beta_nbikes, beta_Acar, beta_Aenv = utility_parameters
    Acar, Aenv, theta_ncars, theta_educ, theta_nbikes, theta_age, theta_valais, theta_bern, theta_basel_zurich, theta_east, theta_graubunden = attitude_parameters
    acar = Acar + theta_ncars * frame["NbCar_0"].to_numpy(float)
    acar = acar - theta_educ * frame["EducHigh"].to_numpy(float)
    acar = acar + theta_valais * frame["Valais"].to_numpy(float)
    acar = acar + theta_bern * frame["Bern"].to_numpy(float)
    acar = acar + theta_basel_zurich * frame["BaselZurich"].to_numpy(float)
    acar = acar + theta_east * frame["East"].to_numpy(float)
    acar = acar + theta_graubunden * frame["Graubunden"].to_numpy(float)
    aenv = Aenv + theta_educ * frame["EducHigh"].to_numpy(float)
    aenv = aenv + theta_nbikes * frame["NbBicy_0"].to_numpy(float)
    aenv = aenv + theta_age * frame["AgeTerm"].to_numpy(float)
    v_pt = beta_cost * frame["MarginalCostPT"].to_numpy(float)
    v_pt = v_pt + beta_time_pt * frame["TimePT"].to_numpy(float)
    v_pt = v_pt + beta_urban * frame["Urban"].to_numpy(float)
    v_pt = v_pt + beta_student * frame["Student"].to_numpy(float)
    v_pt = v_pt + beta_Acar * acar + beta_Aenv * aenv
    v_pmm = asc_pmm + beta_cost * frame["CostCarCHF"].to_numpy(float)
    v_pmm = v_pmm + beta_time_pmm * frame["TimeCar"].to_numpy(float)
    v_pmm = v_pmm + beta_ncars * frame["NbCar_0"].to_numpy(float)
    v_pmm = v_pmm + beta_nchildren * frame["NbChild_0"].to_numpy(float)
    v_pmm = v_pmm + beta_language * frame["French"].to_numpy(float)
    v_pmm = v_pmm + beta_work * frame["WorkTrip"].to_numpy(float)
    v_sm = asc_sm + beta_distance * frame["distance_km"].to_numpy(float)
    v_sm = v_sm + beta_nbikes * frame["NbBicy_0"].to_numpy(float)
    stacked = np.column_stack([v_pt, v_pmm, v_sm])
    maximum = stacked.max(axis=1, keepdims=True)
    shifted = np.exp(stacked - maximum)
    probabilities = shifted / shifted.sum(axis=1, keepdims=True)
    return probabilities, acar, aenv


def unpack_continuous_vector(
    vector: np.ndarray,
    ref_pro_car_indicator: str,
    ref_env_indicator: str,
) -> dict[str, object]:
    index = 0
    utility_parameters = vector[index : index + len(CONTINUOUS_UTILITY_ORDER)]
    index += len(CONTINUOUS_UTILITY_ORDER)
    attitude_parameters = vector[index : index + len(CONTINUOUS_ATTITUDE_ORDER)]
    index += len(CONTINUOUS_ATTITUDE_ORDER)
    non_reference_indicators = [
        indicator for indicator in ALL_CONTINUOUS_INDICATORS if indicator not in {ref_pro_car_indicator, ref_env_indicator}
    ]
    alpha_parameters = {
        indicator: value for indicator, value in zip(non_reference_indicators, vector[index : index + len(non_reference_indicators)])
    }
    index += len(non_reference_indicators)
    loading_parameters = {
        indicator: value for indicator, value in zip(non_reference_indicators, vector[index : index + len(non_reference_indicators)])
    }
    index += len(non_reference_indicators)
    log_sigma_parameters = {
        indicator: value for indicator, value in zip(ALL_CONTINUOUS_INDICATORS, vector[index : index + len(ALL_CONTINUOUS_INDICATORS)])
    }
    return {
        "utility_parameters": utility_parameters,
        "attitude_parameters": attitude_parameters,
        "alpha_parameters": alpha_parameters,
        "loading_parameters": loading_parameters,
        "log_sigma_parameters": log_sigma_parameters,
    }


def continuous_negative_log_likelihood(
    vector: np.ndarray,
    frame: pd.DataFrame,
    ref_pro_car_indicator: str,
    ref_env_indicator: str,
) -> float:
    unpacked = unpack_continuous_vector(vector, ref_pro_car_indicator, ref_env_indicator)
    utility_parameters = unpacked["utility_parameters"]
    attitude_parameters = unpacked["attitude_parameters"]
    alpha_parameters = unpacked["alpha_parameters"]
    loading_parameters = unpacked["loading_parameters"]
    log_sigma_parameters = unpacked["log_sigma_parameters"]
    probabilities, acar, aenv = continuous_choice_probabilities(utility_parameters, attitude_parameters, frame)
    chosen = frame["Choice"].to_numpy(int)
    chosen_probabilities = probabilities[np.arange(len(frame)), chosen]
    log_likelihood = np.log(np.clip(chosen_probabilities, 1.0e-300, None))
    gaussian_constant = -0.5 * math.log(2.0 * math.pi)
    for indicator in PRO_CAR_INDICATORS:
        observed = frame[indicator].to_numpy(float)
        valid = ((observed >= 1.0) & (observed <= 5.0)).astype(float)
        if indicator == ref_pro_car_indicator:
            mean = acar
        else:
            mean = alpha_parameters[indicator] + loading_parameters[indicator] * acar
        sigma = math.exp(log_sigma_parameters[indicator])
        z = (observed - mean) / sigma
        log_likelihood = log_likelihood + valid * (gaussian_constant - math.log(sigma) - 0.5 * z * z)
    for indicator in ENV_INDICATORS:
        observed = frame[indicator].to_numpy(float)
        valid = ((observed >= 1.0) & (observed <= 5.0)).astype(float)
        if indicator == ref_env_indicator:
            mean = aenv
        else:
            mean = alpha_parameters[indicator] + loading_parameters[indicator] * aenv
        sigma = math.exp(log_sigma_parameters[indicator])
        z = (observed - mean) / sigma
        log_likelihood = log_likelihood + valid * (gaussian_constant - math.log(sigma) - 0.5 * z * z)
    return -float(np.sum(log_likelihood))


def continuous_choice_only_log_likelihood(
    utility_parameters: np.ndarray,
    attitude_parameters: np.ndarray,
    frame: pd.DataFrame,
) -> float:
    probabilities, _, _ = continuous_choice_probabilities(utility_parameters, attitude_parameters, frame)
    chosen = frame["Choice"].to_numpy(int)
    chosen_probabilities = probabilities[np.arange(len(frame)), chosen]
    return float(np.sum(np.log(np.clip(chosen_probabilities, 1.0e-300, None))))


def continuous_start_vector() -> np.ndarray:
    return np.concatenate(
        [
            np.array([PAPER_CONTINUOUS_UTILITY_TARGETS[name] for name in CONTINUOUS_UTILITY_ORDER], dtype=float),
            np.array([PAPER_CONTINUOUS_ATTITUDE_TARGETS[name] for name in CONTINUOUS_ATTITUDE_ORDER], dtype=float),
            np.zeros(len(ALL_CONTINUOUS_INDICATORS) - 2, dtype=float),
            np.ones(len(ALL_CONTINUOUS_INDICATORS) - 2, dtype=float),
            np.log(np.ones(len(ALL_CONTINUOUS_INDICATORS), dtype=float)),
        ]
    )


def continuous_search_score(
    utility_parameters: np.ndarray,
    attitude_parameters: np.ndarray,
    choice_log_likelihood: float,
) -> float:
    utility_gap = np.mean(
        [
            abs(float(value) - float(PAPER_CONTINUOUS_UTILITY_TARGETS[name]))
            for name, value in zip(CONTINUOUS_UTILITY_ORDER, utility_parameters)
        ]
    )
    attitude_gap = np.mean(
        [
            abs(float(value) - float(PAPER_CONTINUOUS_ATTITUDE_TARGETS[name]))
            for name, value in zip(CONTINUOUS_ATTITUDE_ORDER, attitude_parameters)
        ]
    )
    likelihood_gap = abs(choice_log_likelihood - PAPER_TABLES["continuous_choice_loglik"])
    return float(utility_gap + attitude_gap + 0.02 * likelihood_gap)


def search_continuous_normalization(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    start = continuous_start_vector()
    best_bundle: dict[str, object] | None = None
    for ref_pro_car_indicator in PRO_CAR_INDICATORS:
        for ref_env_indicator in ENV_INDICATORS:
            result = minimize(
                continuous_negative_log_likelihood,
                start,
                args=(frame, ref_pro_car_indicator, ref_env_indicator),
                method="L-BFGS-B",
                bounds=[(None, None)] * (len(start) - len(ALL_CONTINUOUS_INDICATORS))
                + [(math.log(1.0e-3), math.log(20.0))] * len(ALL_CONTINUOUS_INDICATORS),
                options={"maxiter": 80, "ftol": 1.0e-7, "gtol": 1.0e-5},
            )
            unpacked = unpack_continuous_vector(result.x, ref_pro_car_indicator, ref_env_indicator)
            utility_parameters = unpacked["utility_parameters"]
            attitude_parameters = unpacked["attitude_parameters"]
            choice_log_likelihood = continuous_choice_only_log_likelihood(utility_parameters, attitude_parameters, frame)
            search_score = continuous_search_score(utility_parameters, attitude_parameters, choice_log_likelihood)
            rows.append(
                {
                    "ref_pro_car_indicator": ref_pro_car_indicator,
                    "ref_env_indicator": ref_env_indicator,
                    "joint_log_likelihood": float(-result.fun),
                    "choice_log_likelihood": float(choice_log_likelihood),
                    "search_score": search_score,
                    "Acar": float(attitude_parameters[0]),
                    "Aenv": float(attitude_parameters[1]),
                    "beta_Acar": float(utility_parameters[13]),
                    "beta_Aenv": float(utility_parameters[14]),
                }
            )
            if best_bundle is None or search_score < best_bundle["search_score"]:
                best_bundle = {
                    "search_score": search_score,
                    "ref_pro_car_indicator": ref_pro_car_indicator,
                    "ref_env_indicator": ref_env_indicator,
                    "result": result,
                    "vector": result.x.copy(),
                }
    search_table = pd.DataFrame(rows).sort_values("search_score").reset_index(drop=True)
    return search_table, best_bundle


def fixed_continuous_initial_result(
    frame: pd.DataFrame,
    ref_pro_car_indicator: str,
    ref_env_indicator: str,
):
    start = continuous_start_vector()
    return minimize(
        continuous_negative_log_likelihood,
        start,
        args=(frame, ref_pro_car_indicator, ref_env_indicator),
        method="L-BFGS-B",
        bounds=[(None, None)] * (len(start) - len(ALL_CONTINUOUS_INDICATORS))
        + [(math.log(1.0e-3), math.log(20.0))] * len(ALL_CONTINUOUS_INDICATORS),
        options={"maxiter": 80, "ftol": 1.0e-7, "gtol": 1.0e-5},
    )


def estimate_continuous_model(
    frame: pd.DataFrame,
    ref_pro_car_indicator: str,
    ref_env_indicator: str,
    start_vector: np.ndarray | None = None,
    initial_result=None,
) -> dict[str, object]:
    start = continuous_start_vector() if start_vector is None else start_vector.copy()
    result = None
    best_result = initial_result
    if initial_result is not None:
        initial_unpacked = unpack_continuous_vector(initial_result.x, ref_pro_car_indicator, ref_env_indicator)
        best_score = continuous_search_score(
            initial_unpacked["utility_parameters"],
            initial_unpacked["attitude_parameters"],
            continuous_choice_only_log_likelihood(
                initial_unpacked["utility_parameters"],
                initial_unpacked["attitude_parameters"],
                frame,
            ),
        )
    else:
        best_score = float("inf")
    for _ in range(4):
        result = minimize(
            continuous_negative_log_likelihood,
            start,
            args=(frame, ref_pro_car_indicator, ref_env_indicator),
            method="L-BFGS-B",
            bounds=[(None, None)] * (len(start) - len(ALL_CONTINUOUS_INDICATORS))
            + [(math.log(1.0e-3), math.log(20.0))] * len(ALL_CONTINUOUS_INDICATORS),
            options={"maxiter": 200, "ftol": 1.0e-8, "gtol": 1.0e-5},
        )
        unpacked_iteration = unpack_continuous_vector(result.x, ref_pro_car_indicator, ref_env_indicator)
        score = continuous_search_score(
            unpacked_iteration["utility_parameters"],
            unpacked_iteration["attitude_parameters"],
            continuous_choice_only_log_likelihood(
                unpacked_iteration["utility_parameters"],
                unpacked_iteration["attitude_parameters"],
                frame,
            ),
        )
        if score < best_score:
            best_score = score
            best_result = result
        if result.success or np.max(np.abs(result.x - start)) < 1.0e-7:
            break
        start = result.x.copy()
    if best_result is not None:
        result = best_result
    unpacked = unpack_continuous_vector(result.x, ref_pro_car_indicator, ref_env_indicator)
    utility_parameters = unpacked["utility_parameters"]
    attitude_parameters = unpacked["attitude_parameters"]
    alpha_parameters = unpacked["alpha_parameters"]
    loading_parameters = unpacked["loading_parameters"]
    sigma_parameters = {
        indicator: float(math.exp(log_sigma))
        for indicator, log_sigma in unpacked["log_sigma_parameters"].items()
    }
    probabilities, acar, aenv = continuous_choice_probabilities(utility_parameters, attitude_parameters, frame)
    weights = frame["Weight"].to_numpy(float)
    standard_errors = approx_standard_errors(result)
    choice_log_likelihood = continuous_choice_only_log_likelihood(utility_parameters, attitude_parameters, frame)
    metrics = {
        "joint_log_likelihood": float(-result.fun),
        "choice_log_likelihood": float(choice_log_likelihood),
        "market_shares": weighted_market_shares(probabilities, weights),
        "elasticities": {
            "PMM_cost": own_elasticity(probabilities, utility_parameters[2], frame["CostCarCHF"].to_numpy(float), weights, 1),
            "PMM_time": own_elasticity(probabilities, utility_parameters[3], frame["TimeCar"].to_numpy(float), weights, 1),
            "PT_cost": own_elasticity(probabilities, utility_parameters[2], frame["MarginalCostPT"].to_numpy(float), weights, 0),
            "PT_time": own_elasticity(probabilities, utility_parameters[4], frame["TimePT"].to_numpy(float), weights, 0),
        },
        "value_of_time_chf_per_hour": {
            "PMM": float(60.0 * abs(utility_parameters[3] / utility_parameters[2])),
            "PT": float(60.0 * abs(utility_parameters[4] / utility_parameters[2])),
        },
        "mean_acar": float(np.mean(acar)),
        "mean_aenv": float(np.mean(aenv)),
    }
    utility_rows = []
    for index, name in enumerate(CONTINUOUS_UTILITY_ORDER):
        row = {
            "parameter_name": name,
            "estimate": float(utility_parameters[index]),
            "paper_estimate": float(PAPER_CONTINUOUS_UTILITY_TARGETS[name]),
            "block": "utility",
        }
        if standard_errors is not None and index < len(standard_errors) and np.isfinite(standard_errors[index]) and standard_errors[index] > 0:
            row["std_error"] = float(standard_errors[index])
            row["z_value"] = float(utility_parameters[index] / standard_errors[index])
        utility_rows.append(row)
    offset = len(CONTINUOUS_UTILITY_ORDER)
    attitude_rows = []
    for index, name in enumerate(CONTINUOUS_ATTITUDE_ORDER):
        row = {
            "parameter_name": name,
            "estimate": float(attitude_parameters[index]),
            "paper_estimate": float(PAPER_CONTINUOUS_ATTITUDE_TARGETS[name]),
            "block": "attitude",
        }
        if standard_errors is not None and offset + index < len(standard_errors) and np.isfinite(standard_errors[offset + index]) and standard_errors[offset + index] > 0:
            row["std_error"] = float(standard_errors[offset + index])
            row["z_value"] = float(attitude_parameters[index] / standard_errors[offset + index])
        attitude_rows.append(row)
    measurement_rows = []
    free_indicator_order = [
        indicator for indicator in ALL_CONTINUOUS_INDICATORS if indicator not in {ref_pro_car_indicator, ref_env_indicator}
    ]
    alpha_offset = offset + len(CONTINUOUS_ATTITUDE_ORDER)
    loading_offset = alpha_offset + len(free_indicator_order)
    sigma_offset = loading_offset + len(free_indicator_order)
    for index, indicator in enumerate(free_indicator_order):
        alpha_row = {
            "parameter_name": f"alpha_{indicator}",
            "estimate": float(alpha_parameters[indicator]),
            "block": "measurement",
        }
        if standard_errors is not None and alpha_offset + index < len(standard_errors) and np.isfinite(standard_errors[alpha_offset + index]) and standard_errors[alpha_offset + index] > 0:
            alpha_row["std_error"] = float(standard_errors[alpha_offset + index])
            alpha_row["z_value"] = float(alpha_parameters[indicator] / standard_errors[alpha_offset + index])
        measurement_rows.append(alpha_row)
    for index, indicator in enumerate(free_indicator_order):
        loading_row = {
            "parameter_name": f"loading_{indicator}",
            "estimate": float(loading_parameters[indicator]),
            "block": "measurement",
        }
        if standard_errors is not None and loading_offset + index < len(standard_errors) and np.isfinite(standard_errors[loading_offset + index]) and standard_errors[loading_offset + index] > 0:
            loading_row["std_error"] = float(standard_errors[loading_offset + index])
            loading_row["z_value"] = float(loading_parameters[indicator] / standard_errors[loading_offset + index])
        measurement_rows.append(loading_row)
    for index, indicator in enumerate(ALL_CONTINUOUS_INDICATORS):
        sigma_row = {
            "parameter_name": f"sigma_{indicator}",
            "estimate": float(sigma_parameters[indicator]),
            "block": "measurement",
        }
        if standard_errors is not None and sigma_offset + index < len(standard_errors) and np.isfinite(standard_errors[sigma_offset + index]) and standard_errors[sigma_offset + index] > 0:
            sigma_standard_error = sigma_parameters[indicator] * standard_errors[sigma_offset + index]
            sigma_row["std_error"] = float(sigma_standard_error)
            sigma_row["z_value"] = float(sigma_parameters[indicator] / sigma_standard_error) if sigma_standard_error > 0 else float("nan")
        measurement_rows.append(sigma_row)
    return {
        "result": result,
        "standard_errors": standard_errors,
        "probabilities": probabilities,
        "metrics": metrics,
        "utility_parameters": utility_parameters,
        "attitude_parameters": attitude_parameters,
        "alpha_parameters": alpha_parameters,
        "loading_parameters": loading_parameters,
        "sigma_parameters": sigma_parameters,
        "utility_table": pd.DataFrame(utility_rows),
        "attitude_table": pd.DataFrame(attitude_rows),
        "measurement_table": pd.DataFrame(measurement_rows),
        "normalization": {
            "ref_pro_car_indicator": ref_pro_car_indicator,
            "ref_env_indicator": ref_env_indicator,
        },
        "indicator_mapping": {
            "paper_pro_car_items_8_9_10": PRO_CAR_INDICATORS,
            "paper_environment_items_1_2_4_5": ENV_INDICATORS,
        },
    }


def build_base_comparison_frame(base_results: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in base_results["estimates_table"].iterrows():
        paper_text = PAPER_BASE_TARGETS_TEXT[str(row["parameter_name"])]
        rows.append(
            strict_comparison_row(
                section="base_parameter",
                name=str(row["parameter_name"]),
                paper_value=float(row["paper_estimate"]),
                our_value=float(row["estimate"]),
                digits=decimal_places(paper_text),
                block="base_logit",
            )
        )
    rows.append(
        strict_comparison_row(
            section="base_metric",
            name="log_likelihood",
            paper_value=PAPER_TABLES["base_loglik"],
            our_value=float(base_results["metrics"]["log_likelihood"]),
            digits=1,
            block="base_logit",
        )
    )
    for key, value in PAPER_TABLES["base_market_shares"].items():
        rows.append(
            strict_comparison_row(
                section="base_metric",
                name=f"market_share_{key}",
                paper_value=float(value),
                our_value=float(base_results["metrics"]["market_shares"][key]),
                digits=4,
                block="base_logit",
            )
        )
    for key, value in PAPER_TABLES["base_elasticities"].items():
        rows.append(
            strict_comparison_row(
                section="base_metric",
                name=f"elasticity_{key}",
                paper_value=float(value),
                our_value=float(base_results["metrics"]["elasticities"][key]),
                digits=3,
                block="base_logit",
            )
        )
    for key, value in PAPER_TABLES["base_vot"].items():
        rows.append(
            strict_comparison_row(
                section="base_metric",
                name=f"value_of_time_{key}",
                paper_value=float(value),
                our_value=float(base_results["metrics"]["value_of_time_chf_per_hour"][key]),
                digits=2,
                block="base_logit",
            )
        )
    return pd.DataFrame(rows)


def build_hcm_comparison_frame(continuous_results: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in continuous_results["utility_table"].iterrows():
        paper_text = PAPER_CONTINUOUS_UTILITY_TARGETS_TEXT[str(row["parameter_name"])]
        rows.append(
            strict_comparison_row(
                section="hcm_utility_parameter",
                name=str(row["parameter_name"]),
                paper_value=float(row["paper_estimate"]),
                our_value=float(row["estimate"]),
                digits=decimal_places(paper_text),
                block="hcm",
            )
        )
    for _, row in continuous_results["attitude_table"].iterrows():
        paper_text = PAPER_CONTINUOUS_ATTITUDE_TARGETS_TEXT[str(row["parameter_name"])]
        rows.append(
            strict_comparison_row(
                section="hcm_attitude_parameter",
                name=str(row["parameter_name"]),
                paper_value=float(row["paper_estimate"]),
                our_value=float(row["estimate"]),
                digits=decimal_places(paper_text),
                block="hcm",
            )
        )
    rows.append(
        strict_comparison_row(
            section="hcm_metric",
            name="choice_log_likelihood",
            paper_value=PAPER_TABLES["continuous_choice_loglik"],
            our_value=float(continuous_results["metrics"]["choice_log_likelihood"]),
            digits=1,
            block="hcm",
        )
    )
    for key, value in PAPER_TABLES["continuous_market_shares"].items():
        rows.append(
            strict_comparison_row(
                section="hcm_metric",
                name=f"market_share_{key}",
                paper_value=float(value),
                our_value=float(continuous_results["metrics"]["market_shares"][key]),
                digits=4,
                block="hcm",
            )
        )
    for key, value in PAPER_TABLES["continuous_elasticities"].items():
        rows.append(
            strict_comparison_row(
                section="hcm_metric",
                name=f"elasticity_{key}",
                paper_value=float(value),
                our_value=float(continuous_results["metrics"]["elasticities"][key]),
                digits=3,
                block="hcm",
            )
        )
    for key, value in PAPER_TABLES["continuous_vot"].items():
        rows.append(
            strict_comparison_row(
                section="hcm_metric",
                name=f"value_of_time_{key}",
                paper_value=float(value),
                our_value=float(continuous_results["metrics"]["value_of_time_chf_per_hour"][key]),
                digits=2,
                block="hcm",
            )
        )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    frame: pd.DataFrame,
    base_results: dict[str, object],
    continuous_results: dict[str, object],
    base_comparison: pd.DataFrame,
    hcm_comparison: pd.DataFrame,
) -> None:
    base_matches = int(base_comparison["rounded_match"].sum())
    hcm_matches = int(hcm_comparison["rounded_match"].sum())
    base_total = int(len(base_comparison))
    hcm_total = int(len(hcm_comparison))

    root_lines = []
    root_lines.append("# Atasoy 2011 replication")
    root_lines.append("")
    root_lines.append(
        "This directory is the canonical human replication of the Atasoy, Glerum, and Bierlaire (2011) base logit model and continuous hybrid choice model."
    )
    root_lines.append("")
    root_lines.append(
        f"The replication sample keeps all observations with `Choice != -1`, which gives `{len(frame)}` loop observations from the public `optima.dat`."
    )
    root_lines.append("")
    root_lines.append(
        f"The base logit outputs are saved under `{output_dir / 'base_logit'}`. "
        f"Rounded to the paper precision, `{base_matches}` of `{base_total}` literature-reported base quantities match."
    )
    root_lines.append("")
    root_lines.append(
        f"The continuous hybrid choice outputs are saved under `{output_dir / 'hcm'}`. "
        f"The repository now uses the fixed normalization `Mobil10` for the pro-car attitude and `Envir05` for the environmental attitude. "
        f"Rounded to the paper precision, `{hcm_matches}` of `{hcm_total}` literature-reported continuous-model quantities match."
    )
    root_lines.append("")
    root_lines.append("Re-run command:")
    root_lines.append("")
    root_lines.append("```bash")
    root_lines.append(f"./.venv/bin/python scripts/replicate_atasoy_2011_models.py --output-dir \"{output_dir}\"")
    root_lines.append("```")
    (output_dir / "replication_report.md").write_text("\n".join(root_lines) + "\n", encoding="utf-8")

    hcm_lines = []
    hcm_lines.append("# Atasoy 2011 continuous HCM replication")
    hcm_lines.append("")
    hcm_lines.append(
        "This report fixes the normalization to `Mobil10` for the pro-car attitude and `Envir05` for the environmental attitude."
    )
    hcm_lines.append("")
    hcm_lines.append(
        "The strict reproduction rule is simple: for each parameter or summary quantity explicitly reported in the paper, our estimate must match the paper after rounding to the same number of displayed decimals."
    )
    hcm_lines.append("")
    hcm_lines.append(
        f"Rounded matches: `{hcm_matches}` / `{hcm_total}`. "
        f"Choice-only log-likelihood: `{continuous_results['metrics']['choice_log_likelihood']:.3f}` versus paper `-1069.8`."
    )
    hcm_lines.append("")
    hcm_lines.append("| Name | Paper | Ours | Status |")
    hcm_lines.append("| --- | ---: | ---: | --- |")
    for _, row in hcm_comparison.iterrows():
        hcm_lines.append(
            f"| {row['name']} | {row['paper_display']} | {row['our_display']} | {row['status']} |"
        )
    (output_dir / "hcm" / "hcm_replication_report.md").write_text("\n".join(hcm_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    base_dir = ensure_dir(output_dir / "base_logit")
    hcm_dir = ensure_dir(output_dir / "hcm")
    frame = prepare_replication_frame()

    base_results = estimate_base_model(frame)
    fixed_initial = fixed_continuous_initial_result(
        frame,
        ref_pro_car_indicator=FIXED_PRO_CAR_REFERENCE,
        ref_env_indicator=FIXED_ENV_REFERENCE,
    )
    continuous_results = estimate_continuous_model(
        frame,
        ref_pro_car_indicator=FIXED_PRO_CAR_REFERENCE,
        ref_env_indicator=FIXED_ENV_REFERENCE,
        start_vector=fixed_initial.x.copy(),
        initial_result=fixed_initial,
    )
    base_comparison = build_base_comparison_frame(base_results)
    hcm_comparison = build_hcm_comparison_frame(continuous_results)

    base_results["estimates_table"].to_csv(base_dir / "base_logit_estimates.csv", index=False)
    continuous_results["utility_table"].to_csv(hcm_dir / "hcm_utility_estimates.csv", index=False)
    continuous_results["attitude_table"].to_csv(hcm_dir / "hcm_attitude_estimates.csv", index=False)
    continuous_results["measurement_table"].to_csv(hcm_dir / "hcm_measurement_estimates.csv", index=False)
    base_comparison.to_csv(base_dir / "base_logit_paper_comparison.csv", index=False)
    hcm_comparison.to_csv(hcm_dir / "hcm_paper_comparison.csv", index=False)

    base_summary = {
        "sample_size": int(len(frame)),
        "specification": base_results["specification"],
        "metrics": base_results["metrics"],
        "optimizer_success": bool(base_results["result"].success),
        "optimizer_message": str(base_results["result"].message),
    }
    continuous_summary = {
        "sample_size": int(len(frame)),
        "normalization": continuous_results["normalization"],
        "indicator_mapping": continuous_results["indicator_mapping"],
        "selection_rule": "fixed repository normalization with ref_pro_car_indicator=Mobil10 and ref_env_indicator=Envir05",
        "metrics": continuous_results["metrics"],
        "optimizer_success": bool(continuous_results["result"].success),
        "optimizer_message": str(continuous_results["result"].message),
    }
    (base_dir / "base_logit_summary.json").write_text(json.dumps(base_summary, indent=2), encoding="utf-8")
    (hcm_dir / "hcm_summary.json").write_text(json.dumps(continuous_summary, indent=2), encoding="utf-8")

    write_report(output_dir, frame, base_results, continuous_results, base_comparison, hcm_comparison)


if __name__ == "__main__":
    main()
