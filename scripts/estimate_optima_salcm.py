from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from optima_common import CONFIG, DATA_DIR, EXPERIMENT_DIR, archive_experiment_config, ai_collection_dir_for, ensure_dir, write_json


CHOICE_NAMES = ["PT", "CAR", "SLOW_MODES"]
PREF_NAMES = ["ASC_PT", "ASC_CAR", "B_COST", "B_TIME_PT", "B_TIME_CAR", "B_WAIT", "B_DIST"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--max-respondents-per-model", type=int, default=None)
    return parser.parse_args()


def load_ai_data(max_respondents_per_model: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_frames = []
    block_frames = []
    total_tasks = int(CONFIG["survey_design"]["total_tasks"])
    for model_config in CONFIG["llm_models"]:
        base_dir = ai_collection_dir_for(model_config["key"])
        long_path = base_dir / "ai_panel_long.csv"
        block_path = base_dir / "ai_panel_block.csv"
        if not long_path.exists() or not block_path.exists():
            continue
        long_frame = pd.read_csv(long_path)
        block_frame = pd.read_csv(block_path)
        if long_frame.empty or block_frame.empty:
            continue
        block_frame = block_frame.loc[block_frame["n_valid_tasks"] >= total_tasks].copy()
        if max_respondents_per_model is not None:
            keep_ids = block_frame.head(int(max_respondents_per_model))["respondent_id"].tolist()
            block_frame = block_frame.loc[block_frame["respondent_id"].isin(keep_ids)].copy()
        long_frame = long_frame.loc[
            long_frame["respondent_id"].isin(block_frame["respondent_id"]) & (long_frame["is_valid_task_response"] == 1)
        ].copy()
        long_frames.append(long_frame)
        block_frames.append(block_frame)
    if not long_frames or not block_frames:
        return pd.DataFrame(), pd.DataFrame()
    return (
        pd.concat(long_frames, ignore_index=True).sort_values(["respondent_id", "task_index", "alternative_code"]).reset_index(drop=True),
        pd.concat(block_frames, ignore_index=True).sort_values(["respondent_id"]).reset_index(drop=True),
    )


def parameter_names(covariate_names: list[str]) -> list[str]:
    names = []
    for class_index in range(int(CONFIG["salcm"]["n_preference_classes"])):
        for pref_name in PREF_NAMES:
            names.append(f"{pref_name}_C{class_index + 1}")
    names.append("LOG_SCALE_CLASS_2")
    for class_index in range(1, int(CONFIG["salcm"]["n_preference_classes"])):
        names.append(f"CLASS_INTERCEPT_C{class_index + 1}")
        for covariate_name in covariate_names:
            names.append(f"CLASS_{covariate_name}_C{class_index + 1}")
    names.append("SCALE_INTERCEPT_S2")
    for covariate_name in covariate_names:
        names.append(f"SCALE_{covariate_name}_S2")
    return names


def build_matrices(long_frame: pd.DataFrame, block_frame: pd.DataFrame, covariate_names: list[str]) -> dict:
    respondent_ids = block_frame["respondent_id"].tolist()
    task_ids = sorted(long_frame["task_index"].drop_duplicates().tolist())
    n_resp = len(respondent_ids)
    n_tasks = len(task_ids)
    n_alt = 3

    choice = np.full((n_resp, n_tasks), -1, dtype=int)
    availability = np.zeros((n_resp, n_tasks, n_alt), dtype=float)
    cost = np.zeros((n_resp, n_tasks, n_alt), dtype=float)
    time = np.zeros((n_resp, n_tasks, n_alt), dtype=float)
    waiting = np.zeros((n_resp, n_tasks, n_alt), dtype=float)
    distance = np.zeros((n_resp, n_tasks, n_alt), dtype=float)

    respondent_index = {respondent_id: idx for idx, respondent_id in enumerate(respondent_ids)}
    task_index = {task_id: idx for idx, task_id in enumerate(task_ids)}
    for _, row in long_frame.iterrows():
        i = respondent_index[str(row["respondent_id"])]
        t = task_index[int(row["task_index"])]
        j = int(row["alternative_code"])
        availability[i, t, j] = float(row["alt_available"])
        cost[i, t, j] = float(row["alt_cost"])
        time[i, t, j] = float(row["alt_time"])
        waiting[i, t, j] = float(row["alt_waiting"])
        distance[i, t, j] = float(row["alt_distance"])
        if int(row["chosen"]) == 1:
            choice[i, t] = j

    covariates = block_frame[covariate_names].to_numpy(dtype=float)
    weights = block_frame["normalized_weight"].to_numpy(dtype=float)
    diagnostics = block_frame[
        [
            "respondent_id",
            "model_key",
            "prompt_arm",
            "label_flip_rate",
            "order_flip_rate",
            "monotonicity_compliance_rate",
            "dominance_violation_rate",
            "confidence_mean",
            *[column for column in block_frame.columns if str(column).startswith("top_attr_share_")],
        ]
    ].copy()
    return {
        "respondent_ids": respondent_ids,
        "task_ids": task_ids,
        "choice": choice,
        "availability": availability,
        "cost": cost,
        "time": time,
        "waiting": waiting,
        "distance": distance,
        "covariates": covariates,
        "weights": weights,
        "diagnostics": diagnostics,
    }


def unpack(theta: np.ndarray, covariate_names: list[str]) -> dict:
    c = int(CONFIG["salcm"]["n_preference_classes"])
    k = len(covariate_names)
    pointer = 0
    class_params = []
    for _ in range(c):
        class_params.append({name: float(theta[pointer + idx]) for idx, name in enumerate(PREF_NAMES)})
        pointer += len(PREF_NAMES)
    log_scale = float(theta[pointer])
    pointer += 1
    class_gamma = []
    for _ in range(c - 1):
        gamma = np.array(theta[pointer : pointer + k + 1], dtype=float)
        class_gamma.append(gamma)
        pointer += k + 1
    scale_gamma = np.array(theta[pointer : pointer + k + 1], dtype=float)
    return {
        "class_params": class_params,
        "log_scale": log_scale,
        "class_gamma": class_gamma,
        "scale_gamma": scale_gamma,
    }


def utility_for_class(params: dict, matrices: dict) -> np.ndarray:
    utility = np.zeros_like(matrices["cost"], dtype=float)
    utility[:, :, 0] = (
        params["ASC_PT"]
        + params["B_COST"] * matrices["cost"][:, :, 0]
        + params["B_TIME_PT"] * matrices["time"][:, :, 0]
        + params["B_WAIT"] * matrices["waiting"][:, :, 0]
    )
    utility[:, :, 1] = (
        params["ASC_CAR"]
        + params["B_COST"] * matrices["cost"][:, :, 1]
        + params["B_TIME_CAR"] * matrices["time"][:, :, 1]
    )
    utility[:, :, 2] = params["B_DIST"] * matrices["distance"][:, :, 2]
    utility[matrices["availability"] == 0] = -1.0e10
    return utility


def respondent_loglikelihood_matrix(theta: np.ndarray, matrices: dict, covariate_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unpacked = unpack(theta, covariate_names)
    class_params = unpacked["class_params"]
    scale_values = np.array([1.0, np.exp(np.clip(unpacked["log_scale"], -2.0, 2.0))], dtype=float)
    n_resp = len(matrices["respondent_ids"])
    c = len(class_params)
    s = len(scale_values)

    w = np.column_stack([np.ones(n_resp), matrices["covariates"]])
    class_logits = [np.zeros(n_resp)]
    for gamma in unpacked["class_gamma"]:
        class_logits.append(w @ gamma)
    class_prob = softmax(np.column_stack(class_logits), axis=1)

    scale_logits = np.column_stack([np.zeros(n_resp), w @ unpacked["scale_gamma"]])
    scale_prob = softmax(scale_logits, axis=1)

    class_scale_loglik = np.zeros((n_resp, c, s), dtype=float)
    for class_index, params in enumerate(class_params):
        base_utility = utility_for_class(params, matrices)
        for scale_index, scale_value in enumerate(scale_values):
            scaled_utility = scale_value * base_utility
            max_u = scaled_utility.max(axis=2, keepdims=True)
            exp_u = np.exp(scaled_utility - max_u)
            denom = np.clip(exp_u.sum(axis=2), 1e-300, None)
            prob = exp_u / denom[:, :, None]
            chosen_prob = np.take_along_axis(prob, matrices["choice"][:, :, None], axis=2).squeeze(-1)
            class_scale_loglik[:, class_index, scale_index] = np.log(np.clip(chosen_prob, 1e-300, None)).sum(axis=1)
    return class_scale_loglik, class_prob, scale_prob, scale_values


def objective(theta: np.ndarray, matrices: dict, covariate_names: list[str]) -> float:
    class_scale_loglik, class_prob, scale_prob, _ = respondent_loglikelihood_matrix(theta, matrices, covariate_names)
    joint_log = (
        np.log(np.clip(class_prob[:, :, None], 1e-300, None))
        + np.log(np.clip(scale_prob[:, None, :], 1e-300, None))
        + class_scale_loglik
    )
    respondent_loglik = logsumexp(joint_log.reshape(len(matrices["respondent_ids"]), -1), axis=1)
    return float(-(matrices["weights"] * respondent_loglik).sum())


def initial_theta(covariate_names: list[str], mnl_estimates: pd.DataFrame | None) -> np.ndarray:
    base = {name: 0.0 for name in PREF_NAMES}
    if mnl_estimates is not None and not mnl_estimates.empty:
        estimate_map = dict(zip(mnl_estimates["parameter_name"], mnl_estimates["estimate"]))
        base["ASC_PT"] = float(estimate_map.get("ASC_PT", 0.0))
        base["ASC_CAR"] = float(estimate_map.get("ASC_CAR", 0.0))
        base["B_COST"] = float(estimate_map.get("B_COST", -0.1))
        base["B_TIME_PT"] = float(estimate_map.get("B_TIME_PT", estimate_map.get("B_TIME", -0.1)))
        base["B_TIME_CAR"] = float(estimate_map.get("B_TIME_CAR", estimate_map.get("B_TIME", -0.1)))
        base["B_WAIT"] = float(estimate_map.get("B_WAIT", -0.1))
        base["B_DIST"] = float(estimate_map.get("B_DIST", -0.1))

    values = []
    class_scales = [1.0, 0.8, 1.2]
    asc_shifts = [0.0, 0.2, -0.2]
    for class_index in range(int(CONFIG["salcm"]["n_preference_classes"])):
        for pref_name in PREF_NAMES:
            val = base[pref_name]
            if pref_name.startswith("ASC_"):
                val += asc_shifts[class_index]
            else:
                val *= class_scales[class_index]
            values.append(val)
    values.append(np.log(0.8))
    for class_index in range(1, int(CONFIG["salcm"]["n_preference_classes"])):
        values.append(-0.3 * class_index)
        values.extend([0.0] * len(covariate_names))
    values.append(-0.2)
    values.extend([0.0] * len(covariate_names))
    return np.array(values, dtype=float)


def bounds(covariate_names: list[str]) -> list[tuple[float | None, float | None]]:
    total = len(parameter_names(covariate_names))
    bound_list = [(-20.0, 20.0)] * total
    log_scale_index = int(CONFIG["salcm"]["n_preference_classes"]) * len(PREF_NAMES)
    bound_list[log_scale_index] = (-2.0, 2.0)
    return bound_list


def posterior_probabilities(theta: np.ndarray, matrices: dict, covariate_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_scale_loglik, class_prob, scale_prob, scale_values = respondent_loglikelihood_matrix(theta, matrices, covariate_names)
    joint_log = (
        np.log(np.clip(class_prob[:, :, None], 1e-300, None))
        + np.log(np.clip(scale_prob[:, None, :], 1e-300, None))
        + class_scale_loglik
    )
    norm_const = logsumexp(joint_log.reshape(len(matrices["respondent_ids"]), -1), axis=1)
    posterior = np.exp(joint_log - norm_const[:, None, None])
    return posterior, class_prob, scale_values


def human_baseline_estimates() -> dict[str, float]:
    path = EXPERIMENT_DIR / "outputs" / "human_baseline_mnl" / "mnl_estimates.csv"
    if not path.exists():
        return {name: np.nan for name in PREF_NAMES}
    frame = pd.read_csv(path)
    mapping = dict(zip(frame["parameter_name"], frame["estimate"]))
    return {name: float(mapping.get(name, np.nan)) for name in PREF_NAMES}


def human_choice_share() -> dict[str, float]:
    path = EXPERIMENT_DIR / "outputs" / "human_baseline_mnl" / "mnl_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in payload["choice_share"].items()}


def regime_label(row: pd.Series) -> str:
    if row["normalized_coefficient_distance"] == row["normalized_coefficient_distance_min"]:
        return "human_like_tradeoff"
    if row["label_flip_rate"] == row["label_flip_rate_max"]:
        return "label_sensitive"
    if row["scale_value"] < 1.0:
        return "low_consistency"
    return "distorted_tradeoff"


def safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    total = float(np.nansum(weights))
    if total <= 1e-12:
        return np.nan
    return float(np.average(values.fillna(0.0), weights=weights))


def main() -> None:
    args = parse_args()
    archive_experiment_config(EXPERIMENT_DIR)
    output_dir = ensure_dir(EXPERIMENT_DIR / "outputs" / args.output_subdir)
    long_frame, block_frame = load_ai_data(args.max_respondents_per_model)
    if long_frame.empty or block_frame.empty:
        raise RuntimeError("No valid pooled AI panel data available for SALCM estimation.")

    covariate_names = list(CONFIG["salcm"]["membership_covariates"])
    matrices = build_matrices(long_frame, block_frame, covariate_names)
    mnl_path = EXPERIMENT_DIR / "outputs" / "pooled_ai_panel_mnl" / "mnl_estimates.csv"
    mnl_estimates = pd.read_csv(mnl_path) if mnl_path.exists() else None
    theta0 = initial_theta(covariate_names, mnl_estimates)
    result = minimize(
        fun=lambda x: objective(np.array(x, dtype=float), matrices, covariate_names),
        x0=theta0,
        method="L-BFGS-B",
        bounds=bounds(covariate_names),
        options={"maxiter": int(CONFIG["salcm"]["maxiter"]), "maxfun": 100000, "ftol": 1e-10},
    )
    theta_hat = np.array(result.x, dtype=float)
    parameter_frame = pd.DataFrame({"parameter_name": parameter_names(covariate_names), "estimate": theta_hat})
    parameter_frame.to_csv(output_dir / "salcm_estimates.csv", index=False)
    long_frame.to_csv(output_dir / "estimation_input_long.csv", index=False)
    block_frame.to_csv(output_dir / "estimation_input_block.csv", index=False)

    posterior, class_prob, scale_values = posterior_probabilities(theta_hat, matrices, covariate_names)
    posterior_rows = []
    for respondent_index, respondent_id in enumerate(matrices["respondent_ids"]):
        row = {
            "respondent_id": respondent_id,
            "most_likely_state": int(np.argmax(posterior[respondent_index].reshape(-1))),
        }
        state_pointer = 0
        for class_index in range(int(CONFIG["salcm"]["n_preference_classes"])):
            for scale_index in range(int(CONFIG["salcm"]["n_scale_classes"])):
                row[f"posterior_c{class_index + 1}_s{scale_index + 1}"] = float(posterior[respondent_index, class_index, scale_index])
                state_pointer += 1
        posterior_rows.append(row)
    posterior_frame = pd.DataFrame(posterior_rows)
    posterior_frame.to_csv(output_dir / "salcm_posterior_membership.csv", index=False)

    human_estimates = human_baseline_estimates()
    human_share = human_choice_share()
    unpacked = unpack(theta_hat, covariate_names)
    regime_rows = []
    diagnostics = matrices["diagnostics"].copy()
    chosen_rows = long_frame.loc[(long_frame["chosen"] == 1) & (long_frame["is_valid_task_response"] == 1), ["respondent_id", "alternative_name"]].copy()

    for class_index in range(int(CONFIG["salcm"]["n_preference_classes"])):
        for scale_index in range(int(CONFIG["salcm"]["n_scale_classes"])):
            posterior_weight = posterior[:, class_index, scale_index]
            posterior_weight = posterior_weight / np.clip(posterior_weight.sum(), 1e-12, None)
            diag = diagnostics.copy()
            diag["posterior_weight"] = posterior_weight
            weighted_choice = chosen_rows.merge(
                pd.DataFrame({"respondent_id": matrices["respondent_ids"], "posterior_weight": posterior_weight}),
                on="respondent_id",
                how="left",
            )
            choice_share = (
                weighted_choice.groupby("alternative_name")["posterior_weight"].sum() / np.clip(weighted_choice["posterior_weight"].sum(), 1e-12, None)
            ).to_dict()
            class_params = unpacked["class_params"][class_index]
            coeff_distance = float(
                np.sqrt(sum((class_params[name] - human_estimates.get(name, 0.0)) ** 2 for name in PREF_NAMES))
                / np.clip(np.sqrt(sum((human_estimates.get(name, 0.0)) ** 2 for name in PREF_NAMES)), 1e-12, None)
            )
            sign_mismatch = int(sum(np.sign(class_params[name]) != np.sign(human_estimates.get(name, 0.0)) for name in PREF_NAMES))
            mode_share_deviation = float(
                0.5
                * sum(
                    abs(float(choice_share.get(name, 0.0)) - float(human_share.get(name, 0.0)))
                    for name in ["PT", "CAR", "SLOW_MODES"]
                )
            )
            regime_rows.append(
                {
                    "preference_class": class_index + 1,
                    "scale_class": scale_index + 1,
                    "scale_value": float(scale_values[scale_index]),
                    "posterior_mass": float(posterior[:, class_index, scale_index].mean()),
                    "sign_mismatches": sign_mismatch,
                    "normalized_coefficient_distance": coeff_distance,
                    "mode_share_deviation": mode_share_deviation,
                    "label_flip_rate": safe_weighted_mean(diag["label_flip_rate"], diag["posterior_weight"]),
                    "order_flip_rate": safe_weighted_mean(diag["order_flip_rate"], diag["posterior_weight"]),
                    "monotonicity_compliance_rate": safe_weighted_mean(diag["monotonicity_compliance_rate"], diag["posterior_weight"]),
                    "dominance_violation_rate": safe_weighted_mean(diag["dominance_violation_rate"], diag["posterior_weight"]),
                    "confidence_mean": safe_weighted_mean(diag["confidence_mean"], diag["posterior_weight"]),
                    "choice_share_PT": float(choice_share.get("PT", 0.0)),
                    "choice_share_CAR": float(choice_share.get("CAR", 0.0)),
                    "choice_share_SLOW_MODES": float(choice_share.get("SLOW_MODES", 0.0)),
                    **{f"{name}_estimate": float(class_params[name]) for name in PREF_NAMES},
                }
            )
    regime_frame = pd.DataFrame(regime_rows)
    regime_frame["normalized_coefficient_distance_min"] = regime_frame["normalized_coefficient_distance"].min()
    regime_frame["label_flip_rate_max"] = regime_frame["label_flip_rate"].max()
    regime_frame["regime_label"] = regime_frame.apply(regime_label, axis=1)
    regime_frame = regime_frame.drop(columns=["normalized_coefficient_distance_min", "label_flip_rate_max"])
    regime_frame.to_csv(output_dir / "salcm_regime_summaries.csv", index=False)

    block_scores = []
    regime_score = regime_frame.set_index(["preference_class", "scale_class"])[
        ["normalized_coefficient_distance", "mode_share_deviation", "dominance_violation_rate", "label_flip_rate"]
    ].sum(axis=1)
    for respondent_index, respondent_id in enumerate(matrices["respondent_ids"]):
        score = 0.0
        for class_index in range(int(CONFIG["salcm"]["n_preference_classes"])):
            for scale_index in range(int(CONFIG["salcm"]["n_scale_classes"])):
                score += float(posterior[respondent_index, class_index, scale_index]) * float(regime_score.loc[(class_index + 1, scale_index + 1)])
        block_scores.append({"respondent_id": respondent_id, "posterior_distortion_score": score})
    pd.DataFrame(block_scores).to_csv(output_dir / "salcm_block_distortion_scores.csv", index=False)

    summary = {
        "n_respondents": int(len(matrices["respondent_ids"])),
        "n_tasks_per_respondent": int(len(matrices["task_ids"])),
        "n_preference_classes": int(CONFIG["salcm"]["n_preference_classes"]),
        "n_scale_classes": int(CONFIG["salcm"]["n_scale_classes"]),
        "final_loglikelihood": float(-result.fun),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "posterior_probability_min": float(posterior_frame[[column for column in posterior_frame.columns if column.startswith("posterior_")]].sum(axis=1).min()),
        "posterior_probability_max": float(posterior_frame[[column for column in posterior_frame.columns if column.startswith("posterior_")]].sum(axis=1).max()),
        "class_masses": {
            f"C{class_index + 1}_S{scale_index + 1}": float(posterior[:, class_index, scale_index].mean())
            for class_index in range(int(CONFIG["salcm"]["n_preference_classes"]))
            for scale_index in range(int(CONFIG["salcm"]["n_scale_classes"]))
        },
    }
    write_json(output_dir / "salcm_summary.json", summary)
    print(
        f"[estimate_optima_salcm] respondents={summary['n_respondents']} tasks={summary['n_tasks_per_respondent']} "
        f"loglik={summary['final_loglikelihood']:.3f}"
    )


if __name__ == "__main__":
    main()
