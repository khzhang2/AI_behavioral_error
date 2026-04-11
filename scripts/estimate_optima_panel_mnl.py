from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from optima_common import CONFIG, DATA_DIR, EXPERIMENT_DIR, SOURCE_DATA_DIR, archive_experiment_config, ai_collection_dir_for, ensure_dir, write_json


PARAMETER_NAMES = ["ASC_PT", "ASC_CAR", "B_COST", "B_TIME_PT", "B_TIME_CAR", "B_WAIT", "B_DIST"]
ALT_CODE = {"PT": 0, "CAR": 1, "SLOW_MODES": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["human", "ai_pooled"], required=True)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--max-respondents", type=int, default=None)
    return parser.parse_args()


def human_long() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_DATA_DIR / "human_cleaned_wide.csv").copy().sort_values("respondent_id").reset_index(drop=True)
    rows = []
    for _, row in frame.iterrows():
        rows.extend(
            [
                {
                    "dataset": "human",
                    "model_key": "human",
                    "respondent_id": row["respondent_id"],
                    "task_index": 1,
                    "normalized_weight": float(row["normalized_weight"]),
                    "alternative_name": "PT",
                    "alternative_code": 0,
                    "chosen": int(int(row["Choice"]) == 0),
                    "alt_available": 1,
                    "alt_time": float(row["TimePT"]),
                    "alt_waiting": float(row["WaitingTimePT"]),
                    "alt_cost": float(row["MarginalCostPT"]),
                    "alt_distance": 0.0,
                },
                {
                    "dataset": "human",
                    "model_key": "human",
                    "respondent_id": row["respondent_id"],
                    "task_index": 1,
                    "normalized_weight": float(row["normalized_weight"]),
                    "alternative_name": "CAR",
                    "alternative_code": 1,
                    "chosen": int(int(row["Choice"]) == 1),
                    "alt_available": int(row["CAR_AVAILABLE"]),
                    "alt_time": float(row["TimeCar"]),
                    "alt_waiting": 0.0,
                    "alt_cost": float(row["CostCarCHF"]),
                    "alt_distance": 0.0,
                },
                {
                    "dataset": "human",
                    "model_key": "human",
                    "respondent_id": row["respondent_id"],
                    "task_index": 1,
                    "normalized_weight": float(row["normalized_weight"]),
                    "alternative_name": "SLOW_MODES",
                    "alternative_code": 2,
                    "chosen": int(int(row["Choice"]) == 2),
                    "alt_available": 1,
                    "alt_time": 0.0,
                    "alt_waiting": 0.0,
                    "alt_cost": 0.0,
                    "alt_distance": float(row["distance_km"]),
                },
            ]
        )
    return pd.DataFrame(rows)


def pooled_ai_long() -> pd.DataFrame:
    frames = []
    for model_config in CONFIG["llm_models"]:
        path = ai_collection_dir_for(model_config["key"]) / "ai_panel_long.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.loc[pooled["is_valid_task_response"] == 1].copy()
    pooled["dataset"] = "ai"
    return pooled


def load_long_dataset(dataset: str, max_respondents: int | None) -> pd.DataFrame:
    frame = human_long() if dataset == "human" else pooled_ai_long()
    if frame.empty:
        return frame
    respondent_ids = frame["respondent_id"].drop_duplicates().tolist()
    if max_respondents is not None:
        respondent_ids = respondent_ids[: int(max_respondents)]
        frame = frame.loc[frame["respondent_id"].isin(respondent_ids)].copy()
    return frame.sort_values(["respondent_id", "task_index", "alternative_code"]).reset_index(drop=True)


def utility_matrix(frame: pd.DataFrame, theta: np.ndarray) -> np.ndarray:
    asc_pt, asc_car, b_cost, b_time_pt, b_time_car, b_wait, b_dist = theta
    utility = np.zeros(len(frame), dtype=float)
    is_pt = frame["alternative_name"].to_numpy() == "PT"
    is_car = frame["alternative_name"].to_numpy() == "CAR"
    is_slow = frame["alternative_name"].to_numpy() == "SLOW_MODES"
    utility[is_pt] = asc_pt + b_cost * frame.loc[is_pt, "alt_cost"] + b_time_pt * frame.loc[is_pt, "alt_time"] + b_wait * frame.loc[is_pt, "alt_waiting"]
    utility[is_car] = asc_car + b_cost * frame.loc[is_car, "alt_cost"] + b_time_car * frame.loc[is_car, "alt_time"]
    utility[is_slow] = b_dist * frame.loc[is_slow, "alt_distance"]
    utility[frame["alt_available"].to_numpy(dtype=int) == 0] = -1.0e10
    return utility


def negative_loglikelihood(theta: np.ndarray, frame: pd.DataFrame) -> float:
    work = frame.copy()
    work["utility"] = utility_matrix(work, theta)
    group_cols = ["respondent_id", "task_index"]
    max_u = work.groupby(group_cols)["utility"].transform("max")
    work["exp_u"] = np.exp(work["utility"] - max_u)
    denom = work.groupby(group_cols)["exp_u"].transform("sum")
    work["prob"] = work["exp_u"] / np.clip(denom, 1e-300, None)
    chosen = work.loc[work["chosen"] == 1].copy()
    chosen["logprob"] = np.log(np.clip(chosen["prob"], 1e-300, None))
    respondent_ll = chosen.groupby("respondent_id").agg(
        normalized_weight=("normalized_weight", "first"),
        logprob_sum=("logprob", "sum"),
    )
    return float(-(respondent_ll["normalized_weight"] * respondent_ll["logprob_sum"]).sum())


def estimate(frame: pd.DataFrame) -> tuple[np.ndarray, float, object]:
    theta0 = np.array([0.0, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1], dtype=float)
    result = minimize(
        fun=lambda x: negative_loglikelihood(np.array(x, dtype=float), frame),
        x0=theta0,
        method="BFGS",
        options={"maxiter": int(CONFIG["panel_mnl"]["maxiter"]), "gtol": 1e-6},
    )
    return np.array(result.x, dtype=float), float(result.fun), result


def standard_errors(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance = None
    if hasattr(result, "hess_inv") and result.hess_inv is not None:
        if hasattr(result.hess_inv, "todense"):
            covariance = np.asarray(result.hess_inv.todense(), dtype=float)
        else:
            covariance = np.asarray(result.hess_inv, dtype=float)
    if covariance is None or covariance.shape[0] != len(PARAMETER_NAMES):
        covariance = np.full((len(PARAMETER_NAMES), len(PARAMETER_NAMES)), np.nan)
    std_error = np.sqrt(np.where(np.diag(covariance) > 0, np.diag(covariance), np.nan))
    z_value = result.x / std_error
    p_value = 2.0 * norm.sf(np.abs(z_value))
    return std_error, z_value, p_value


def null_loglikelihood(frame: pd.DataFrame) -> float:
    chosen = frame.loc[frame["chosen"] == 1].copy()
    availability = frame.groupby(["respondent_id", "task_index"])["alt_available"].sum().reset_index(name="n_avail")
    chosen = chosen.merge(availability, on=["respondent_id", "task_index"], how="left")
    chosen["logprob"] = -np.log(np.clip(chosen["n_avail"], 1.0, None))
    respondent_ll = chosen.groupby("respondent_id").agg(
        normalized_weight=("normalized_weight", "first"),
        logprob_sum=("logprob", "sum"),
    )
    return float((respondent_ll["normalized_weight"] * respondent_ll["logprob_sum"]).sum())


def empirical_choice_share(frame: pd.DataFrame) -> dict[str, float]:
    chosen = frame.loc[frame["chosen"] == 1].copy()
    shares = chosen["alternative_name"].value_counts(normalize=True)
    return {str(key): float(value) for key, value in shares.items()}


def grouped_choice_share(frame: pd.DataFrame, group_cols: list[str]) -> dict[str, dict[str, float]]:
    if any(column not in frame.columns for column in group_cols):
        return {}
    chosen = frame.loc[frame["chosen"] == 1].copy()
    payload: dict[str, dict[str, float]] = {}
    for key, group in chosen.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        group_name = "|".join(str(item) for item in key)
        shares = group["alternative_name"].value_counts(normalize=True)
        payload[group_name] = {str(name): float(value) for name, value in shares.items()}
    return payload


def main() -> None:
    args = parse_args()
    archive_experiment_config(EXPERIMENT_DIR)
    output_dir = ensure_dir(EXPERIMENT_DIR / "outputs" / args.output_subdir)
    frame = load_long_dataset(args.dataset, args.max_respondents)
    if frame.empty:
        raise RuntimeError(f"No observations found for dataset={args.dataset}")

    theta_hat, final_negloglik, result = estimate(frame)
    std_error, z_value, p_value = standard_errors(result)
    estimates = pd.DataFrame(
        {
            "parameter_name": PARAMETER_NAMES,
            "estimate": theta_hat,
            "std_error": std_error,
            "z_value": z_value,
            "p_value": p_value,
        }
    )
    estimates.to_csv(output_dir / "mnl_estimates.csv", index=False)
    frame.to_csv(output_dir / "estimation_input_long.csv", index=False)

    summary = {
        "dataset": args.dataset,
        "n_rows_long": int(len(frame)),
        "n_respondents": int(frame["respondent_id"].nunique()),
        "n_tasks": int(frame.loc[frame["chosen"] == 1, ["respondent_id", "task_index"]].drop_duplicates().shape[0]),
        "final_loglikelihood": float(-final_negloglik),
        "null_loglikelihood": float(null_loglikelihood(frame)),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "choice_share": empirical_choice_share(frame),
    }
    if args.dataset == "ai_pooled":
        summary["choice_share_by_model"] = grouped_choice_share(frame, ["model_key"])
        summary["choice_share_by_prompt_arm"] = grouped_choice_share(frame, ["prompt_arm"])
        summary["choice_share_by_model_prompt_arm"] = grouped_choice_share(frame, ["model_key", "prompt_arm"])
        summary["choice_share_by_task_role"] = grouped_choice_share(frame, ["task_role"])
    write_json(output_dir / "mnl_summary.json", summary)
    print(
        f"[estimate_optima_panel_mnl] dataset={args.dataset} respondents={summary['n_respondents']} "
        f"tasks={summary['n_tasks']} loglik={summary['final_loglikelihood']:.3f}"
    )


if __name__ == "__main__":
    main()
