from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from ai_behavioral_error.io import write_csv


def add_model_features(long_frame: pd.DataFrame) -> pd.DataFrame:
    frame = long_frame.copy()
    frame["availability_share"] = frame["availability_pct"] / 100.0
    frame["is_e_scooter"] = (frame["alternative_id"] == "e_scooter").astype(int)
    frame["is_bikesharing"] = (frame["alternative_id"] == "bikesharing").astype(int)
    frame["is_private_car"] = (frame["alternative_id"] == "private_car").astype(int)
    return frame


def _log_likelihood(beta: np.ndarray, x: np.ndarray, y: np.ndarray, group_slices: list[np.ndarray]) -> float:
    total = 0.0
    for group_index in group_slices:
        utilities = x[group_index] @ beta
        centered = utilities - utilities.max()
        exp_utilities = np.exp(centered)
        probabilities = exp_utilities / exp_utilities.sum()
        chosen_mask = y[group_index] == 1
        total += np.log(probabilities[chosen_mask][0])
    return total


def _negative_log_likelihood(beta: np.ndarray, x: np.ndarray, y: np.ndarray, group_slices: list[np.ndarray]) -> float:
    return -_log_likelihood(beta, x, y, group_slices)


def fit_conditional_logit(long_frame: pd.DataFrame, feature_names: list[str], output_dir: Path) -> pd.DataFrame:
    model_frame = add_model_features(long_frame).sort_values(["choice_set_id", "display_label"]).reset_index(drop=True)
    valid_group_ids = (
        model_frame.groupby("choice_set_id")["choice"]
        .sum()
        .loc[lambda series: series == 1]
        .index
    )
    model_frame = model_frame[model_frame["choice_set_id"].isin(valid_group_ids)].reset_index(drop=True)

    if model_frame.empty:
        coefficient_frame = pd.DataFrame(columns=["feature", "coefficient", "std_error", "z_value", "p_value"])
        write_csv(output_dir / "mnl_coefficients.csv", coefficient_frame)
        (output_dir / "mnl_summary.txt").write_text("No valid choice sets were available for estimation.")
        return coefficient_frame

    x = model_frame[feature_names].to_numpy(dtype=float)
    y = model_frame["choice"].to_numpy(dtype=int)

    group_slices = [
        group.index.to_numpy()
        for _, group in model_frame.groupby("choice_set_id", sort=False)
    ]

    start = np.zeros(len(feature_names), dtype=float)
    result = minimize(
        fun=_negative_log_likelihood,
        x0=start,
        args=(x, y, group_slices),
        method="BFGS",
    )

    covariance = np.asarray(result.hess_inv)
    standard_errors = np.sqrt(np.diag(covariance))
    z_values = result.x / standard_errors
    p_values = 2 * (1 - norm.cdf(np.abs(z_values)))

    coefficient_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": result.x,
            "std_error": standard_errors,
            "z_value": z_values,
            "p_value": p_values,
        }
    )

    write_csv(output_dir / "mnl_coefficients.csv", coefficient_frame)
    (output_dir / "mnl_summary.txt").write_text(
        "\n".join(
            [
                "Conditional logit estimated with custom maximum likelihood.",
                f"success: {result.success}",
                f"status: {result.status}",
                f"message: {result.message}",
                f"log_likelihood: {-result.fun:.6f}",
                f"iterations: {result.nit}",
            ]
        )
    )
    return coefficient_frame
