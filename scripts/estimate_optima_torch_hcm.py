from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_local_torch_dir() -> Path | None:
    for candidate in (PROJECT_ROOT / ".python_packages" / "cu118",):
        if candidate.exists():
            return candidate
    return None


LOCAL_TORCH_DIR = resolve_local_torch_dir()
if LOCAL_TORCH_DIR is not None:
    sys.path.insert(0, str(LOCAL_TORCH_DIR))

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import norm

from optima_common import AI_COLLECTION_DIR, CONFIG, DATA_DIR, EXPERIMENT_DIR, INDICATOR_NAMES, OUTPUT_DIR, archive_experiment_config, ensure_dir, ensure_pt_non_wait_columns, write_json
from optima_hcm_model_spec import INDICATOR_SPECS, PARAMETER_ORDER, POSITIVE_PARAMETERS

torch.set_default_dtype(torch.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["human", "ai"], required=True)
    parser.add_argument("--n-draws", type=int, required=True)
    parser.add_argument("--output-subdir", type=str, required=True)
    parser.add_argument("--start-values", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


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
    frame = frame.sort_values("respondent_id").reset_index(drop=True)
    if max_rows is not None:
        frame = frame.head(int(max_rows)).copy()
    return frame


def initial_theta() -> np.ndarray:
    values = []
    for name in PARAMETER_ORDER:
        if name in {"SIGMA_CAR", "SIGMA_ENV"}:
            values.append(1.0)
        elif name in {"DELTA_1", "DELTA_2"}:
            values.append(0.8)
        else:
            values.append(0.0)
    return np.array(values, dtype=float)


def bounds() -> list[tuple[float | None, float | None]]:
    result = []
    for name in PARAMETER_ORDER:
        if name in POSITIVE_PARAMETERS:
            result.append((1e-6, 20.0))
        else:
            result.append((-20.0, 20.0))
    return result


def prepare_tensors(frame: pd.DataFrame, draw_path: Path, n_draws: int, device: torch.device) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    draws = np.load(draw_path)[: len(frame), :n_draws, :]
    tensors = {
        "choice": torch.tensor(frame["Choice"].to_numpy(dtype=int), dtype=torch.long, device=device),
        "car_available": torch.tensor(frame["CAR_AVAILABLE"].to_numpy(dtype=float), device=device),
        "normalized_weight": torch.tensor(frame["normalized_weight"].to_numpy(dtype=float), device=device),
        "work_trip": torch.tensor(frame["work_trip"].to_numpy(dtype=float), device=device),
        "other_trip": torch.tensor(frame["other_trip"].to_numpy(dtype=float), device=device),
        "TimePT_scaled": torch.tensor(frame["TimePT_scaled"].to_numpy(dtype=float), device=device),
        "TimeCar_scaled": torch.tensor(frame["TimeCar_scaled"].to_numpy(dtype=float), device=device),
        "WaitingTimePT_scaled": torch.tensor(frame["WaitingTimePT_scaled"].to_numpy(dtype=float), device=device),
        "MarginalCostPT_scaled": torch.tensor(frame["MarginalCostPT_scaled"].to_numpy(dtype=float), device=device),
        "CostCarCHF_scaled": torch.tensor(frame["CostCarCHF_scaled"].to_numpy(dtype=float), device=device),
        "distance_km_scaled": torch.tensor(frame["distance_km_scaled"].to_numpy(dtype=float), device=device),
        "high_education": torch.tensor(frame["high_education"].to_numpy(dtype=float), device=device),
        "top_manager": torch.tensor(frame["top_manager"].to_numpy(dtype=float), device=device),
        "employees": torch.tensor(frame["employees"].to_numpy(dtype=float), device=device),
        "age_30_less": torch.tensor(frame["age_30_less"].to_numpy(dtype=float), device=device),
        "ScaledIncome": torch.tensor(frame["ScaledIncome"].to_numpy(dtype=float), device=device),
        "car_oriented_parents": torch.tensor(frame["car_oriented_parents"].to_numpy(dtype=float), device=device),
        "childSuburb": torch.tensor(frame["childSuburb"].to_numpy(dtype=float), device=device),
        "city_center_as_kid": torch.tensor(frame["city_center_as_kid"].to_numpy(dtype=float), device=device),
        "artisans": torch.tensor(frame["artisans"].to_numpy(dtype=float), device=device),
        "low_education": torch.tensor(frame["low_education"].to_numpy(dtype=float), device=device),
    }
    for indicator_name in INDICATOR_NAMES:
        tensors[indicator_name] = torch.tensor(frame[indicator_name].to_numpy(dtype=int), dtype=torch.long, device=device)
    return tensors, draws


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def probability_ordered_probit(observed: torch.Tensor, latent_index: torch.Tensor, delta_1: torch.Tensor, delta_2: torch.Tensor) -> torch.Tensor:
    tau_1 = delta_1
    tau_2 = delta_1 + delta_2
    thresholds = [-tau_2, -tau_1, torch.tensor(0.0, device=latent_index.device), tau_1, tau_2]
    observed_zero = observed[:, None] - 1
    probs = torch.zeros_like(latent_index)
    for category in range(6):
        upper = torch.ones_like(latent_index) if category == 5 else normal_cdf(thresholds[category] - latent_index)
        lower = torch.zeros_like(latent_index) if category == 0 else normal_cdf(thresholds[category - 1] - latent_index)
        probs = probs + (observed_zero == category).to(latent_index.dtype) * (upper - lower)
    return torch.clamp(probs, min=1e-30, max=1.0)


def unpack(theta: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: theta[index] for index, name in enumerate(PARAMETER_ORDER)}


def weighted_loglikelihood_contributions(theta: torch.Tensor, tensors: dict[str, torch.Tensor], draws_np: np.ndarray, device: torch.device) -> torch.Tensor:
    params = unpack(theta)
    draws = torch.tensor(draws_np, device=device)
    omega_car = draws[:, :, 0]
    omega_env = draws[:, :, 1]

    lv_car = (
        params["LV_CAR_INTERCEPT"]
        + params["LV_CAR_HIGH_EDU"] * tensors["high_education"][:, None]
        + params["LV_CAR_TOP_MANAGER"] * tensors["top_manager"][:, None]
        + params["LV_CAR_EMPLOYEES"] * tensors["employees"][:, None]
        + params["LV_CAR_AGE_30_LESS"] * tensors["age_30_less"][:, None]
        + params["LV_CAR_SCALED_INCOME"] * tensors["ScaledIncome"][:, None]
        + params["LV_CAR_PARENTS"] * tensors["car_oriented_parents"][:, None]
        + params["SIGMA_CAR"] * omega_car
    )
    lv_env = (
        params["LV_ENV_INTERCEPT"]
        + params["LV_ENV_CHILD_SUBURB"] * tensors["childSuburb"][:, None]
        + params["LV_ENV_SCALED_INCOME"] * tensors["ScaledIncome"][:, None]
        + params["LV_ENV_CITY_CENTER_KID"] * tensors["city_center_as_kid"][:, None]
        + params["LV_ENV_ARTISANS"] * tensors["artisans"][:, None]
        + params["LV_ENV_HIGH_EDU"] * tensors["high_education"][:, None]
        + params["LV_ENV_LOW_EDU"] * tensors["low_education"][:, None]
        + params["SIGMA_ENV"] * omega_env
    )

    utility_pt = (
        params["ASC_PT"]
        + params["B_COST"] * tensors["MarginalCostPT_scaled"][:, None]
        + params["B_TIME_PT"] * tensors["TimePT_scaled"][:, None]
        + params["B_WAIT_WORK"] * tensors["WaitingTimePT_scaled"][:, None] * tensors["work_trip"][:, None]
        + params["B_WAIT_OTHER"] * tensors["WaitingTimePT_scaled"][:, None] * tensors["other_trip"][:, None]
        + params["B_LV_CAR_TO_PT"] * lv_car
        + params["B_LV_ENV_TO_PT"] * lv_env
    )
    utility_car = (
        params["ASC_CAR"]
        + params["B_COST"] * tensors["CostCarCHF_scaled"][:, None]
        + params["B_TIME_CAR"] * tensors["TimeCar_scaled"][:, None]
        + params["B_LV_CAR_TO_CAR"] * lv_car
        + params["B_LV_ENV_TO_CAR"] * lv_env
    )
    utility_slow = (
        params["B_DIST_WORK"] * tensors["distance_km_scaled"][:, None] * tensors["work_trip"][:, None]
        + params["B_DIST_OTHER"] * tensors["distance_km_scaled"][:, None] * tensors["other_trip"][:, None]
        + 0.0 * omega_car
    )
    utilities = torch.stack([utility_pt, utility_car, utility_slow], dim=1)
    availability = torch.stack(
        [
            torch.ones_like(tensors["car_available"]),
            tensors["car_available"],
            torch.ones_like(tensors["car_available"]),
        ],
        dim=1,
    )[:, :, None]
    masked = torch.where(availability > 0, utilities, torch.full_like(utilities, -1.0e20))
    log_denom = torch.logsumexp(masked, dim=1)
    chosen = torch.gather(masked, 1, tensors["choice"][:, None, None].expand(-1, 1, masked.shape[2])).squeeze(1)
    choice_prob = torch.exp(chosen - log_denom)

    indicator_product = torch.ones_like(choice_prob)
    for indicator_name, spec in INDICATOR_SPECS.items():
        intercept = params[spec["intercept"]]
        if spec["latent"] == "car":
            loading = torch.tensor(spec["loading"], device=device) if spec["normalized"] else params[str(spec["loading"])]
            latent_index = intercept + loading * lv_car
        else:
            loading = torch.tensor(spec["loading"], device=device) if spec["normalized"] else params[str(spec["loading"])]
            latent_index = intercept + loading * lv_env
        indicator_product = indicator_product * probability_ordered_probit(
            tensors[indicator_name],
            latent_index,
            params["DELTA_1"],
            params["DELTA_2"],
        )

    conditional = torch.clamp(choice_prob * indicator_product, min=1e-300, max=1.0)
    unconditional = torch.clamp(conditional.mean(dim=1), min=1e-300, max=1.0)
    weighted_loglik = tensors["normalized_weight"] * torch.log(unconditional)
    return weighted_loglik


def total_negative_loglikelihood(theta: torch.Tensor, tensors: dict[str, torch.Tensor], draws_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return -weighted_loglikelihood_contributions(theta, tensors, draws_np, device).sum()


def optimize(theta0: np.ndarray, tensors: dict[str, torch.Tensor], draws_np: np.ndarray, device: torch.device) -> tuple[np.ndarray, float, dict]:
    trace: dict[str, float | int | list] = {"iterations": []}
    cache: dict[str, np.ndarray | float] = {}

    def objective(theta_np: np.ndarray):
        if cache.get("x") is not None and np.array_equal(cache["x"], theta_np):
            return float(cache["loss"]), np.array(cache["grad"], dtype=float)
        theta = torch.tensor(theta_np, device=device, requires_grad=True)
        loss = total_negative_loglikelihood(theta, tensors, draws_np, device)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Torch objective became NaN or Inf.")
        loss.backward()
        grad = theta.grad.detach().cpu().numpy().astype(float)
        loss_value = float(loss.detach().cpu().item())
        cache["x"] = theta_np.copy()
        cache["loss"] = loss_value
        cache["grad"] = grad.copy()
        return loss_value, grad

    def fun(theta_np: np.ndarray) -> float:
        loss_value, grad = objective(theta_np)
        trace["iterations"].append({"loss": loss_value, "grad_norm": float(np.linalg.norm(grad))})
        return loss_value

    def jac(theta_np: np.ndarray) -> np.ndarray:
        return objective(theta_np)[1]

    result = minimize(
        fun=fun,
        x0=theta0,
        jac=jac,
        method="L-BFGS-B",
        bounds=bounds(),
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7, "maxcor": 100},
    )
    trace["success"] = bool(result.success)
    trace["message"] = str(result.message)
    trace["n_iterations"] = int(result.nit)
    return np.array(result.x, dtype=float), float(result.fun), trace


def stable_inverse(matrix: np.ndarray) -> tuple[np.ndarray, str]:
    symmetric = 0.5 * (matrix + matrix.T)
    identity = np.eye(symmetric.shape[0], dtype=float)
    fallback_inverse: np.ndarray | None = None
    fallback_label = "pinv"
    for ridge in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
        adjusted = symmetric + ridge * identity
        try:
            inverse = np.linalg.inv(adjusted)
            inverse = 0.5 * (inverse + inverse.T)
            if fallback_inverse is None:
                fallback_inverse = inverse
                fallback_label = f"inverse_ridge_{ridge:g}"
            if np.all(np.diag(inverse) > 0):
                return inverse, f"inverse_ridge_{ridge:g}"
        except np.linalg.LinAlgError:
            continue
        try:
            inverse = np.linalg.pinv(adjusted, rcond=1e-10)
            inverse = 0.5 * (inverse + inverse.T)
            if fallback_inverse is None:
                fallback_inverse = inverse
                fallback_label = f"pinv_ridge_{ridge:g}"
            if np.all(np.diag(inverse) > 0):
                return inverse, f"pinv_ridge_{ridge:g}"
        except np.linalg.LinAlgError:
            continue
    if fallback_inverse is not None:
        return fallback_inverse, fallback_label
    inverse = np.linalg.pinv(symmetric, rcond=1e-10)
    return 0.5 * (inverse + inverse.T), "pinv"


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def infer_statistics(theta_hat: np.ndarray, frame: pd.DataFrame, draw_path: Path, n_draws: int) -> tuple[dict[str, np.ndarray], dict]:
    stats_device = torch.device("cpu")
    tensors, draws_np = prepare_tensors(frame, draw_path, n_draws, stats_device)
    theta = torch.tensor(theta_hat, device=stats_device, requires_grad=True)

    def objective(parameter_vector: torch.Tensor) -> torch.Tensor:
        return total_negative_loglikelihood(parameter_vector, tensors, draws_np, stats_device)

    hessian = torch.autograd.functional.hessian(objective, theta).detach().cpu().numpy().astype(float)
    covariance, inverse_method = stable_inverse(hessian)
    std_error = np.sqrt(np.where(np.diag(covariance) > 0, np.diag(covariance), np.nan))
    z_value = theta_hat / std_error
    p_value = 2.0 * norm.sf(np.abs(z_value))

    robust_std_error = np.full_like(std_error, np.nan)
    robust_z_value = np.full_like(std_error, np.nan)
    robust_p_value = np.full_like(std_error, np.nan)
    robust_status = "not_computed"

    try:
        def contribution_objective(parameter_vector: torch.Tensor) -> torch.Tensor:
            return -weighted_loglikelihood_contributions(parameter_vector, tensors, draws_np, stats_device)

        try:
            jacobian = torch.autograd.functional.jacobian(contribution_objective, theta, vectorize=True)
        except TypeError:
            jacobian = torch.autograd.functional.jacobian(contribution_objective, theta)
        score_matrix = jacobian.detach().cpu().numpy().astype(float)
        meat = score_matrix.T @ score_matrix
        robust_covariance = covariance @ meat @ covariance
        robust_covariance = 0.5 * (robust_covariance + robust_covariance.T)
        robust_std_error = np.sqrt(np.where(np.diag(robust_covariance) > 0, np.diag(robust_covariance), np.nan))
        robust_z_value = theta_hat / robust_std_error
        robust_p_value = 2.0 * norm.sf(np.abs(robust_z_value))
        robust_status = "computed"
    except Exception as exc:
        robust_status = f"failed: {exc}"

    stats = {
        "std_error": std_error,
        "z_value": z_value,
        "p_value": p_value,
        "robust_std_error": robust_std_error,
        "robust_z_value": robust_z_value,
        "robust_p_value": robust_p_value,
    }
    diagnostics = {
        "stats_device": str(stats_device),
        "inverse_method": inverse_method,
        "robust_status": robust_status,
        "hessian_condition_number": finite_or_none(np.linalg.cond(hessian)),
    }
    return stats, diagnostics


def estimates_frame(theta_hat: np.ndarray, stats: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    payload = {
        "parameter_name": PARAMETER_ORDER,
        "estimate": theta_hat,
    }
    if stats is not None:
        payload.update(stats)
    return pd.DataFrame(
        payload
    )


def evaluate_loglikelihood(frame: pd.DataFrame, theta_values: dict[str, float], draw_path: Path, n_draws: int, device: torch.device) -> float:
    tensors, draws_np = prepare_tensors(frame, draw_path, n_draws, device)
    theta = np.array([float(theta_values[name]) for name in PARAMETER_ORDER], dtype=float)
    with torch.no_grad():
        loss = total_negative_loglikelihood(torch.tensor(theta, device=device), tensors, draws_np, device)
    return float(-loss.detach().cpu().item())


def main() -> None:
    args = parse_args()
    archive_experiment_config(EXPERIMENT_DIR)
    output_dir = ensure_dir(OUTPUT_DIR / args.output_subdir)
    frame = load_dataset(args.dataset, args.max_rows)
    draw_path = DATA_DIR / f"shared_sobol_draws_{int(args.n_draws)}.npy"
    requested_device = args.device or CONFIG["torch"]["default_device"]
    if requested_device == "cuda_if_available":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

    theta0 = initial_theta()
    if args.start_values:
        start_frame = pd.read_csv(args.start_values)
        mapping = dict(zip(start_frame["parameter_name"], start_frame["estimate"]))
        theta0 = np.array([float(mapping.get(name, default)) for name, default in zip(PARAMETER_ORDER, theta0)], dtype=float)

    tensors, draws_np = prepare_tensors(frame, draw_path, int(args.n_draws), device)
    start_time = time.time()
    theta_hat, final_negloglik, trace = optimize(theta0, tensors, draws_np, device)
    runtime_seconds = time.time() - start_time
    stats_start_time = time.time()
    stats, diagnostics = infer_statistics(theta_hat, frame, draw_path, int(args.n_draws))
    statistics_runtime_seconds = time.time() - stats_start_time

    estimates = estimates_frame(theta_hat, stats)
    estimates.to_csv(output_dir / "torch_hcm_estimates.csv", index=False)
    frame.to_csv(output_dir / "estimation_input.csv", index=False)
    write_json(
        output_dir / "torch_hcm_summary.json",
        {
            "dataset": args.dataset,
            "n_rows": int(len(frame)),
            "n_draws": int(args.n_draws),
            "device": str(device),
            "final_loglikelihood": float(-final_negloglik),
            "runtime_seconds": float(runtime_seconds),
            "statistics_runtime_seconds": float(statistics_runtime_seconds),
            "local_torch_dir": str(LOCAL_TORCH_DIR),
            **diagnostics,
        },
    )
    write_json(output_dir / "optimization_trace.json", trace)
    print(f"[torch_hcm] dataset={args.dataset} rows={len(frame)} draws={args.n_draws} device={device} loglik={-final_negloglik:.3f} runtime={runtime_seconds:.2f}s")


if __name__ == "__main__":
    main()
