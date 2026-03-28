from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def resolve_local_torch_dir() -> Path | None:
    for candidate in (
        PROJECT_ROOT / ".python_packages" / "cu118",
        PROJECT_ROOT / ".python_packages" / "cu126",
    ):
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
from scipy.stats import norm, qmc


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXPERIMENT_DIR / "data"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

torch.set_default_dtype(torch.float64)

ALT_SUFFIX = {
    "e_scooter": "ES",
    "bikesharing": "BS",
    "walking": "WALK",
    "private_car": "CAR",
    "carsharing": "CS",
    "ridepooling": "RP",
    "public_transport": "PT",
}

VALUE_PREFIX = {
    "is_available": "AV",
    "travel_time_min": "TIME",
    "access_time_min": "ACCESS",
    "waiting_time_min": "WAIT",
    "egress_time_min": "EGRESS",
    "detour_time_min": "DETOUR",
    "parking_search_time_min": "PARKING",
    "availability_pct": "AVAILABILITY",
    "cost_eur": "COST",
    "range_km": "RANGE",
    "crowding_pct": "CROWDING",
    "transfer_count": "TRANSFER",
    "scheme_free_floating": "FREEFLOAT",
    "scheme_hybrid": "HYBRID",
    "pedelec": "PEDELEC",
}

DRAW_NAMES = ["z_cost", "z_es", "z_bs", "z_walk", "z_car", "z_cs", "z_rp", "z_pt"]


def resolve_output_dir(output_subdir: str | None) -> Path:
    if not output_subdir:
        return DEFAULT_OUTPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_human_targets() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "human_table4_pooled_full.csv")


def parameter_order() -> list[str]:
    return load_human_targets()["parameter_name"].tolist()


def initial_theta_raw(order: list[str], phi_constraint: str) -> np.ndarray:
    human = load_human_targets()
    mapping = dict(zip(human["parameter_name"], human["human_estimate"]))
    theta = []
    for name in order:
        value = float(mapping[name])
        if name == "PHI_POOL" and phi_constraint == "softplus":
            value = inverse_softplus(max(value - 1e-6, 1e-9))
        theta.append(value)
    return np.array(theta, dtype=float)


def inverse_softplus(value: float) -> float:
    if value > 30:
        return value
    return math.log(math.expm1(value))


def transform_params(theta_raw: torch.Tensor, order: list[str], phi_constraint: str) -> dict[str, torch.Tensor]:
    params = {}
    for index, name in enumerate(order):
        value = theta_raw[index]
        if name == "PHI_POOL" and phi_constraint == "softplus":
            value = torch.nn.functional.softplus(value) + 1e-6
        params[name] = value
    return params


def transform_estimates(theta_raw_np: np.ndarray, order: list[str], phi_constraint: str) -> tuple[np.ndarray, np.ndarray]:
    transformed = []
    jac_diag = []
    for value, name in zip(theta_raw_np, order):
        if name == "PHI_POOL" and phi_constraint == "softplus":
            sigma = 1.0 / (1.0 + math.exp(-value))
            transformed.append(math.log1p(math.exp(value)) + 1e-6)
            jac_diag.append(sigma)
        else:
            transformed.append(value)
            jac_diag.append(1.0)
    return np.array(transformed, dtype=float), np.array(jac_diag, dtype=float)


def complete_panel_only(long_frame: pd.DataFrame, tasks_per_respondent: int) -> pd.DataFrame:
    task_rows = long_frame[["respondent_id", "task_id", "is_valid_choice"]].drop_duplicates()
    task_rows = task_rows.loc[task_rows["is_valid_choice"] == 1].copy()
    valid_counts = task_rows.groupby("respondent_id").size()
    valid_ids = valid_counts.loc[valid_counts == tasks_per_respondent].index.tolist()
    return long_frame.loc[long_frame["respondent_id"].isin(valid_ids)].copy()


def build_wide_estimation_frame(long_frame: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "respondent_id",
        "persona_id",
        "subsample",
        "block_id",
        "task_id",
        "task_index_within_subsample",
        "task_in_block",
        "choice_code",
        "chosen_alternative_id",
        "age_years",
        "household_cars",
        "accessible_bikes",
        "pt_pass",
        "maas_subscription",
    ]
    base = long_frame[base_columns].drop_duplicates().set_index(base_columns)

    wide = base.copy()
    for column_name in VALUE_PREFIX:
        pivot = long_frame.pivot_table(
            index=base.index.names,
            columns="alternative_id",
            values=column_name,
            aggfunc="first",
        )
        pivot.columns = [f"{VALUE_PREFIX[column_name]}_{ALT_SUFFIX[str(alt)]}" for alt in pivot.columns]
        wide = wide.join(pivot)

    wide = wide.reset_index().rename(
        columns={
            "respondent_id": "RESPONDENT_ID",
            "persona_id": "PERSONA_ID",
            "subsample": "SUBSAMPLE",
            "block_id": "BLOCK_ID",
            "task_id": "TASK_ID",
            "task_index_within_subsample": "TASK_INDEX_WITHIN_SUBSAMPLE",
            "task_in_block": "TASK_IN_BLOCK",
            "choice_code": "CHOICE",
            "chosen_alternative_id": "CHOSEN_ALTERNATIVE_ID",
            "age_years": "AGE",
            "household_cars": "HHCAR",
            "accessible_bikes": "HHBIKE",
            "pt_pass": "PTPASS",
            "maas_subscription": "MAAS",
        }
    )

    for prefix in VALUE_PREFIX.values():
        for suffix in ALT_SUFFIX.values():
            column_name = f"{prefix}_{suffix}"
            if column_name not in wide.columns:
                wide[column_name] = 0.0

    numeric_columns = [column for column in wide.columns if column not in {"PERSONA_ID", "SUBSAMPLE", "TASK_ID", "CHOSEN_ALTERNATIVE_ID"}]
    for column_name in numeric_columns:
        wide[column_name] = pd.to_numeric(wide[column_name], errors="coerce").fillna(0.0)

    return wide


def generate_sobol_draws(n_respondents: int, n_draws: int, n_dims: int, seed: int) -> np.ndarray:
    n_points = n_respondents * n_draws
    m = math.ceil(math.log2(max(2, n_points)))
    sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
    draws = sampler.random_base2(m=m)[:n_points]
    draws = np.clip(draws, 1e-10, 1 - 1e-10)
    return norm.ppf(draws).reshape(n_respondents, n_draws, n_dims)


def prepare_tensors(device: torch.device, max_respondents: int | None = None) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    config = load_json(DATA_DIR / "experiment_config.json")
    long_frame = pd.read_csv(DEFAULT_OUTPUT_DIR / "pooled_choices_long.csv")
    long_frame = complete_panel_only(long_frame, int(config["tasks_per_respondent"]))
    wide = build_wide_estimation_frame(long_frame)
    wide = wide.sort_values(["RESPONDENT_ID", "TASK_INDEX_WITHIN_SUBSAMPLE"]).reset_index(drop=True)
    if max_respondents is not None:
        keep_ids = sorted(wide["RESPONDENT_ID"].unique().tolist())[:max_respondents]
        wide = wide.loc[wide["RESPONDENT_ID"].isin(keep_ids)].copy()
        wide = wide.sort_values(["RESPONDENT_ID", "TASK_INDEX_WITHIN_SUBSAMPLE"]).reset_index(drop=True)

    n_respondents = int(wide["RESPONDENT_ID"].nunique())
    tasks = int(config["tasks_per_respondent"])

    def task_tensor(column: str) -> torch.Tensor:
        array = wide[column].to_numpy(dtype=float).reshape(n_respondents, tasks)
        return torch.tensor(array, device=device)

    respondent_frame = wide.groupby("RESPONDENT_ID").first().reset_index()
    tensors = {
        "choice": torch.tensor(wide["CHOICE"].to_numpy(dtype=int).reshape(n_respondents, tasks) - 1, dtype=torch.long, device=device),
        "age": torch.tensor(respondent_frame["AGE"].to_numpy(dtype=float), device=device),
        "hhcar": torch.tensor(respondent_frame["HHCAR"].to_numpy(dtype=float), device=device),
        "hhbike": torch.tensor(respondent_frame["HHBIKE"].to_numpy(dtype=float), device=device),
        "ptpass": torch.tensor(respondent_frame["PTPASS"].to_numpy(dtype=float), device=device),
        "maas": torch.tensor(respondent_frame["MAAS"].to_numpy(dtype=float), device=device),
    }

    for prefix in VALUE_PREFIX.values():
        for suffix in ALT_SUFFIX.values():
            column = f"{prefix}_{suffix}"
            tensors[column.lower()] = task_tensor(column)

    return tensors, wide


def negative_loglik_by_respondent(
    theta_raw: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    draws: torch.Tensor,
    order: list[str],
    phi_constraint: str,
) -> torch.Tensor:
    params = transform_params(theta_raw, order, phi_constraint)

    age = tensors["age"][:, None, None]
    hhcar = tensors["hhcar"][:, None, None]
    hhbike = tensors["hhbike"][:, None, None]
    ptpass = tensors["ptpass"][:, None, None]
    maas = tensors["maas"][:, None, None]

    z_cost = draws[:, :, 0][:, None, :]
    z_es = draws[:, :, 1][:, None, :]
    z_bs = draws[:, :, 2][:, None, :]
    z_walk = draws[:, :, 3][:, None, :]
    z_car = draws[:, :, 4][:, None, :]
    z_cs = draws[:, :, 5][:, None, :]
    z_rp = draws[:, :, 6][:, None, :]
    z_pt = draws[:, :, 7][:, None, :]

    beta_cost = -torch.exp(params["B_COST_LOGMEAN"] + params["SIGMA_COST"] * z_cost)

    u_es = (
        params["ASC_ES"]
        + params["B_AGE_ES"] * age
        + params["B_BIKEACC_ES"] * hhbike
        + params["B_CARACC_ES"] * hhcar
        + params["B_PTPASS_ES"] * ptpass
        + params["B_MAAS_ES"] * maas
        + params["SIGMA_ES"] * z_es
        + params["B_TIME_ES"] * tensors["time_es"][:, :, None]
        + params["B_ACCESS_SHARED"] * tensors["access_es"][:, :, None]
        + params["B_EGRESS_SHARED"] * tensors["egress_es"][:, :, None]
        + params["B_PARKING"] * tensors["parking_es"][:, :, None]
        + beta_cost * tensors["cost_es"][:, :, None]
        + params["B_AVAILABILITY"] * tensors["availability_es"][:, :, None]
        + params["B_SCHEME_SD_FREE_FLOAT"] * tensors["freefloat_es"][:, :, None]
        + params["B_RANGE"] * tensors["range_es"][:, :, None]
    )
    u_bs = (
        params["ASC_BS"]
        + params["B_AGE_BS"] * age
        + params["B_BIKEACC_BS"] * hhbike
        + params["B_CARACC_BS"] * hhcar
        + params["B_PTPASS_BS"] * ptpass
        + params["B_MAAS_BS"] * maas
        + params["SIGMA_BS"] * z_bs
        + params["B_TIME_BS"] * tensors["time_bs"][:, :, None]
        + params["B_ACCESS_SHARED"] * tensors["access_bs"][:, :, None]
        + params["B_EGRESS_SHARED"] * tensors["egress_bs"][:, :, None]
        + params["B_PARKING"] * tensors["parking_bs"][:, :, None]
        + beta_cost * tensors["cost_bs"][:, :, None]
        + params["B_AVAILABILITY"] * tensors["availability_bs"][:, :, None]
        + params["B_SCHEME_SD_FREE_FLOAT"] * tensors["freefloat_bs"][:, :, None]
        + params["B_PEDELEC"] * tensors["pedelec_bs"][:, :, None]
        + params["B_RANGE"] * tensors["range_bs"][:, :, None]
    )
    u_walk = (
        params["ASC_WALK"]
        + params["B_AGE_WALK"] * age
        + params["B_BIKEACC_WALK"] * hhbike
        + params["B_CARACC_WALK"] * hhcar
        + params["B_PTPASS_WALK"] * ptpass
        + params["B_MAAS_WALK"] * maas
        + params["SIGMA_WALK"] * z_walk
        + params["B_TIME_WALK"] * tensors["time_walk"][:, :, None]
    )
    u_car = (
        params["SIGMA_CAR"] * z_car
        + params["PHI_POOL"]
        * (
            params["B_TIME_CAR"] * tensors["time_car"][:, :, None]
            + params["B_ACCESS_OWNED"] * tensors["access_car"][:, :, None]
            + params["B_EGRESS_OWNED"] * tensors["egress_car"][:, :, None]
            + params["B_PARKING"] * tensors["parking_car"][:, :, None]
            + beta_cost * tensors["cost_car"][:, :, None]
        )
    )
    u_cs = (
        params["ASC_CS"]
        + params["B_AGE_CS"] * age
        + params["B_BIKEACC_CS"] * hhbike
        + params["B_CARACC_CS"] * hhcar
        + params["B_PTPASS_CS"] * ptpass
        + params["B_MAAS_CS"] * maas
        + params["SIGMA_CS"] * z_cs
        + params["B_TIME_CS"] * tensors["time_cs"][:, :, None]
        + params["B_ACCESS_SHARED"] * tensors["access_cs"][:, :, None]
        + params["B_EGRESS_SHARED"] * tensors["egress_cs"][:, :, None]
        + params["B_PARKING"] * tensors["parking_cs"][:, :, None]
        + beta_cost * tensors["cost_cs"][:, :, None]
        + params["B_SCHEME_LD_FREE_FLOAT"] * tensors["freefloat_cs"][:, :, None]
        + params["B_SCHEME_LD_HYBRID"] * tensors["hybrid_cs"][:, :, None]
    )
    u_rp = (
        params["ASC_RP"]
        + params["B_AGE_RP"] * age
        + params["B_BIKEACC_RP"] * hhbike
        + params["B_CARACC_RP"] * hhcar
        + params["B_PTPASS_RP"] * ptpass
        + params["B_MAAS_RP"] * maas
        + params["SIGMA_RP"] * z_rp
        + params["B_TIME_RP"] * tensors["time_rp"][:, :, None]
        + params["B_DETOUR_RP"] * tensors["detour_rp"][:, :, None]
        + params["B_ACCESS_SHARED"] * (tensors["access_rp"][:, :, None] + tensors["wait_rp"][:, :, None])
        + params["B_EGRESS_SHARED"] * tensors["egress_rp"][:, :, None]
        + beta_cost * tensors["cost_rp"][:, :, None]
        + params["B_CROWDING"] * tensors["crowding_rp"][:, :, None]
    )
    u_pt = (
        params["ASC_PT"]
        + params["B_AGE_PT"] * age
        + params["B_BIKEACC_PT"] * hhbike
        + params["B_CARACC_PT"] * hhcar
        + params["B_PTPASS_PT"] * ptpass
        + params["B_MAAS_PT"] * maas
        + params["SIGMA_PT"] * z_pt
        + params["B_TIME_PT"] * tensors["time_pt"][:, :, None]
        + params["B_ACCESS_SHARED"] * (tensors["access_pt"][:, :, None] + tensors["wait_pt"][:, :, None])
        + params["B_EGRESS_OWNED"] * tensors["egress_pt"][:, :, None]
        + beta_cost * tensors["cost_pt"][:, :, None]
        + params["B_CROWDING"] * tensors["crowding_pt"][:, :, None]
        + params["B_TRANSFER_PT"] * tensors["transfer_pt"][:, :, None]
    )

    utilities = torch.stack([u_es, u_bs, u_walk, u_car, u_cs, u_rp, u_pt], dim=2)
    availability = torch.stack(
        [
            tensors["av_es"],
            tensors["av_bs"],
            tensors["av_walk"],
            tensors["av_car"],
            tensors["av_cs"],
            tensors["av_rp"],
            tensors["av_pt"],
        ],
        dim=2,
    )[:, :, :, None]
    masked_utilities = torch.where(availability > 0, utilities, torch.full_like(utilities, -1.0e20))
    log_denom = torch.logsumexp(masked_utilities, dim=2)
    choice_index = tensors["choice"][:, :, None, None].expand(-1, -1, 1, draws.shape[1])
    chosen_utility = torch.gather(masked_utilities, 2, choice_index).squeeze(2)
    panel_log_prob_by_draw = (chosen_utility - log_denom).sum(dim=1)
    respondent_log_prob = torch.logsumexp(panel_log_prob_by_draw, dim=1) - math.log(draws.shape[1])
    return -respondent_log_prob


def total_negative_loglik(
    theta_raw: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    draws: torch.Tensor,
    order: list[str],
    phi_constraint: str,
) -> torch.Tensor:
    return negative_loglik_by_respondent(theta_raw, tensors, draws, order, phi_constraint).sum()


def optimize_torch_lbfgs(
    theta0: np.ndarray,
    tensors: dict[str, torch.Tensor],
    draws: torch.Tensor,
    order: list[str],
    device: torch.device,
    phi_constraint: str,
) -> tuple[np.ndarray, float, dict]:
    theta = torch.tensor(theta0, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [theta],
        max_iter=250,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )
    trace = {"closure_calls": 0, "loss_history": []}
    last_loss = None

    def closure():
        nonlocal last_loss
        optimizer.zero_grad()
        loss = total_negative_loglik(theta, tensors, draws, order, phi_constraint)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Torch objective became NaN/Inf during optimization.")
        loss.backward()
        last_loss = float(loss.detach().cpu().item())
        trace["closure_calls"] += 1
        trace["loss_history"].append(last_loss)
        return loss

    optimizer.step(closure)
    return theta.detach().cpu().numpy(), float(last_loss), trace


def scipy_bounds(order: list[str], phi_constraint: str) -> list[tuple[float | None, float | None]]:
    bounds: list[tuple[float | None, float | None]] = []
    for name in order:
        if name == "PHI_POOL" and phi_constraint == "bound_only":
            bounds.append((1e-6, None))
        else:
            bounds.append((None, None))
    return bounds


def optimize_scipy_lbfgsb(
    theta0: np.ndarray,
    tensors: dict[str, torch.Tensor],
    draws: torch.Tensor,
    order: list[str],
    device: torch.device,
    phi_constraint: str,
) -> tuple[np.ndarray, float, dict]:
    trace = {"loss_history": [], "n_function_evals": 0}
    cache: dict[str, object] = {"x": None, "loss": None, "grad": None}

    def objective_and_grad(theta_np: np.ndarray) -> tuple[float, np.ndarray]:
        cached_x = cache["x"]
        if cached_x is not None and np.array_equal(theta_np, cached_x):
            return float(cache["loss"]), np.array(cache["grad"], dtype=float)
        theta = torch.tensor(theta_np, device=device, requires_grad=True)
        loss = total_negative_loglik(theta, tensors, draws, order, phi_constraint)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Torch objective became NaN/Inf during scipy optimization.")
        loss.backward()
        loss_value = float(loss.detach().cpu().item())
        grad_value = theta.grad.detach().cpu().numpy().astype(float)
        trace["n_function_evals"] += 1
        trace["loss_history"].append(loss_value)
        cache["x"] = np.array(theta_np, copy=True)
        cache["loss"] = loss_value
        cache["grad"] = np.array(grad_value, copy=True)
        return loss_value, grad_value

    result = minimize(
        fun=lambda x: objective_and_grad(x)[0],
        x0=theta0.astype(float),
        jac=lambda x: objective_and_grad(x)[1],
        method="L-BFGS-B",
        bounds=scipy_bounds(order, phi_constraint),
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7, "maxcor": 100},
    )
    trace["success"] = bool(result.success)
    trace["message"] = str(result.message)
    trace["nit"] = int(result.nit)
    return result.x.astype(float), float(result.fun), trace


def compute_covariances(
    theta_raw_np: np.ndarray,
    tensors: dict[str, torch.Tensor],
    draws: torch.Tensor,
    order: list[str],
    device: torch.device,
    phi_constraint: str,
) -> tuple[np.ndarray, np.ndarray]:
    theta_raw = torch.tensor(theta_raw_np, device=device, requires_grad=True)

    def objective(vector: torch.Tensor) -> torch.Tensor:
        return total_negative_loglik(vector, tensors, draws, order, phi_constraint)

    hessian = torch.autograd.functional.hessian(objective, theta_raw, vectorize=True).detach().cpu().numpy()
    cov_raw = np.linalg.pinv(hessian)

    n_resp = int(tensors["choice"].shape[0])
    opg = np.zeros_like(cov_raw)
    chunk_size = 64
    for start in range(0, n_resp, chunk_size):
        end = min(start + chunk_size, n_resp)

        def chunk_vector_fn(vector: torch.Tensor) -> torch.Tensor:
            return negative_loglik_by_respondent(vector, tensors, draws, order, phi_constraint)[start:end]

        jac = torch.autograd.functional.jacobian(chunk_vector_fn, theta_raw, vectorize=True).detach().cpu().numpy()
        if jac.ndim == 2 and jac.shape[0] == len(order):
            jac = jac.T
        opg += jac.T @ jac

    robust_cov_raw = cov_raw @ opg @ cov_raw
    return cov_raw, robust_cov_raw


def summarize_estimates(
    theta_raw_np: np.ndarray,
    cov_raw: np.ndarray,
    robust_cov_raw: np.ndarray,
    order: list[str],
    phi_constraint: str,
) -> pd.DataFrame:
    estimates, jac_diag = transform_estimates(theta_raw_np, order, phi_constraint)
    jac = np.diag(jac_diag)
    cov = jac @ cov_raw @ jac
    robust_cov = jac @ robust_cov_raw @ jac

    std_error = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    robust_std_error = np.sqrt(np.clip(np.diag(robust_cov), a_min=0.0, a_max=None))
    z_value = np.divide(estimates, std_error, out=np.full_like(estimates, np.nan), where=std_error > 0)
    robust_z = np.divide(estimates, robust_std_error, out=np.full_like(estimates, np.nan), where=robust_std_error > 0)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(z_value)))
    robust_p = 2.0 * (1.0 - norm.cdf(np.abs(robust_z)))

    frame = pd.DataFrame(
        {
            "parameter_name": order,
            "estimate": estimates,
            "std_error": std_error,
            "z_value": z_value,
            "p_value": p_value,
            "robust_std_error": robust_std_error,
            "robust_z_value": robust_z,
            "robust_p_value": robust_p,
            "significant_5pct": (robust_p <= 0.05).astype(int),
        }
    )
    return frame


def compute_null_loglik(tensors: dict[str, torch.Tensor]) -> float:
    availability = torch.stack(
        [
            tensors["av_es"],
            tensors["av_bs"],
            tensors["av_walk"],
            tensors["av_car"],
            tensors["av_cs"],
            tensors["av_rp"],
            tensors["av_pt"],
        ],
        dim=2,
    )
    available_counts = availability.sum(dim=2).detach().cpu().numpy()
    return -float(np.log(available_counts).sum())


def compare_frames(left: pd.DataFrame, right: pd.DataFrame, right_label: str) -> tuple[pd.DataFrame, dict]:
    comparison = left.merge(right, on="parameter_name", how="left", suffixes=("_torch", f"_{right_label}"))
    estimate_right = f"estimate_{right_label}"
    comparison["difference_torch_minus_other"] = comparison["estimate_torch"] - comparison[estimate_right]
    comparison["sign_match"] = (
        (comparison["estimate_torch"] > 0) == (comparison[estimate_right] > 0)
    ).astype(int)
    summary = {
        "n_compared_parameters": int(comparison["estimate_torch"].notna().sum()),
        "n_sign_matches": int(comparison["sign_match"].sum()),
        "sign_match_rate": float(comparison["sign_match"].mean()),
    }
    return comparison, summary


def write_summary(
    output_dir: Path,
    model_summary: dict,
    choices: pd.DataFrame,
    torch_vs_biogeme_summary: dict,
    human_summary: dict,
) -> None:
    choice_shares = choices["chosen_alternative_name"].value_counts().to_dict()
    lines = [
        "# Torch 实验结果摘要",
        "",
        "## 运行摘要",
        "",
        f"- synthetic respondents：`{model_summary['n_respondents']}`",
        f"- choice tasks per respondent：`6`",
        f"- 总 choices：`{model_summary['n_observations']}`",
        f"- valid choice rate：`{model_summary['valid_choice_rate']:.4f}`",
        f"- draws：`{model_summary['n_draws']}`",
        f"- device：`{model_summary['device']}`",
        f"- runtime_seconds：`{model_summary['runtime_seconds']:.2f}`",
        "",
        "## 选择分布",
        "",
    ]
    for name, count in choice_shares.items():
        lines.append(f"- {name}：`{count}`")
    lines.extend(
        [
            "",
            "## Torch mixed logit 拟合",
            "",
            f"- final loglikelihood：`{model_summary['final_loglikelihood']:.3f}`",
            f"- init loglikelihood：`{model_summary['init_loglikelihood']:.3f}`",
            f"- null loglikelihood：`{model_summary['null_loglikelihood']:.3f}`",
            f"- rho_square_vs_init：`{model_summary['rho_square']:.4f}`",
            f"- rho_square_vs_null：`{model_summary['rho_square_null']:.4f}`",
            "",
            "## Torch vs Biogeme(32 draws)",
            "",
            f"- 可比较参数数：`{torch_vs_biogeme_summary['n_compared_parameters']}`",
            f"- 符号一致数：`{torch_vs_biogeme_summary['n_sign_matches']}`",
            f"- sign match rate：`{torch_vs_biogeme_summary['sign_match_rate']:.4f}`",
            "",
            "## Torch vs Human",
            "",
            f"- 可比较参数数：`{human_summary['n_compared_parameters']}`",
            f"- 符号一致数：`{human_summary['n_sign_matches']}`",
            f"- sign match rate：`{human_summary['sign_match_rate']:.4f}`",
        ]
    )
    (output_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-draws", type=int, default=32)
    parser.add_argument("--output-subdir", type=str, default="torch_32_draws_full")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-respondents", type=int, default=None)
    parser.add_argument("--optimizer-backend", choices=["torch_lbfgs", "scipy_lbfgsb"], default="torch_lbfgs")
    parser.add_argument("--phi-constraint", choices=["softplus", "bound_only"], default="softplus")
    args = parser.parse_args()

    if LOCAL_TORCH_DIR is None:
        raise ModuleNotFoundError("Local torch package was not found under .python_packages/cu118 or .python_packages/cu126.")

    requested_device = args.device
    if requested_device is None:
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    output_dir = resolve_output_dir(args.output_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    config = load_json(DATA_DIR / "experiment_config.json")
    order = parameter_order()
    theta0 = initial_theta_raw(order, phi_constraint=args.phi_constraint)
    tensors, wide = prepare_tensors(device, max_respondents=args.max_respondents)
    n_respondents = int(wide["RESPONDENT_ID"].nunique())
    draws_np = generate_sobol_draws(
        n_respondents=n_respondents,
        n_draws=int(args.n_draws),
        n_dims=len(DRAW_NAMES),
        seed=int(config["estimation_seed"]),
    )
    draws = torch.tensor(draws_np, device=device)

    init_theta = torch.tensor(theta0, device=device)
    init_negloglik = float(total_negative_loglik(init_theta, tensors, draws, order, args.phi_constraint).detach().cpu().item())
    if args.optimizer_backend == "torch_lbfgs":
        theta_hat_raw, final_negloglik, trace = optimize_torch_lbfgs(theta0, tensors, draws, order, device, args.phi_constraint)
    else:
        theta_hat_raw, final_negloglik, trace = optimize_scipy_lbfgsb(theta0, tensors, draws, order, device, args.phi_constraint)
    cov_raw, robust_cov_raw = compute_covariances(theta_hat_raw, tensors, draws, order, device, args.phi_constraint)
    estimates = summarize_estimates(theta_hat_raw, cov_raw, robust_cov_raw, order, args.phi_constraint)

    estimates.to_csv(output_dir / "torch_mixed_estimates.csv", index=False)
    wide.to_csv(output_dir / "torch_estimation_wide.csv", index=False)
    (output_dir / "optimization_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

    null_loglik = compute_null_loglik(tensors)
    model_summary = {
        "experiment_name": EXPERIMENT_DIR.name,
        "model_name": "krauss_public_materials_pooled_7alts_panel_mixed_logit_torch",
        "estimation_backend": "torch",
        "device": str(device),
        "n_draws": int(args.n_draws),
        "n_respondents": n_respondents,
        "n_observations": int(wide.shape[0]),
        "init_loglikelihood": -init_negloglik,
        "final_loglikelihood": -final_negloglik,
        "null_loglikelihood": null_loglik,
        "rho_square": 1.0 - ((-final_negloglik) / (-init_negloglik)),
        "rho_square_null": 1.0 - ((-final_negloglik) / null_loglik),
        "valid_choice_rate": 1.0,
        "runtime_seconds": time.perf_counter() - t0,
        "optimizer": args.optimizer_backend,
        "phi_constraint": args.phi_constraint,
        "local_torch_dir": str(LOCAL_TORCH_DIR),
    }
    (output_dir / "torch_mixed_model_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")

    human = load_human_targets()[["parameter_name", "human_estimate"]]
    torch_human = estimates[["parameter_name", "estimate"]].merge(human, on="parameter_name", how="left")
    torch_human["difference_ai_minus_human"] = torch_human["estimate"] - torch_human["human_estimate"]
    torch_human["sign_match"] = ((torch_human["estimate"] > 0) == (torch_human["human_estimate"] > 0)).astype(int)
    torch_human.to_csv(output_dir / "ai_vs_human_comparison.csv", index=False)
    human_summary = {
        "n_compared_parameters": int(len(torch_human)),
        "n_sign_matches": int(torch_human["sign_match"].sum()),
        "sign_match_rate": float(torch_human["sign_match"].mean()),
    }
    (output_dir / "ai_vs_human_summary.json").write_text(json.dumps(human_summary, indent=2), encoding="utf-8")

    if args.max_respondents is None:
        biogeme_estimates = pd.read_csv(DEFAULT_OUTPUT_DIR / "biogeme_32_draws_full" / "biogeme_mixed_estimates.csv")
        torch_for_compare = estimates.rename(columns={"estimate": "estimate_torch"})[["parameter_name", "estimate_torch"]]
        biogeme_for_compare = biogeme_estimates.rename(columns={"estimate": "estimate_biogeme"})[["parameter_name", "estimate_biogeme"]]
        estimator_comparison, estimator_summary = compare_frames(torch_for_compare, biogeme_for_compare, "biogeme")
        estimator_comparison.to_csv(output_dir / "torch_vs_biogeme_32_comparison.csv", index=False)
        (output_dir / "torch_vs_biogeme_32_summary.json").write_text(json.dumps(estimator_summary, indent=2), encoding="utf-8")
    else:
        estimator_summary = {"n_compared_parameters": 0, "n_sign_matches": 0, "sign_match_rate": 0.0}

    choices = pd.read_csv(DEFAULT_OUTPUT_DIR / "parsed_choices.csv")
    write_summary(output_dir, model_summary, choices, estimator_summary, human_summary)
    print(
        f"[torch] respondents={n_respondents} observations={wide.shape[0]} draws={args.n_draws} "
        f"device={device} loglik={-final_negloglik:.3f} runtime={model_summary['runtime_seconds']:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
