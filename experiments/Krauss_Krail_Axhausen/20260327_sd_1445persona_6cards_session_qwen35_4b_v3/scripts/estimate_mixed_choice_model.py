from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, qmc


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DATA_DIR = EXPERIMENT_DIR / "data"

torch.set_default_dtype(torch.float64)

PARAMETER_ORDER = [
    "ASC_ES_REL_CAR",
    "ASC_BS_REL_CAR",
    "ASC_WALK_REL_CAR",
    "AGE_ES_REL_CAR",
    "AGE_BS_REL_CAR",
    "AGE_WALK_REL_CAR",
    "BIKEACC_ES_REL_CAR",
    "BIKEACC_BS_REL_CAR",
    "BIKEACC_WALK_REL_CAR",
    "CARACC_ES_REL_CAR",
    "CARACC_BS_REL_CAR",
    "CARACC_WALK_REL_CAR",
    "PTPASS_ES_REL_CAR",
    "PTPASS_BS_REL_CAR",
    "PTPASS_WALK_REL_CAR",
    "MAAS_ES_REL_CAR",
    "MAAS_BS_REL_CAR",
    "MAAS_WALK_REL_CAR",
    "SIGMA_ES",
    "SIGMA_BS",
    "SIGMA_WALK",
    "B_TIME_ES",
    "B_TIME_BS",
    "B_TIME_WALK",
    "B_TIME_CAR",
    "B_ACCESS_SHARED",
    "B_ACCESS_CAR",
    "B_EGRESS_SHARED",
    "B_EGRESS_CAR",
    "B_PARKING",
    "B_COST_LOGMEAN",
    "SIGMA_COST",
    "B_AVAILABILITY",
    "B_SCHEME_SD",
    "B_RANGE",
    "B_PEDELEC",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sobol_normal_draws(n_draws: int, dim: int) -> np.ndarray:
    m = math.ceil(math.log2(n_draws + 1))
    sampler = qmc.Sobol(d=dim, scramble=False)
    uniform_draws = sampler.random_base2(m=m)[1 : n_draws + 1]
    uniform_draws = np.clip(uniform_draws, 1e-10, 1 - 1e-10)
    return norm.ppf(uniform_draws)


def initial_theta() -> np.ndarray:
    human = pd.read_csv(DATA_DIR / "human_table4_sd_subset.csv")
    mapping = dict(zip(human["parameter_name"], human["human_estimate"]))
    return np.array([mapping[name] for name in PARAMETER_ORDER], dtype=float)


def prepare_tensors() -> tuple[dict[str, torch.Tensor], list[int]]:
    data = pd.read_csv(OUTPUT_DIR / "ai_choices_wide.csv")
    data = data.loc[data["choice"] > 0].copy()
    data = data.sort_values(["respondent_id", "task_index"]).reset_index(drop=True)

    respondent_ids = sorted(data["respondent_id"].unique().tolist())
    tasks_per_respondent = data.groupby("respondent_id").size().iloc[0]
    n_respondents = len(respondent_ids)

    def task_matrix(column: str) -> torch.Tensor:
        array = data[column].to_numpy(dtype=float).reshape(n_respondents, tasks_per_respondent)
        return torch.tensor(array)

    respondent_frame = data.groupby("respondent_id").first().reset_index()

    tensors = {
        "choice": torch.tensor(data["choice"].to_numpy(dtype=int).reshape(n_respondents, tasks_per_respondent) - 1),
        "age": torch.tensor(respondent_frame["age"].to_numpy(dtype=float)),
        "hhcar": torch.tensor(respondent_frame["hhcar"].to_numpy(dtype=float)),
        "hhbike": torch.tensor(respondent_frame["hhbike"].to_numpy(dtype=float)),
        "ptpass": torch.tensor(respondent_frame["ptpass"].to_numpy(dtype=float)),
        "maas": torch.tensor(respondent_frame["maas"].to_numpy(dtype=float)),
        "time_es": task_matrix("time_es"),
        "time_bs": task_matrix("time_bs"),
        "time_walk": task_matrix("time_walk"),
        "time_car": task_matrix("time_car"),
        "access_es": task_matrix("access_es"),
        "access_bs": task_matrix("access_bs"),
        "access_car": task_matrix("access_car"),
        "egress_es": task_matrix("egress_es"),
        "egress_bs": task_matrix("egress_bs"),
        "egress_car": task_matrix("egress_car"),
        "parking_es": task_matrix("parking_es"),
        "parking_bs": task_matrix("parking_bs"),
        "parking_car": task_matrix("parking_car"),
        "cost_es": task_matrix("cost_es"),
        "cost_bs": task_matrix("cost_bs"),
        "cost_car": task_matrix("cost_car"),
        "availability_es": task_matrix("availability_es"),
        "availability_bs": task_matrix("availability_bs"),
        "freefloat_es": task_matrix("freefloat_es"),
        "freefloat_bs": task_matrix("freefloat_bs"),
        "range_es": task_matrix("range_es"),
        "range_bs": task_matrix("range_bs"),
        "pedelec_bs": task_matrix("pedelec_bs"),
    }
    return tensors, respondent_ids


def unpack(theta: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: theta[index] for index, name in enumerate(PARAMETER_ORDER)}


def negative_log_likelihood(
    theta: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    draws: torch.Tensor,
    respondent_batch_size: int,
    draw_batch_size: int,
) -> torch.Tensor:
    params = unpack(theta)
    n_draws = draws.shape[0]
    n_respondents = int(tensors["choice"].shape[0])
    total_negloglik = torch.tensor(0.0)

    for respondent_start in range(0, n_respondents, respondent_batch_size):
        respondent_end = min(respondent_start + respondent_batch_size, n_respondents)
        respondent_slice = slice(respondent_start, respondent_end)

        age = tensors["age"][respondent_slice][:, None]
        hhcar = tensors["hhcar"][respondent_slice][:, None]
        hhbike = tensors["hhbike"][respondent_slice][:, None]
        ptpass = tensors["ptpass"][respondent_slice][:, None]
        maas = tensors["maas"][respondent_slice][:, None]

        time_es = tensors["time_es"][respondent_slice]
        time_bs = tensors["time_bs"][respondent_slice]
        time_walk = tensors["time_walk"][respondent_slice]
        time_car = tensors["time_car"][respondent_slice]
        access_es = tensors["access_es"][respondent_slice]
        access_bs = tensors["access_bs"][respondent_slice]
        access_car = tensors["access_car"][respondent_slice]
        egress_es = tensors["egress_es"][respondent_slice]
        egress_bs = tensors["egress_bs"][respondent_slice]
        egress_car = tensors["egress_car"][respondent_slice]
        parking_es = tensors["parking_es"][respondent_slice]
        parking_bs = tensors["parking_bs"][respondent_slice]
        parking_car = tensors["parking_car"][respondent_slice]
        cost_es = tensors["cost_es"][respondent_slice]
        cost_bs = tensors["cost_bs"][respondent_slice]
        cost_car = tensors["cost_car"][respondent_slice]
        availability_es = tensors["availability_es"][respondent_slice]
        availability_bs = tensors["availability_bs"][respondent_slice]
        freefloat_es = tensors["freefloat_es"][respondent_slice]
        freefloat_bs = tensors["freefloat_bs"][respondent_slice]
        range_es = tensors["range_es"][respondent_slice]
        range_bs = tensors["range_bs"][respondent_slice]
        pedelec_bs = tensors["pedelec_bs"][respondent_slice]
        choice = tensors["choice"][respondent_slice]

        running_logsumexp = None
        for draw_start in range(0, n_draws, draw_batch_size):
            draw_end = min(draw_start + draw_batch_size, n_draws)
            draw_chunk = draws[draw_start:draw_end]

            z_cost = draw_chunk[:, 0]
            z_es = draw_chunk[:, 1]
            z_bs = draw_chunk[:, 2]
            z_walk = draw_chunk[:, 3]

            asc_es = (
                params["ASC_ES_REL_CAR"]
                + params["AGE_ES_REL_CAR"] * age
                + params["BIKEACC_ES_REL_CAR"] * hhbike
                + params["CARACC_ES_REL_CAR"] * hhcar
                + params["PTPASS_ES_REL_CAR"] * ptpass
                + params["MAAS_ES_REL_CAR"] * maas
                + params["SIGMA_ES"] * z_es[None, :]
            )
            asc_bs = (
                params["ASC_BS_REL_CAR"]
                + params["AGE_BS_REL_CAR"] * age
                + params["BIKEACC_BS_REL_CAR"] * hhbike
                + params["CARACC_BS_REL_CAR"] * hhcar
                + params["PTPASS_BS_REL_CAR"] * ptpass
                + params["MAAS_BS_REL_CAR"] * maas
                + params["SIGMA_BS"] * z_bs[None, :]
            )
            asc_walk = (
                params["ASC_WALK_REL_CAR"]
                + params["AGE_WALK_REL_CAR"] * age
                + params["BIKEACC_WALK_REL_CAR"] * hhbike
                + params["CARACC_WALK_REL_CAR"] * hhcar
                + params["PTPASS_WALK_REL_CAR"] * ptpass
                + params["MAAS_WALK_REL_CAR"] * maas
                + params["SIGMA_WALK"] * z_walk[None, :]
            )
            beta_cost = -torch.exp(params["B_COST_LOGMEAN"] + params["SIGMA_COST"] * z_cost)

            u_es = (
                asc_es[:, None, :]
                + params["B_TIME_ES"] * time_es[:, :, None]
                + params["B_ACCESS_SHARED"] * access_es[:, :, None]
                + params["B_EGRESS_SHARED"] * egress_es[:, :, None]
                + params["B_PARKING"] * parking_es[:, :, None]
                + beta_cost[None, None, :] * cost_es[:, :, None]
                + params["B_AVAILABILITY"] * availability_es[:, :, None]
                + params["B_SCHEME_SD"] * freefloat_es[:, :, None]
                + params["B_RANGE"] * range_es[:, :, None]
            )
            u_bs = (
                asc_bs[:, None, :]
                + params["B_TIME_BS"] * time_bs[:, :, None]
                + params["B_ACCESS_SHARED"] * access_bs[:, :, None]
                + params["B_EGRESS_SHARED"] * egress_bs[:, :, None]
                + params["B_PARKING"] * parking_bs[:, :, None]
                + beta_cost[None, None, :] * cost_bs[:, :, None]
                + params["B_AVAILABILITY"] * availability_bs[:, :, None]
                + params["B_SCHEME_SD"] * freefloat_bs[:, :, None]
                + params["B_RANGE"] * range_bs[:, :, None]
                + params["B_PEDELEC"] * pedelec_bs[:, :, None]
            )
            u_walk = asc_walk[:, None, :] + params["B_TIME_WALK"] * time_walk[:, :, None]
            u_car = (
                params["B_TIME_CAR"] * time_car[:, :, None]
                + params["B_ACCESS_CAR"] * access_car[:, :, None]
                + params["B_EGRESS_CAR"] * egress_car[:, :, None]
                + params["B_PARKING"] * parking_car[:, :, None]
                + beta_cost[None, None, :] * cost_car[:, :, None]
            )

            utilities = torch.stack([u_es, u_bs, u_walk, u_car], dim=2)
            log_denom = torch.logsumexp(utilities, dim=2)
            choice_index = choice[:, :, None, None].expand(-1, -1, 1, draw_end - draw_start)
            chosen_utilities = torch.gather(utilities, 2, choice_index).squeeze(2)
            panel_log_prob_draw = (chosen_utilities - log_denom).sum(dim=1)
            chunk_logsumexp = torch.logsumexp(panel_log_prob_draw, dim=1)
            if running_logsumexp is None:
                running_logsumexp = chunk_logsumexp
            else:
                running_logsumexp = torch.logaddexp(running_logsumexp, chunk_logsumexp)

        total_negloglik = total_negloglik - (running_logsumexp - math.log(n_draws)).sum()

    return total_negloglik


def optimize_model(
    tensors: dict[str, torch.Tensor],
    draws: torch.Tensor,
    respondent_batch_size: int,
    draw_batch_size: int,
) -> tuple[np.ndarray, float]:
    theta = torch.tensor(initial_theta(), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [theta],
        max_iter=200,
        tolerance_grad=1e-6,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    last_loss = None

    def closure():
        nonlocal last_loss
        optimizer.zero_grad()
        loss = negative_log_likelihood(theta, tensors, draws, respondent_batch_size, draw_batch_size)
        loss.backward()
        last_loss = float(loss.detach().cpu().item())
        return loss

    optimizer.step(closure)
    return theta.detach().cpu().numpy(), float(last_loss)


def compute_hessian_and_stats(
    estimates: np.ndarray,
    tensors: dict[str, torch.Tensor],
    draws: torch.Tensor,
    respondent_batch_size: int,
    draw_batch_size: int,
) -> pd.DataFrame:
    theta_hat = torch.tensor(estimates, requires_grad=True)

    def objective(theta_vector: torch.Tensor) -> torch.Tensor:
        return negative_log_likelihood(theta_vector, tensors, draws, respondent_batch_size, draw_batch_size)

    hessian = torch.autograd.functional.hessian(objective, theta_hat).detach().cpu().numpy()
    covariance = np.linalg.pinv(hessian)
    std_error = np.sqrt(np.clip(np.diag(covariance), a_min=0.0, a_max=None))
    z_value = np.divide(estimates, std_error, out=np.full_like(estimates, np.nan), where=std_error > 0)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(z_value)))
    return pd.DataFrame(
        {
            "parameter_name": PARAMETER_ORDER,
            "estimate": estimates,
            "std_error": std_error,
            "z_value": z_value,
            "p_value": p_value,
            "significant_5pct": (p_value < 0.05).astype(int),
        }
    )


def main() -> None:
    config = load_json(DATA_DIR / "experiment_config.json")
    tensors, respondent_ids = prepare_tensors()
    draws_np = sobol_normal_draws(
        n_draws=int(config["mixed_logit"]["n_draws"]),
        dim=len(config["mixed_logit"]["random_dimensions"]),
    )
    draws = torch.tensor(draws_np)
    respondent_batch_size = int(config["mixed_logit"].get("respondent_batch_size", 128))
    draw_batch_size = int(config["mixed_logit"].get("draw_batch_size", 256))

    estimates, final_negloglik = optimize_model(tensors, draws, respondent_batch_size, draw_batch_size)
    estimates_frame = compute_hessian_and_stats(
        estimates,
        tensors,
        draws,
        respondent_batch_size,
        draw_batch_size,
    )
    estimates_frame.to_csv(OUTPUT_DIR / "mixed_choice_estimates.csv", index=False)

    n_obs = int(tensors["choice"].numel())
    null_loglik = n_obs * math.log(1.0 / 4.0)
    final_loglik = -final_negloglik
    summary = {
        "model_type": "paper_aligned_sd_panel_mixed_logit_subset",
        "draw_type": config["mixed_logit"]["draw_type"],
        "n_draws": int(config["mixed_logit"]["n_draws"]),
        "respondent_batch_size": respondent_batch_size,
        "draw_batch_size": draw_batch_size,
        "n_respondents": len(respondent_ids),
        "n_observations": n_obs,
        "n_parameters": len(PARAMETER_ORDER),
        "null_loglikelihood": null_loglik,
        "final_loglikelihood": final_loglik,
        "rho_square": 1 - (final_loglik / null_loglik),
        "normalization": config["mixed_logit"]["normalization"],
    }
    (OUTPUT_DIR / "mixed_choice_model_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
