from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

import biogeme.biogeme as bio
import biogeme.database as db
import biogeme.models as models
from biogeme.expressions import Beta, MonteCarlo, PanelLikelihoodTrajectory, Variable, bioDraws, exp, log


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXPERIMENT_DIR / "data"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

ALT_SUFFIX = {
    "e_scooter": "ES",
    "bikesharing": "BS",
    "walking": "WALK",
    "private_car": "CAR",
    "carsharing": "CS",
    "ridepooling": "RP",
    "public_transport": "PT",
}

ALT_CODE = {
    "e_scooter": 1,
    "bikesharing": 2,
    "walking": 3,
    "private_car": 4,
    "carsharing": 5,
    "ridepooling": 6,
    "public_transport": 7,
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

DRAW_COMPONENTS = [
    ("z_cost", "SOBOL_NORMAL_1"),
    ("z_es", "SOBOL_NORMAL_2"),
    ("z_bs", "SOBOL_NORMAL_3"),
    ("z_walk", "SOBOL_NORMAL_4"),
    ("z_car", "SOBOL_NORMAL_5"),
    ("z_cs", "SOBOL_NORMAL_6"),
    ("z_rp", "SOBOL_NORMAL_7"),
    ("z_pt", "SOBOL_NORMAL_8"),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_output_dir(output_subdir: str | None) -> Path:
    if not output_subdir:
        return DEFAULT_OUTPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_human_targets() -> dict[str, float]:
    human = pd.read_csv(DATA_DIR / "human_table4_pooled_full.csv")
    return dict(zip(human["parameter_name"], human["human_estimate"]))


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
    base = long_frame[base_columns].drop_duplicates().set_index(
        [
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
    )

    numeric_columns = list(VALUE_PREFIX.keys())
    wide = base.copy()
    for column_name in numeric_columns:
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

    wide["AV_WALK"] = wide["AV_WALK"].clip(lower=0, upper=1)
    wide["AV_ES"] = wide["AV_ES"].clip(lower=0, upper=1)
    wide["AV_BS"] = wide["AV_BS"].clip(lower=0, upper=1)
    wide["AV_CAR"] = wide["AV_CAR"].clip(lower=0, upper=1)
    wide["AV_CS"] = wide["AV_CS"].clip(lower=0, upper=1)
    wide["AV_RP"] = wide["AV_RP"].clip(lower=0, upper=1)
    wide["AV_PT"] = wide["AV_PT"].clip(lower=0, upper=1)
    return wide


def make_sobol_generator(dim_index: int, total_dims: int, seed: int):
    def sobol_normal(sample_size: int, number_of_draws: int) -> np.ndarray:
        n_points = sample_size * number_of_draws
        m = math.ceil(math.log2(max(2, n_points)))
        sampler = qmc.Sobol(d=total_dims, scramble=True, seed=seed)
        draws = sampler.random_base2(m=m)[:n_points, dim_index]
        draws = np.clip(draws, 1e-10, 1 - 1e-10)
        return norm.ppf(draws).reshape(sample_size, number_of_draws)

    return sobol_normal


def build_sobol_generators(seed: int) -> dict[str, tuple]:
    generators = {}
    total_dims = len(DRAW_COMPONENTS)
    for index, (_, draw_type) in enumerate(DRAW_COMPONENTS):
        generators[draw_type] = (
            make_sobol_generator(index, total_dims, seed),
            f"Scrambled Sobol normal draw, dimension {index + 1} of {total_dims}",
        )
    return generators


def beta(name: str, start_values: dict[str, float], lower: float | None = None, upper: float | None = None) -> Beta:
    return Beta(name, float(start_values.get(name, 0.0)), lower, upper, 0)


def write_estimates(results, output_dir: Path) -> pd.DataFrame:
    estimates = results.getEstimatedParameters(onlyRobust=False).reset_index()
    if "index" in estimates.columns:
        estimates = estimates.rename(columns={"index": "parameter_name"})
    elif estimates.columns[0] != "parameter_name":
        estimates = estimates.rename(columns={estimates.columns[0]: "parameter_name"})

    rename_map = {
        "Value": "estimate",
        "Std err": "std_error",
        "t-test": "z_value",
        "p-value": "p_value",
        "Rob. Std err": "robust_std_error",
        "Rob. t-test": "robust_z_value",
        "Rob. p-value": "robust_p_value",
    }
    estimates = estimates.rename(columns={old: new for old, new in rename_map.items() if old in estimates.columns})
    p_column = "robust_p_value" if "robust_p_value" in estimates.columns else "p_value"
    estimates["significant_5pct"] = (estimates[p_column] <= 0.05).astype(int)
    estimates.to_csv(output_dir / "biogeme_mixed_estimates.csv", index=False)
    return estimates


def write_model_summary(results, wide_frame: pd.DataFrame, n_draws: int, output_dir: Path) -> None:
    summary = {
        "experiment_name": EXPERIMENT_DIR.name,
        "model_name": "krauss_public_materials_pooled_7alts_panel_mixed_logit_biogeme",
        "n_respondents": int(wide_frame["RESPONDENT_ID"].nunique()),
        "n_observations": int(len(wide_frame)),
        "n_draws": int(n_draws),
        "final_loglikelihood": float(results.data.logLike),
        "null_loglikelihood": float(results.data.nullLogLike) if results.data.nullLogLike is not None else None,
        "rho_square": float(results.data.rhoSquare) if results.data.rhoSquare is not None else None,
        "rho_square_null": float(results.data.rhoSquareNull) if results.data.rhoSquareNull is not None else None,
        "n_parameters": int(results.numberOfFreeParameters()),
    }
    (output_dir / "biogeme_mixed_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_biogeme_parameter_file(path: Path, n_draws: int, seed: int, optimization_algorithm: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# Auto-generated parameter file for v7 pooled public-materials replication",
                "",
                "[Specification]",
                "missing_data = 99999",
                "",
                "[Estimation]",
                f'optimization_algorithm = "{optimization_algorithm}"',
                'save_iterations = "False"',
                "max_number_parameters_to_report = 25",
                "",
                "[Output]",
                'only_robust_stats = "False"',
                'generate_html = "False"',
                'generate_pickle = "False"',
                "",
                "[MonteCarlo]",
                f"number_of_draws = {int(n_draws)}",
                f"seed = {int(seed)}",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-draws", type=int, default=None)
    parser.add_argument("--max-respondents", type=int, default=None)
    parser.add_argument("--output-subdir", type=str, default=None)
    args = parser.parse_args()

    config = load_json(DATA_DIR / "experiment_config.json")
    n_draws = int(args.n_draws if args.n_draws is not None else config["biogeme"]["n_draws"])
    output_dir = resolve_output_dir(args.output_subdir)
    start_values = load_human_targets()
    optimization_algorithm = str(config["biogeme"].get("optimization_algorithm", "scipy"))

    long_frame = pd.read_csv(DEFAULT_OUTPUT_DIR / "pooled_choices_long.csv")
    long_frame = complete_panel_only(long_frame, int(config["tasks_per_respondent"]))
    if args.max_respondents is not None:
        keep_ids = sorted(long_frame["respondent_id"].unique().tolist())[: args.max_respondents]
        long_frame = long_frame.loc[long_frame["respondent_id"].isin(keep_ids)].copy()
    if long_frame.empty:
        raise ValueError("No complete valid panels available in pooled_choices_long.csv.")

    wide_frame = build_wide_estimation_frame(long_frame)
    wide_frame.to_csv(output_dir / "biogeme_estimation_wide.csv", index=False)
    estimation_frame = wide_frame.drop(columns=["PERSONA_ID", "SUBSAMPLE", "TASK_ID", "CHOSEN_ALTERNATIVE_ID"]).copy()
    parameter_file = output_dir / "biogeme.toml"
    write_biogeme_parameter_file(
        parameter_file,
        n_draws=n_draws,
        seed=int(config["estimation_seed"]),
        optimization_algorithm=optimization_algorithm,
    )

    database = db.Database("kka_public_materials_pooled", estimation_frame)
    database.panel("RESPONDENT_ID")
    database.setRandomNumberGenerators(build_sobol_generators(int(config["estimation_seed"])))

    AGE = Variable("AGE")
    HHCAR = Variable("HHCAR")
    HHBIKE = Variable("HHBIKE")
    PTPASS = Variable("PTPASS")
    MAAS = Variable("MAAS")
    CHOICE = Variable("CHOICE")

    AV_ES = Variable("AV_ES")
    AV_BS = Variable("AV_BS")
    AV_WALK = Variable("AV_WALK")
    AV_CAR = Variable("AV_CAR")
    AV_CS = Variable("AV_CS")
    AV_RP = Variable("AV_RP")
    AV_PT = Variable("AV_PT")

    TIME_ES = Variable("TIME_ES")
    TIME_BS = Variable("TIME_BS")
    TIME_WALK = Variable("TIME_WALK")
    TIME_CAR = Variable("TIME_CAR")
    TIME_CS = Variable("TIME_CS")
    TIME_RP = Variable("TIME_RP")
    TIME_PT = Variable("TIME_PT")

    ACCESS_ES = Variable("ACCESS_ES")
    ACCESS_BS = Variable("ACCESS_BS")
    ACCESS_CAR = Variable("ACCESS_CAR")
    ACCESS_CS = Variable("ACCESS_CS")
    ACCESS_RP = Variable("ACCESS_RP")
    ACCESS_PT = Variable("ACCESS_PT")

    WAIT_RP = Variable("WAIT_RP")
    WAIT_PT = Variable("WAIT_PT")

    EGRESS_ES = Variable("EGRESS_ES")
    EGRESS_BS = Variable("EGRESS_BS")
    EGRESS_CAR = Variable("EGRESS_CAR")
    EGRESS_CS = Variable("EGRESS_CS")
    EGRESS_RP = Variable("EGRESS_RP")
    EGRESS_PT = Variable("EGRESS_PT")

    DETOUR_RP = Variable("DETOUR_RP")

    PARKING_ES = Variable("PARKING_ES")
    PARKING_BS = Variable("PARKING_BS")
    PARKING_CAR = Variable("PARKING_CAR")
    PARKING_CS = Variable("PARKING_CS")

    COST_ES = Variable("COST_ES")
    COST_BS = Variable("COST_BS")
    COST_CAR = Variable("COST_CAR")
    COST_CS = Variable("COST_CS")
    COST_RP = Variable("COST_RP")
    COST_PT = Variable("COST_PT")

    AVAILABILITY_ES = Variable("AVAILABILITY_ES")
    AVAILABILITY_BS = Variable("AVAILABILITY_BS")
    FREEFLOAT_ES = Variable("FREEFLOAT_ES")
    FREEFLOAT_BS = Variable("FREEFLOAT_BS")
    FREEFLOAT_CS = Variable("FREEFLOAT_CS")
    HYBRID_CS = Variable("HYBRID_CS")
    RANGE_ES = Variable("RANGE_ES")
    RANGE_BS = Variable("RANGE_BS")
    PEDELEC_BS = Variable("PEDELEC_BS")
    CROWDING_RP = Variable("CROWDING_RP")
    CROWDING_PT = Variable("CROWDING_PT")
    TRANSFER_PT = Variable("TRANSFER_PT")

    ASC_ES = beta("ASC_ES", start_values)
    ASC_BS = beta("ASC_BS", start_values)
    ASC_WALK = beta("ASC_WALK", start_values)
    ASC_CS = beta("ASC_CS", start_values)
    ASC_RP = beta("ASC_RP", start_values)
    ASC_PT = beta("ASC_PT", start_values)

    B_AGE_ES = beta("B_AGE_ES", start_values)
    B_AGE_BS = beta("B_AGE_BS", start_values)
    B_AGE_WALK = beta("B_AGE_WALK", start_values)
    B_AGE_CS = beta("B_AGE_CS", start_values)
    B_AGE_RP = beta("B_AGE_RP", start_values)
    B_AGE_PT = beta("B_AGE_PT", start_values)

    B_BIKEACC_ES = beta("B_BIKEACC_ES", start_values)
    B_BIKEACC_BS = beta("B_BIKEACC_BS", start_values)
    B_BIKEACC_WALK = beta("B_BIKEACC_WALK", start_values)
    B_BIKEACC_CS = beta("B_BIKEACC_CS", start_values)
    B_BIKEACC_RP = beta("B_BIKEACC_RP", start_values)
    B_BIKEACC_PT = beta("B_BIKEACC_PT", start_values)

    B_CARACC_ES = beta("B_CARACC_ES", start_values)
    B_CARACC_BS = beta("B_CARACC_BS", start_values)
    B_CARACC_WALK = beta("B_CARACC_WALK", start_values)
    B_CARACC_CS = beta("B_CARACC_CS", start_values)
    B_CARACC_RP = beta("B_CARACC_RP", start_values)
    B_CARACC_PT = beta("B_CARACC_PT", start_values)

    B_PTPASS_ES = beta("B_PTPASS_ES", start_values)
    B_PTPASS_BS = beta("B_PTPASS_BS", start_values)
    B_PTPASS_WALK = beta("B_PTPASS_WALK", start_values)
    B_PTPASS_CS = beta("B_PTPASS_CS", start_values)
    B_PTPASS_RP = beta("B_PTPASS_RP", start_values)
    B_PTPASS_PT = beta("B_PTPASS_PT", start_values)

    B_MAAS_ES = beta("B_MAAS_ES", start_values)
    B_MAAS_BS = beta("B_MAAS_BS", start_values)
    B_MAAS_WALK = beta("B_MAAS_WALK", start_values)
    B_MAAS_CS = beta("B_MAAS_CS", start_values)
    B_MAAS_RP = beta("B_MAAS_RP", start_values)
    B_MAAS_PT = beta("B_MAAS_PT", start_values)

    B_TIME_ES = beta("B_TIME_ES", start_values)
    B_TIME_BS = beta("B_TIME_BS", start_values)
    B_TIME_WALK = beta("B_TIME_WALK", start_values)
    B_TIME_CAR = beta("B_TIME_CAR", start_values)
    B_TIME_CS = beta("B_TIME_CS", start_values)
    B_TIME_RP = beta("B_TIME_RP", start_values)
    B_TIME_PT = beta("B_TIME_PT", start_values)

    B_ACCESS_SHARED = beta("B_ACCESS_SHARED", start_values)
    B_ACCESS_OWNED = beta("B_ACCESS_OWNED", start_values)
    B_EGRESS_SHARED = beta("B_EGRESS_SHARED", start_values)
    B_EGRESS_OWNED = beta("B_EGRESS_OWNED", start_values)
    B_DETOUR_RP = beta("B_DETOUR_RP", start_values)
    B_PARKING = beta("B_PARKING", start_values)
    B_COST_LOGMEAN = beta("B_COST_LOGMEAN", start_values)
    SIGMA_COST = beta("SIGMA_COST", start_values)
    B_AVAILABILITY = beta("B_AVAILABILITY", start_values)
    B_SCHEME_SD_FREE_FLOAT = beta("B_SCHEME_SD_FREE_FLOAT", start_values)
    B_SCHEME_LD_FREE_FLOAT = beta("B_SCHEME_LD_FREE_FLOAT", start_values)
    B_SCHEME_LD_HYBRID = beta("B_SCHEME_LD_HYBRID", start_values)
    B_PEDELEC = beta("B_PEDELEC", start_values)
    B_RANGE = beta("B_RANGE", start_values)
    B_CROWDING = beta("B_CROWDING", start_values)
    B_TRANSFER_PT = beta("B_TRANSFER_PT", start_values)
    SIGMA_ES = beta("SIGMA_ES", start_values)
    SIGMA_BS = beta("SIGMA_BS", start_values)
    SIGMA_WALK = beta("SIGMA_WALK", start_values)
    SIGMA_CAR = beta("SIGMA_CAR", start_values)
    SIGMA_CS = beta("SIGMA_CS", start_values)
    SIGMA_RP = beta("SIGMA_RP", start_values)
    SIGMA_PT = beta("SIGMA_PT", start_values)
    PHI_POOL = beta("PHI_POOL", start_values, lower=1e-6)

    z_cost = bioDraws("z_cost", "SOBOL_NORMAL_1")
    z_es = bioDraws("z_es", "SOBOL_NORMAL_2")
    z_bs = bioDraws("z_bs", "SOBOL_NORMAL_3")
    z_walk = bioDraws("z_walk", "SOBOL_NORMAL_4")
    z_car = bioDraws("z_car", "SOBOL_NORMAL_5")
    z_cs = bioDraws("z_cs", "SOBOL_NORMAL_6")
    z_rp = bioDraws("z_rp", "SOBOL_NORMAL_7")
    z_pt = bioDraws("z_pt", "SOBOL_NORMAL_8")

    B_COST = -exp(B_COST_LOGMEAN + SIGMA_COST * z_cost)
    RND_ASC_ES = SIGMA_ES * z_es
    RND_ASC_BS = SIGMA_BS * z_bs
    RND_ASC_WALK = SIGMA_WALK * z_walk
    RND_ASC_CAR = SIGMA_CAR * z_car
    RND_ASC_CS = SIGMA_CS * z_cs
    RND_ASC_RP = SIGMA_RP * z_rp
    RND_ASC_PT = SIGMA_PT * z_pt

    V = {
        1: ASC_ES
        + B_AGE_ES * AGE
        + B_BIKEACC_ES * HHBIKE
        + B_CARACC_ES * HHCAR
        + B_PTPASS_ES * PTPASS
        + B_MAAS_ES * MAAS
        + RND_ASC_ES
        + B_TIME_ES * TIME_ES
        + B_ACCESS_SHARED * ACCESS_ES
        + B_EGRESS_SHARED * EGRESS_ES
        + B_PARKING * PARKING_ES
        + B_COST * COST_ES
        + B_AVAILABILITY * AVAILABILITY_ES
        + B_SCHEME_SD_FREE_FLOAT * FREEFLOAT_ES
        + B_RANGE * RANGE_ES,
        2: ASC_BS
        + B_AGE_BS * AGE
        + B_BIKEACC_BS * HHBIKE
        + B_CARACC_BS * HHCAR
        + B_PTPASS_BS * PTPASS
        + B_MAAS_BS * MAAS
        + RND_ASC_BS
        + B_TIME_BS * TIME_BS
        + B_ACCESS_SHARED * ACCESS_BS
        + B_EGRESS_SHARED * EGRESS_BS
        + B_PARKING * PARKING_BS
        + B_COST * COST_BS
        + B_AVAILABILITY * AVAILABILITY_BS
        + B_SCHEME_SD_FREE_FLOAT * FREEFLOAT_BS
        + B_PEDELEC * PEDELEC_BS
        + B_RANGE * RANGE_BS,
        3: ASC_WALK
        + B_AGE_WALK * AGE
        + B_BIKEACC_WALK * HHBIKE
        + B_CARACC_WALK * HHCAR
        + B_PTPASS_WALK * PTPASS
        + B_MAAS_WALK * MAAS
        + RND_ASC_WALK
        + B_TIME_WALK * TIME_WALK,
        4: RND_ASC_CAR
        + PHI_POOL * (
            B_TIME_CAR * TIME_CAR
            + B_ACCESS_OWNED * ACCESS_CAR
            + B_EGRESS_OWNED * EGRESS_CAR
            + B_PARKING * PARKING_CAR
            + B_COST * COST_CAR
        ),
        5: ASC_CS
        + B_AGE_CS * AGE
        + B_BIKEACC_CS * HHBIKE
        + B_CARACC_CS * HHCAR
        + B_PTPASS_CS * PTPASS
        + B_MAAS_CS * MAAS
        + RND_ASC_CS
        + B_TIME_CS * TIME_CS
        + B_ACCESS_SHARED * ACCESS_CS
        + B_EGRESS_SHARED * EGRESS_CS
        + B_PARKING * PARKING_CS
        + B_COST * COST_CS
        + B_SCHEME_LD_FREE_FLOAT * FREEFLOAT_CS
        + B_SCHEME_LD_HYBRID * HYBRID_CS,
        6: ASC_RP
        + B_AGE_RP * AGE
        + B_BIKEACC_RP * HHBIKE
        + B_CARACC_RP * HHCAR
        + B_PTPASS_RP * PTPASS
        + B_MAAS_RP * MAAS
        + RND_ASC_RP
        + B_TIME_RP * TIME_RP
        + B_DETOUR_RP * DETOUR_RP
        + B_ACCESS_SHARED * (ACCESS_RP + WAIT_RP)
        + B_EGRESS_SHARED * EGRESS_RP
        + B_COST * COST_RP
        + B_CROWDING * CROWDING_RP,
        7: ASC_PT
        + B_AGE_PT * AGE
        + B_BIKEACC_PT * HHBIKE
        + B_CARACC_PT * HHCAR
        + B_PTPASS_PT * PTPASS
        + B_MAAS_PT * MAAS
        + RND_ASC_PT
        + B_TIME_PT * TIME_PT
        + B_ACCESS_SHARED * (ACCESS_PT + WAIT_PT)
        + B_EGRESS_OWNED * EGRESS_PT
        + B_COST * COST_PT
        + B_CROWDING * CROWDING_PT
        + B_TRANSFER_PT * TRANSFER_PT,
    }

    av = {1: AV_ES, 2: AV_BS, 3: AV_WALK, 4: AV_CAR, 5: AV_CS, 6: AV_RP, 7: AV_PT}
    obsprob = models.logit(V, av, CHOICE)
    panelprob = PanelLikelihoodTrajectory(obsprob)
    logprob = log(MonteCarlo(panelprob))

    model = bio.BIOGEME(
        database,
        logprob,
        parameter_file=str(parameter_file),
    )
    model.modelName = "krauss_public_materials_pooled_7alts_panel_mixed_logit_biogeme"
    model.calculateNullLoglikelihood(av)
    results = model.estimate()

    estimates = write_estimates(results, output_dir=output_dir)
    write_model_summary(results, wide_frame, n_draws, output_dir=output_dir)
    print(
        f"[biogeme] respondents={wide_frame['RESPONDENT_ID'].nunique()} "
        f"observations={len(wide_frame)} draws={n_draws} "
        f"optimizer={optimization_algorithm} "
        f"loglik={results.data.logLike:.3f} "
        f"params={len(estimates)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
