from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from replicate_atasoy_2011_models import (
    BASE_PARAMETER_ORDER,
    PAPER_BASE_TARGETS,
    PRO_CAR_INDICATORS,
    ENV_INDICATORS,
    RAW_DATA_FILE,
    approx_standard_errors,
    base_choice_probabilities,
    own_elasticity,
    weighted_market_shares,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT_DIR / "experiments" / "Swissmetro"
HUMAN_BASE_SUMMARY_FILE = ROOT_DIR / "data" / "Swissmetro" / "demographic_choice_psychometric" / "atasoy_2011_replication" / "base_logit_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dirs", nargs="+", required=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def deduplicated_person_frame() -> pd.DataFrame:
    raw = pd.read_csv(RAW_DATA_FILE, sep="\t")
    keep_columns = ["ID", "LangCode", "UrbRur", "OccupStat", "Education", "Region", "Weight"]
    person = raw[keep_columns].drop_duplicates(subset=["ID"]).copy()
    return person.rename(columns={"ID": "human_id"})


def validate_experiment_dir(experiment_dir: Path) -> None:
    required = [
        experiment_dir / "outputs" / "run_respondents.json",
        experiment_dir / "parsed_task_responses.csv",
        experiment_dir / "parsed_attitudes.csv",
        experiment_dir / "ai_panel_long.csv",
        experiment_dir / "persona_samples.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files for {experiment_dir.name}: {missing}")
    payload = json.loads((experiment_dir / "outputs" / "run_respondents.json").read_text(encoding="utf-8"))
    if int(payload["completed_respondents"]) != int(payload["target_respondents"]):
        raise RuntimeError(
            f"Experiment {experiment_dir.name} is not complete: {payload['completed_respondents']} / {payload['target_respondents']}"
        )


def build_ai_core_frame(experiment_dir: Path, person_frame: pd.DataFrame) -> pd.DataFrame:
    persona = pd.read_csv(experiment_dir / "persona_samples.csv")
    panel = pd.read_csv(experiment_dir / "ai_panel_long.csv")
    persona = persona.merge(person_frame, on="human_id", how="left", suffixes=("", "_raw"))
    if persona[["LangCode", "UrbRur", "OccupStat", "Education", "Region"]].isna().any().any():
        raise RuntimeError(f"Missing human covariates after merge for {experiment_dir.name}")
    persona = persona.copy()
    persona["French"] = (persona["LangCode"] == 1).astype(int)
    persona["Urban"] = (persona["UrbRur"] == 2).astype(int)
    persona["Student"] = (persona["OccupStat"] == 8).astype(int)
    persona["EducHigh"] = (persona["Education"] >= 6).astype(int)
    persona["WorkTrip"] = persona["work_trip"].astype(int)
    persona["NbCar_0"] = persona["NbCar"].clip(lower=0)
    persona["NbChild_0"] = persona["NbChild"].clip(lower=0)
    persona["NbBicy_0"] = persona["NbBicy"].clip(lower=0)
    keep = [
        "respondent_id",
        "human_id",
        "normalized_weight",
        "French",
        "Urban",
        "Student",
        "EducHigh",
        "WorkTrip",
        "NbCar_0",
        "NbChild_0",
        "NbBicy_0",
        "LangCode",
        "UrbRur",
        "OccupStat",
        "Education",
        "Region",
    ]
    persona = persona[keep].drop_duplicates(subset=["respondent_id"]).copy()
    frame = panel.merge(persona, on="respondent_id", how="left", suffixes=("", "_persona"))
    if frame[["French", "Urban", "Student", "WorkTrip", "NbCar_0", "NbChild_0", "NbBicy_0"]].isna().any().any():
        raise RuntimeError(f"Missing merged AI covariates for {experiment_dir.name}")
    frame = frame.loc[frame["task_role"] == "core"].copy()
    frame = frame.loc[frame["is_valid_task_response"] == 1].copy()
    frame = frame.sort_values(["respondent_id", "task_index", "alternative_code"]).reset_index(drop=True)
    return frame


def ai_base_choice_probabilities(parameters: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    asc_pmm, asc_sm, beta_cost, beta_time_pmm, beta_time_pt, beta_distance, beta_ncars, beta_nchildren, beta_language, beta_work, beta_urban, beta_student, beta_nbikes = parameters
    utility = np.zeros(len(frame), dtype=float)
    is_pt = frame["alternative_name"].to_numpy() == "PT"
    is_pmm = frame["alternative_name"].to_numpy() == "CAR"
    is_sm = frame["alternative_name"].to_numpy() == "SLOW_MODES"
    utility[is_pt] = (
        beta_cost * frame.loc[is_pt, "alt_cost"].to_numpy(float)
        + beta_time_pt * frame.loc[is_pt, "alt_time"].to_numpy(float)
        + beta_urban * frame.loc[is_pt, "Urban"].to_numpy(float)
        + beta_student * frame.loc[is_pt, "Student"].to_numpy(float)
    )
    utility[is_pmm] = (
        asc_pmm
        + beta_cost * frame.loc[is_pmm, "alt_cost"].to_numpy(float)
        + beta_time_pmm * frame.loc[is_pmm, "alt_time"].to_numpy(float)
        + beta_ncars * frame.loc[is_pmm, "NbCar_0"].to_numpy(float)
        + beta_nchildren * frame.loc[is_pmm, "NbChild_0"].to_numpy(float)
        + beta_language * frame.loc[is_pmm, "French"].to_numpy(float)
        + beta_work * frame.loc[is_pmm, "WorkTrip"].to_numpy(float)
    )
    utility[is_sm] = (
        asc_sm
        + beta_distance * frame.loc[is_sm, "alt_distance"].to_numpy(float)
        + beta_nbikes * frame.loc[is_sm, "NbBicy_0"].to_numpy(float)
    )
    utility[frame["alt_available"].to_numpy(dtype=int) == 0] = -1.0e10
    work = frame[["respondent_id", "task_index", "alternative_code"]].copy()
    work["utility"] = utility
    max_u = work.groupby(["respondent_id", "task_index"])["utility"].transform("max")
    work["exp_u"] = np.exp(work["utility"] - max_u)
    denom = work.groupby(["respondent_id", "task_index"])["exp_u"].transform("sum")
    work["prob"] = work["exp_u"] / np.clip(denom, 1.0e-300, None)
    return work["prob"].to_numpy(float)


def ai_base_negative_log_likelihood(parameters: np.ndarray, frame: pd.DataFrame) -> float:
    probabilities = ai_base_choice_probabilities(parameters, frame)
    chosen_mask = frame["chosen"].to_numpy(dtype=int) == 1
    chosen_probabilities = probabilities[chosen_mask]
    return -float(np.sum(np.log(np.clip(chosen_probabilities, 1.0e-300, None))))


def estimate_ai_base_logit(frame: pd.DataFrame) -> dict[str, object]:
    start = np.array([PAPER_BASE_TARGETS[name] for name in BASE_PARAMETER_ORDER], dtype=float)
    result = minimize(
        ai_base_negative_log_likelihood,
        start,
        args=(frame,),
        method="BFGS",
        options={"maxiter": 500, "gtol": 1.0e-6},
    )
    estimates = result.x
    standard_errors = approx_standard_errors(result)
    probabilities = ai_base_choice_probabilities(estimates, frame)
    work = frame.copy()
    work["prob"] = probabilities
    chosen = work.loc[work["chosen"] == 1].copy()
    weights = chosen["normalized_weight"].to_numpy(float)
    task_probabilities = (
        work[["respondent_id", "task_index", "alternative_name", "prob", "normalized_weight", "alt_cost", "alt_time", "alt_distance"]]
        .copy()
    )
    task_probabilities["weighted_prob"] = task_probabilities["prob"] * task_probabilities["normalized_weight"]
    market_shares = {
        "PT": float(task_probabilities.loc[task_probabilities["alternative_name"] == "PT", "weighted_prob"].sum() / task_probabilities["normalized_weight"].sum() * 3.0),
        "PMM": float(task_probabilities.loc[task_probabilities["alternative_name"] == "CAR", "weighted_prob"].sum() / task_probabilities["normalized_weight"].sum() * 3.0),
        "SM": float(task_probabilities.loc[task_probabilities["alternative_name"] == "SLOW_MODES", "weighted_prob"].sum() / task_probabilities["normalized_weight"].sum() * 3.0),
    }
    # elasticity calculations on long rows
    probability_column = work["prob"].to_numpy(float)
    long_weights = work["normalized_weight"].to_numpy(float)
    metrics = {
        "log_likelihood": float(-result.fun),
        "market_shares": market_shares,
        "value_of_time_chf_per_hour": {
            "PMM": float(60.0 * abs(estimates[3] / estimates[2])),
            "PT": float(60.0 * abs(estimates[4] / estimates[2])),
        },
        "elasticities": {
            "PMM_cost": float(
                np.sum(
                    long_weights[(work["alternative_name"] == "CAR").to_numpy()]
                    * probability_column[(work["alternative_name"] == "CAR").to_numpy()]
                    * estimates[2]
                    * work.loc[work["alternative_name"] == "CAR", "alt_cost"].to_numpy(float)
                    * (1.0 - probability_column[(work["alternative_name"] == "CAR").to_numpy()])
                )
                / np.sum(long_weights[(work["alternative_name"] == "CAR").to_numpy()] * probability_column[(work["alternative_name"] == "CAR").to_numpy()])
            ),
            "PMM_time": float(
                np.sum(
                    long_weights[(work["alternative_name"] == "CAR").to_numpy()]
                    * probability_column[(work["alternative_name"] == "CAR").to_numpy()]
                    * estimates[3]
                    * work.loc[work["alternative_name"] == "CAR", "alt_time"].to_numpy(float)
                    * (1.0 - probability_column[(work["alternative_name"] == "CAR").to_numpy()])
                )
                / np.sum(long_weights[(work["alternative_name"] == "CAR").to_numpy()] * probability_column[(work["alternative_name"] == "CAR").to_numpy()])
            ),
            "PT_cost": float(
                np.sum(
                    long_weights[(work["alternative_name"] == "PT").to_numpy()]
                    * probability_column[(work["alternative_name"] == "PT").to_numpy()]
                    * estimates[2]
                    * work.loc[work["alternative_name"] == "PT", "alt_cost"].to_numpy(float)
                    * (1.0 - probability_column[(work["alternative_name"] == "PT").to_numpy()])
                )
                / np.sum(long_weights[(work["alternative_name"] == "PT").to_numpy()] * probability_column[(work["alternative_name"] == "PT").to_numpy()])
            ),
            "PT_time": float(
                np.sum(
                    long_weights[(work["alternative_name"] == "PT").to_numpy()]
                    * probability_column[(work["alternative_name"] == "PT").to_numpy()]
                    * estimates[4]
                    * work.loc[work["alternative_name"] == "PT", "alt_time"].to_numpy(float)
                    * (1.0 - probability_column[(work["alternative_name"] == "PT").to_numpy()])
                )
                / np.sum(long_weights[(work["alternative_name"] == "PT").to_numpy()] * probability_column[(work["alternative_name"] == "PT").to_numpy()])
            ),
        },
    }
    rows = []
    for index, name in enumerate(BASE_PARAMETER_ORDER):
        row = {
            "parameter_name": name,
            "estimate": float(estimates[index]),
            "paper_human_estimate": float(PAPER_BASE_TARGETS[name]),
        }
        if standard_errors is not None and np.isfinite(standard_errors[index]) and standard_errors[index] > 0:
            row["std_error"] = float(standard_errors[index])
            row["z_value"] = float(estimates[index] / standard_errors[index])
        rows.append(row)
    return {
        "result": result,
        "estimates_table": pd.DataFrame(rows),
        "metrics": metrics,
    }


def hcm_feasibility(experiment_dir: Path) -> dict[str, object]:
    attitudes = pd.read_csv(experiment_dir / "parsed_attitudes.csv")
    available = sorted(attitudes["indicator_name"].dropna().unique().tolist())
    required = sorted(set(PRO_CAR_INDICATORS + ENV_INDICATORS))
    missing = [name for name in required if name not in available]
    return {
        "available_indicators": available,
        "required_for_exact_atasoy_hcm": required,
        "missing_required_indicators": missing,
        "is_exact_atasoy_hcm_feasible": len(missing) == 0,
        "reason": (
            "Exact Atasoy 2011 continuous HCM is not feasible on existing AI outputs because the required paper indicators were not collected."
            if missing
            else "Exact Atasoy 2011 continuous HCM is feasible from existing indicators."
        ),
    }


def write_experiment_report(experiment_dir: Path, base_results: dict[str, object], feasibility: dict[str, object]) -> None:
    human_summary = json.loads(HUMAN_BASE_SUMMARY_FILE.read_text(encoding="utf-8"))
    lines = []
    lines.append(f"# {experiment_dir.name} Atasoy 2011 analysis")
    lines.append("")
    lines.append("This note applies the Atasoy, Glerum, and Bierlaire (2011) base logit specification to the existing AI core-choice outputs of this experiment without sending any new model requests.")
    lines.append("")
    lines.append("## Base logit")
    lines.append("")
    lines.append(
        f"The AI-side Atasoy-style base logit uses the six `core` tasks only. It keeps the paper utility structure and merges the missing socio-demographic controls from the original human source respondent linked through `human_id`."
    )
    lines.append("")
    lines.append("| Metric | Human paper replication | This AI experiment |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| log-likelihood | {human_summary['metrics']['log_likelihood']:.3f} | {base_results['metrics']['log_likelihood']:.3f} |")
    lines.append(f"| PMM VOT (CHF/hour) | {human_summary['metrics']['value_of_time_chf_per_hour']['PMM']:.2f} | {base_results['metrics']['value_of_time_chf_per_hour']['PMM']:.2f} |")
    lines.append(f"| PT VOT (CHF/hour) | {human_summary['metrics']['value_of_time_chf_per_hour']['PT']:.2f} | {base_results['metrics']['value_of_time_chf_per_hour']['PT']:.2f} |")
    lines.append(f"| PMM share | {human_summary['metrics']['market_shares']['PMM']:.4f} | {base_results['metrics']['market_shares']['PMM']:.4f} |")
    lines.append(f"| PT share | {human_summary['metrics']['market_shares']['PT']:.4f} | {base_results['metrics']['market_shares']['PT']:.4f} |")
    lines.append(f"| SM share | {human_summary['metrics']['market_shares']['SM']:.4f} | {base_results['metrics']['market_shares']['SM']:.4f} |")
    lines.append("")
    lines.append("## Exact HCM feasibility")
    lines.append("")
    if feasibility["is_exact_atasoy_hcm_feasible"]:
        lines.append("The exact Atasoy 2011 continuous HCM is feasible from the current AI outputs.")
    else:
        lines.append("The exact Atasoy 2011 continuous HCM is not feasible from the current AI outputs.")
        lines.append("")
        lines.append(f"Missing required indicators: {', '.join(feasibility['missing_required_indicators'])}")
        lines.append("")
        lines.append("The current intervention-regime AI survey collected only these six attitude questions:")
        lines.append("")
        lines.append(", ".join(feasibility["available_indicators"]))
    (experiment_dir / "atasoy_2011_replication" / "ai_atasoy_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_experiment(experiment_dir: Path) -> None:
    validate_experiment_dir(experiment_dir)
    output_dir = ensure_dir(experiment_dir / "atasoy_2011_replication")
    person_frame = deduplicated_person_frame()
    core_frame = build_ai_core_frame(experiment_dir, person_frame)
    base_results = estimate_ai_base_logit(core_frame)
    feasibility = hcm_feasibility(experiment_dir)
    summary = {
        "experiment_name": experiment_dir.name,
        "n_core_rows_long": int(len(core_frame)),
        "n_core_tasks": int(core_frame.loc[core_frame["chosen"] == 1, ["respondent_id", "task_index"]].drop_duplicates().shape[0]),
        "n_respondents": int(core_frame["respondent_id"].nunique()),
        "metrics": base_results["metrics"],
        "optimizer_success": bool(base_results["result"].success),
        "optimizer_message": str(base_results["result"].message),
    }
    base_results["estimates_table"].to_csv(output_dir / "ai_atasoy_base_logit_estimates.csv", index=False)
    (output_dir / "ai_atasoy_base_logit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "ai_atasoy_hcm_feasibility.json").write_text(json.dumps(feasibility, indent=2), encoding="utf-8")
    write_experiment_report(experiment_dir, base_results, feasibility)


def main() -> None:
    args = parse_args()
    for name in args.experiment_dirs:
        experiment_dir = Path(name)
        if not experiment_dir.is_absolute():
            experiment_dir = EXPERIMENTS_DIR / experiment_dir
        analyze_experiment(experiment_dir)


if __name__ == "__main__":
    main()
