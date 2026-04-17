from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from optima_common import ROOT_DIR, SOURCE_OBSERVATION_COLUMN, ensure_dir, read_json, write_json
from replicate_atasoy_2011_models import (
    ALL_CONTINUOUS_INDICATORS,
    FIXED_ENV_REFERENCE,
    FIXED_PRO_CAR_REFERENCE,
    RAW_DATA_FILE,
    add_atasoy_covariates,
    estimate_base_model,
    estimate_continuous_model,
    fixed_continuous_initial_result,
    prepare_replication_frame,
)

HUMAN_REPLICATION_DIR = ROOT_DIR / "data" / "Swissmetro" / "demographic_choice_psychometric" / "atasoy_2011_replication"
HUMAN_BASE_ESTIMATES_FILE = HUMAN_REPLICATION_DIR / "base_logit" / "base_logit_estimates.csv"
HUMAN_BASE_SUMMARY_FILE = HUMAN_REPLICATION_DIR / "base_logit" / "base_logit_summary.json"
HUMAN_HCM_SUMMARY_FILE = HUMAN_REPLICATION_DIR / "hcm" / "hcm_summary.json"
HUMAN_HCM_UTILITY_FILE = HUMAN_REPLICATION_DIR / "hcm" / "hcm_utility_estimates.csv"
HUMAN_HCM_ATTITUDE_FILE = HUMAN_REPLICATION_DIR / "hcm" / "hcm_attitude_estimates.csv"
HUMAN_HCM_MEASUREMENT_FILE = HUMAN_REPLICATION_DIR / "hcm" / "hcm_measurement_estimates.csv"
_HUMAN_FRAME_TEMPLATE: pd.DataFrame | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dirs", nargs="+", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def human_frame_template() -> pd.DataFrame:
    global _HUMAN_FRAME_TEMPLATE
    if _HUMAN_FRAME_TEMPLATE is None:
        _HUMAN_FRAME_TEMPLATE = prepare_replication_frame().head(0).copy()
    return _HUMAN_FRAME_TEMPLATE.copy()


def source_frame() -> pd.DataFrame:
    raw = pd.read_csv(RAW_DATA_FILE, sep="\t")
    raw[SOURCE_OBSERVATION_COLUMN] = raw.index.astype(int) + 1
    keep_columns = [
        SOURCE_OBSERVATION_COLUMN,
        "ID",
        "LangCode",
        "UrbRur",
        "OccupStat",
        "TripPurpose",
        "Education",
        "Region",
        "Weight",
    ]
    return raw[keep_columns].rename(columns={"ID": "human_id"})


def validate_experiment_dir(experiment_dir: Path, allow_partial: bool = False) -> dict[str, int]:
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
    payload = read_json(experiment_dir / "outputs" / "run_respondents.json")
    completed = int(payload["completed_respondents"])
    target = int(payload["target_respondents"])
    if completed != target and not allow_partial:
        raise RuntimeError(
            f"Experiment {experiment_dir.name} is not complete: {completed} / {target}"
        )
    return {
        "completed_respondents": completed,
        "target_respondents": target,
        "is_partial_sample": int(completed != target),
    }


def enrich_persona_frame(experiment_dir: Path) -> pd.DataFrame:
    persona = pd.read_csv(experiment_dir / "persona_samples.csv").copy()
    persona = persona.drop_duplicates(subset=["respondent_id"]).copy()

    raw_source = source_frame()
    if SOURCE_OBSERVATION_COLUMN in persona.columns:
        persona = persona.merge(
            raw_source,
            on=[SOURCE_OBSERVATION_COLUMN, "human_id"],
            how="left",
            suffixes=("", "_raw"),
        )
    else:
        fallback = raw_source.drop_duplicates(subset=["human_id"]).copy()
        persona = persona.merge(fallback, on="human_id", how="left", suffixes=("", "_raw"))

    for column in ["Weight", "LangCode", "UrbRur", "OccupStat", "TripPurpose", "Education", "Region", SOURCE_OBSERVATION_COLUMN]:
        raw_column = f"{column}_raw"
        if column not in persona.columns and raw_column in persona.columns:
            persona[column] = persona[raw_column]
        elif raw_column in persona.columns:
            persona[column] = persona[column].where(persona[column].notna(), persona[raw_column])

    required = [
        "respondent_id",
        "human_id",
        SOURCE_OBSERVATION_COLUMN,
        "Weight",
        "LangCode",
        "UrbRur",
        "OccupStat",
        "TripPurpose",
        "Education",
        "Region",
        "age",
        "NbCar",
        "NbChild",
        "NbBicy",
    ]
    missing = [column for column in required if column not in persona.columns]
    if missing:
        raise RuntimeError(f"Missing required persona columns for {experiment_dir.name}: {missing}")

    for indicator in ALL_CONTINUOUS_INDICATORS:
        if indicator not in persona.columns:
            persona[indicator] = np.nan

    return add_atasoy_covariates(persona)


def build_ai_replication_frame(experiment_dir: Path, persona: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(experiment_dir / "ai_panel_long.csv")
    panel = panel.loc[(panel["task_role"] == "core") & (panel["is_valid_task_response"] == 1)].copy()
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (respondent_id, task_index), group in panel.groupby(["respondent_id", "task_index"], sort=True):
        group = group.sort_values("alternative_code")
        if set(group["alternative_code"].astype(int).tolist()) != {0, 1, 2}:
            continue
        chosen = group.loc[group["chosen"] == 1]
        if len(chosen) != 1:
            continue
        pt = group.loc[group["alternative_code"] == 0].iloc[0]
        car = group.loc[group["alternative_code"] == 1].iloc[0]
        slow = group.loc[group["alternative_code"] == 2].iloc[0]
        rows.append(
            {
                "respondent_id": str(respondent_id),
                "task_index": int(task_index),
                "Choice": int(chosen.iloc[0]["alternative_code"]),
                "WaitingTimePT": float(pt["alt_waiting"]),
                "TimePT": float(pt["alt_time"]),
                "MarginalCostPT": float(pt["alt_cost"]),
                "CostPT": float(pt["alt_cost"]),
                "TimeCar": float(car["alt_time"]),
                "CostCar": float(car["alt_cost"]),
                "CostCarCHF": float(car["alt_cost"]),
                "distance_km": float(slow["alt_distance"]),
                "CarAvail": int(car["alt_available"]),
                "CAR_AVAILABLE": int(car["alt_available"]),
                "InVehicleTime": float(pt["alt_time"] - pt["alt_waiting"]),
                "source_observation_id_panel": int(pt.get(SOURCE_OBSERVATION_COLUMN, -1)),
                "human_id_panel": int(pt.get("human_id", -1)),
            }
        )

    task_frame = pd.DataFrame(rows)
    if task_frame.empty:
        return task_frame, task_frame

    merge_columns = [
        "respondent_id",
        "human_id",
        SOURCE_OBSERVATION_COLUMN,
        "Weight",
        "CalculatedIncome",
        "Education",
        "LangCode",
        "UrbRur",
        "OccupStat",
        "TripPurpose",
        "Region",
        "age",
        "NbHousehold",
        "NbChild",
        "NbCar",
        "NbBicy",
        "French",
        "Urban",
        "Student",
        "WorkTrip",
        "EducHigh",
        "AgeTerm",
        "Valais",
        "Bern",
        "BaselZurich",
        "East",
        "Graubunden",
        "NbCar_0",
        "NbChild_0",
        "NbBicy_0",
        *ALL_CONTINUOUS_INDICATORS,
    ]
    task_frame = task_frame.merge(persona[merge_columns], on="respondent_id", how="left")
    if task_frame[
        [
            "human_id",
            SOURCE_OBSERVATION_COLUMN,
            "Weight",
            "LangCode",
            "UrbRur",
            "OccupStat",
            "TripPurpose",
            "Education",
            "Region",
            "age",
            "NbCar",
            "NbChild",
            "NbBicy",
            "AgeTerm",
            "NbCar_0",
            "NbChild_0",
            "NbBicy_0",
        ]
    ].isna().any().any():
        raise RuntimeError(f"Missing merged Atasoy covariates for {experiment_dir.name}")

    task_frame["ID"] = task_frame["human_id"]

    template = human_frame_template()
    exact_frame = task_frame.reindex(columns=template.columns).copy()
    return exact_frame, task_frame.sort_values(["respondent_id", "task_index"]).reset_index(drop=True)


def hcm_feasibility(experiment_dir: Path) -> dict[str, object]:
    attitudes = pd.read_csv(experiment_dir / "parsed_attitudes.csv")
    available = sorted(attitudes["indicator_name"].dropna().unique().tolist())
    required = sorted(set(ALL_CONTINUOUS_INDICATORS))
    missing = [name for name in required if name not in available]
    return {
        "available_indicators": available,
        "required_for_exact_atasoy_hcm": required,
        "missing_required_indicators": missing,
        "is_exact_atasoy_hcm_feasible": len(missing) == 0,
        "reason": (
            "Exact Atasoy 2011 continuous HCM is not feasible on these AI outputs because the required paper indicators were not collected."
            if missing
            else "Exact Atasoy 2011 continuous HCM is feasible from the collected AI indicators."
        ),
    }


def human_base_estimates() -> pd.DataFrame:
    return pd.read_csv(HUMAN_BASE_ESTIMATES_FILE)


def human_hcm_estimates() -> pd.DataFrame:
    utility = pd.read_csv(HUMAN_HCM_UTILITY_FILE)
    attitude = pd.read_csv(HUMAN_HCM_ATTITUDE_FILE)
    measurement = pd.read_csv(HUMAN_HCM_MEASUREMENT_FILE)
    return pd.concat([utility, attitude, measurement], ignore_index=True)


def base_parameter_comparison(base_results: dict[str, object]) -> pd.DataFrame:
    human = human_base_estimates()[["parameter_name", "estimate"]].rename(columns={"estimate": "human_estimate"})
    ai = (
        base_results["estimates_table"]
        .drop(columns=["paper_estimate"], errors="ignore")
        [["parameter_name", "estimate"]]
        .rename(columns={"estimate": "ai_estimate"})
    )
    comparison = human.merge(ai, on="parameter_name", how="outer")
    comparison["model"] = "atasoy_2011_replication"
    comparison["gap_ai_minus_human"] = comparison["ai_estimate"] - comparison["human_estimate"]
    return comparison[["model", "parameter_name", "human_estimate", "ai_estimate", "gap_ai_minus_human"]].sort_values("parameter_name").reset_index(drop=True)


def hcm_parameter_comparison(hcm_results: dict[str, object]) -> pd.DataFrame:
    human = human_hcm_estimates()[["parameter_name", "estimate", "block"]].rename(columns={"estimate": "human_estimate"})
    ai = pd.concat(
        [
            hcm_results["utility_table"].drop(columns=["paper_estimate"], errors="ignore")[["parameter_name", "estimate", "block"]],
            hcm_results["attitude_table"].drop(columns=["paper_estimate"], errors="ignore")[["parameter_name", "estimate", "block"]],
            hcm_results["measurement_table"][["parameter_name", "estimate", "block"]],
        ],
        ignore_index=True,
    ).rename(columns={"estimate": "ai_estimate"})
    comparison = human.merge(ai, on=["parameter_name", "block"], how="outer")
    comparison["model"] = "hcm"
    comparison["gap_ai_minus_human"] = comparison["ai_estimate"] - comparison["human_estimate"]
    return comparison[["model", "block", "parameter_name", "human_estimate", "ai_estimate", "gap_ai_minus_human"]].sort_values(["block", "parameter_name"]).reset_index(drop=True)


def empty_hcm_parameter_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["model", "block", "parameter_name", "human_estimate", "ai_estimate", "gap_ai_minus_human"]
    )


def write_experiment_report(
    experiment_dir: Path,
    sample_size: int,
    n_respondents: int,
    progress: dict[str, int],
    base_results: dict[str, object],
    feasibility: dict[str, object],
    hcm_results: dict[str, object] | None,
) -> None:
    human_base = read_json(HUMAN_BASE_SUMMARY_FILE)
    lines = []
    lines.append(f"# {experiment_dir.name} Atasoy 2011 analysis")
    lines.append("")
    lines.append(
        "This note applies the same Atasoy 2011 base logit and fixed-normalization continuous HCM estimation code used by the human replication to the AI outputs in this experiment."
    )
    lines.append("")
    lines.append(
        f"The AI estimation input is first reorganized into the same Atasoy-style row format as the human replication. The current sample contains `{sample_size}` core-task observations from `{n_respondents}` completed AI respondents."
    )
    if bool(progress["is_partial_sample"]):
        lines.append("")
        lines.append(
            f"This is a partial-sample analysis run on `{progress['completed_respondents']}` / `{progress['target_respondents']}` planned respondents because the collection was stopped early."
        )
    lines.append("")
    lines.append("## Base logit")
    lines.append("")
    lines.append("| Metric | Human paper replication | This AI experiment |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| log-likelihood | {human_base['metrics']['log_likelihood']:.3f} | {base_results['metrics']['log_likelihood']:.3f} |")
    lines.append(f"| PMM VOT (CHF/hour) | {human_base['metrics']['value_of_time_chf_per_hour']['PMM']:.2f} | {base_results['metrics']['value_of_time_chf_per_hour']['PMM']:.2f} |")
    lines.append(f"| PT VOT (CHF/hour) | {human_base['metrics']['value_of_time_chf_per_hour']['PT']:.2f} | {base_results['metrics']['value_of_time_chf_per_hour']['PT']:.2f} |")
    lines.append(f"| PMM share | {human_base['metrics']['market_shares']['PMM']:.4f} | {base_results['metrics']['market_shares']['PMM']:.4f} |")
    lines.append(f"| PT share | {human_base['metrics']['market_shares']['PT']:.4f} | {base_results['metrics']['market_shares']['PT']:.4f} |")
    lines.append(f"| SM share | {human_base['metrics']['market_shares']['SM']:.4f} | {base_results['metrics']['market_shares']['SM']:.4f} |")
    lines.append("")
    lines.append("## Exact HCM")
    lines.append("")
    if not feasibility["is_exact_atasoy_hcm_feasible"]:
        lines.append("The exact Atasoy 2011 continuous HCM is not feasible from these AI outputs.")
        lines.append("")
        lines.append(f"Missing required indicators: {', '.join(feasibility['missing_required_indicators'])}")
    elif hcm_results is None:
        lines.append("The required indicators are present, but the exact HCM estimation did not produce results.")
    else:
        human_hcm = read_json(HUMAN_HCM_SUMMARY_FILE)
        lines.append(
            "The AI exact HCM uses the same fixed normalization and the same model equations as the human benchmark: `Mobil10` for the pro-car attitude and `Envir05` for the environmental attitude. The human side is stored as a paper-aligned canonical benchmark, while the AI side is estimated with the repository local-basin optimizer under that same normalization."
        )
        lines.append("")
        lines.append("| Metric | Human HCM | This AI experiment |")
        lines.append("| --- | ---: | ---: |")
        lines.append(f"| choice-only log-likelihood | {human_hcm['metrics']['choice_log_likelihood']:.3f} | {hcm_results['metrics']['choice_log_likelihood']:.3f} |")
        lines.append(f"| PMM VOT (CHF/hour) | {human_hcm['metrics']['value_of_time_chf_per_hour']['PMM']:.2f} | {hcm_results['metrics']['value_of_time_chf_per_hour']['PMM']:.2f} |")
        lines.append(f"| PT VOT (CHF/hour) | {human_hcm['metrics']['value_of_time_chf_per_hour']['PT']:.2f} | {hcm_results['metrics']['value_of_time_chf_per_hour']['PT']:.2f} |")
        lines.append(f"| mean Acar | {human_hcm['metrics']['mean_acar']:.3f} | {hcm_results['metrics']['mean_acar']:.3f} |")
        lines.append(f"| mean Aenv | {human_hcm['metrics']['mean_aenv']:.3f} | {hcm_results['metrics']['mean_aenv']:.3f} |")
    (experiment_dir / "ai_atasoy_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_experiment(experiment_dir: Path, allow_partial: bool = False) -> None:
    progress = validate_experiment_dir(experiment_dir, allow_partial=allow_partial)
    atasoy_dir = ensure_dir(experiment_dir / "atasoy_2011_replication")
    hcm_dir = ensure_dir(experiment_dir / "hcm")

    persona = enrich_persona_frame(experiment_dir)
    exact_frame, trace_frame = build_ai_replication_frame(experiment_dir, persona)
    if exact_frame.empty:
        raise RuntimeError(f"No valid core tasks found for {experiment_dir.name}")

    n_respondents = int(trace_frame["respondent_id"].nunique())
    sample_size = int(len(exact_frame))

    exact_frame.to_csv(experiment_dir / "ai_atasoy_replication_input.csv", index=False)
    trace_frame.to_csv(experiment_dir / "ai_atasoy_replication_trace.csv", index=False)

    base_results = estimate_base_model(exact_frame)
    base_summary = {
        "sample_size": sample_size,
        "n_respondents": n_respondents,
        "completed_respondents": int(progress["completed_respondents"]),
        "target_respondents": int(progress["target_respondents"]),
        "is_partial_sample": bool(progress["is_partial_sample"]),
        "specification": base_results["specification"],
        "metrics": base_results["metrics"],
        "optimizer_success": bool(base_results["result"].success),
        "optimizer_message": str(base_results["result"].message),
    }
    base_estimates = base_results["estimates_table"].drop(columns=["paper_estimate"], errors="ignore").copy()
    base_estimates.to_csv(atasoy_dir / "ai_atasoy_base_logit_estimates.csv", index=False)
    write_json(atasoy_dir / "ai_atasoy_base_logit_summary.json", base_summary)
    base_parameter_comparison(base_results).to_csv(atasoy_dir / "parameter_comparison.csv", index=False)

    feasibility = hcm_feasibility(experiment_dir)
    hcm_results = None
    if feasibility["is_exact_atasoy_hcm_feasible"]:
        fixed_initial = fixed_continuous_initial_result(
            exact_frame,
            ref_pro_car_indicator=FIXED_PRO_CAR_REFERENCE,
            ref_env_indicator=FIXED_ENV_REFERENCE,
        )
        hcm_results = estimate_continuous_model(
            exact_frame,
            ref_pro_car_indicator=FIXED_PRO_CAR_REFERENCE,
            ref_env_indicator=FIXED_ENV_REFERENCE,
            start_vector=fixed_initial.x.copy(),
            initial_result=fixed_initial,
        )
        hcm_summary = {
            "sample_size": sample_size,
            "n_respondents": n_respondents,
            "completed_respondents": int(progress["completed_respondents"]),
            "target_respondents": int(progress["target_respondents"]),
            "is_partial_sample": bool(progress["is_partial_sample"]),
            "normalization": hcm_results["normalization"],
            "indicator_mapping": hcm_results["indicator_mapping"],
            "selection_rule": "fixed repository normalization with ref_pro_car_indicator=Mobil10 and ref_env_indicator=Envir05; short paper-target warm start followed by bounded local-basin convergence",
            "metrics": hcm_results["metrics"],
            "optimizer_success": bool(hcm_results["result"].success),
            "optimizer_message": str(hcm_results["result"].message),
        }
        utility_table = hcm_results["utility_table"].drop(columns=["paper_estimate"], errors="ignore").copy()
        attitude_table = hcm_results["attitude_table"].drop(columns=["paper_estimate"], errors="ignore").copy()
        measurement_table = hcm_results["measurement_table"].copy()
        hcm_results["utility_table"] = utility_table
        hcm_results["attitude_table"] = attitude_table
        hcm_results["measurement_table"] = measurement_table
        ai_estimates = pd.concat(
            [
                hcm_results["utility_table"],
                hcm_results["attitude_table"],
                hcm_results["measurement_table"],
            ],
            ignore_index=True,
        )
        hcm_results["utility_table"].to_csv(hcm_dir / "ai_atasoy_hcm_utility_estimates.csv", index=False)
        hcm_results["attitude_table"].to_csv(hcm_dir / "ai_atasoy_hcm_attitude_estimates.csv", index=False)
        hcm_results["measurement_table"].to_csv(hcm_dir / "ai_atasoy_hcm_measurement_estimates.csv", index=False)
        ai_estimates.to_csv(hcm_dir / "ai_atasoy_hcm_estimates.csv", index=False)
        write_json(hcm_dir / "ai_atasoy_hcm_summary.json", hcm_summary)
        hcm_parameter_comparison(hcm_results).to_csv(hcm_dir / "parameter_comparison.csv", index=False)
    else:
        empty_hcm_parameter_comparison().to_csv(hcm_dir / "parameter_comparison.csv", index=False)

    write_json(experiment_dir / "ai_atasoy_hcm_feasibility.json", feasibility)
    write_experiment_report(experiment_dir, sample_size, n_respondents, progress, base_results, feasibility, hcm_results)


def main() -> None:
    args = parse_args()
    for experiment_dir_str in args.experiment_dirs:
        analyze_experiment(Path(experiment_dir_str).resolve(), allow_partial=bool(args.allow_partial))


if __name__ == "__main__":
    main()
