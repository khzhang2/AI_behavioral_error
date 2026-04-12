from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from optima_common import CONFIG, EXPERIMENT_DIR, OUTPUT_DIR, SOURCE_DATA_DIR, archive_experiment_config, ensure_dir, experiment_artifact_path, llm_config_for, write_json


CORE_COLUMNS = [
    "TimePT",
    "WaitingTimePT",
    "MarginalCostPT",
    "TimeCar",
    "CostCarCHF",
    "distance_km",
    "TimePT_scaled",
    "WaitingTimePT_scaled",
    "MarginalCostPT_scaled",
    "TimeCar_scaled",
    "CostCarCHF_scaled",
    "distance_km_scaled",
    "CAR_AVAILABLE",
]

PROFILE_COLUMNS = [
    "respondent_id",
    "human_id",
    "normalized_weight",
    "age",
    "sex_text",
    "age_text",
    "CalculatedIncome",
    "income_text",
    "high_education",
    "low_education",
    "top_manager",
    "employees",
    "artisans",
    "age_30_less",
    "ScaledIncome",
    "car_oriented_parents",
    "city_center_as_kid",
    "childSuburb",
    "NbCar",
    "NbBicy",
    "NbHousehold",
    "NbChild",
    "work_trip",
    "other_trip",
    "trip_purpose_text",
    "education_text",
    "CAR_AVAILABLE",
    "car_availability_text",
    "Envir01",
    "Mobil05",
    "LifSty07",
    "Envir05",
    "Mobil12",
    "LifSty01",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    return parser.parse_args()


def alternative_proxies(row: pd.Series) -> dict[str, float]:
    return {
        "PT": float(row["TimePT_scaled"] + row["WaitingTimePT_scaled"] + row["MarginalCostPT_scaled"]),
        "CAR": float(row["TimeCar_scaled"] + row["CostCarCHF_scaled"]) if int(row["CAR_AVAILABLE"]) == 1 else 1.0e6,
        "SLOW_MODES": float(row["distance_km_scaled"]),
    }


def scenario_bank_from_human_data(frame: pd.DataFrame) -> pd.DataFrame:
    scenario = frame[["respondent_id", "human_id", *CORE_COLUMNS]].copy()
    pt_proxy = frame["TimePT_scaled"] + frame["WaitingTimePT_scaled"] + frame["MarginalCostPT_scaled"]
    car_proxy = frame["TimeCar_scaled"] + frame["CostCarCHF_scaled"]
    slow_proxy = frame["distance_km_scaled"]
    scenario["pt_proxy"] = pt_proxy
    scenario["car_proxy"] = car_proxy.where(frame["CAR_AVAILABLE"] == 1, 1.0e6)
    scenario["slow_proxy"] = slow_proxy
    pairwise_gap = pd.concat(
        [
            (scenario["pt_proxy"] - scenario["car_proxy"]).abs(),
            (scenario["pt_proxy"] - scenario["slow_proxy"]).abs(),
            (scenario["car_proxy"] - scenario["slow_proxy"]).abs(),
        ],
        axis=1,
    ).min(axis=1)
    scenario["complexity_score"] = 1.0 / (1.0 + pairwise_gap)
    scenario["complexity_rank"] = scenario["complexity_score"].rank(pct=True, method="average")
    scenario["scenario_id"] = [f"S{index + 1:04d}" for index in range(len(scenario))]
    return scenario


def build_task_row(
    base_row: pd.Series,
    respondent_row: pd.Series,
    model_key: str,
    prompt_arm: str,
    respondent_id: str,
    task_index: int,
    task_role: str,
    pair_id: str,
    manipulation_type: str,
    anchor_task_index: int | None,
    semantic_labels: bool,
    option_order: str,
) -> dict:
    payload = {
        "model_key": model_key,
        "respondent_id": respondent_id,
        "human_respondent_id": str(respondent_row["respondent_id"]),
        "human_id": int(respondent_row["human_id"]),
        "normalized_weight": float(respondent_row["normalized_weight"]),
        "prompt_arm": prompt_arm,
        "semantic_arm": int(prompt_arm == "semantic_arm"),
        "task_index": int(task_index),
        "task_role": task_role,
        "pair_id": pair_id,
        "manipulation_type": manipulation_type,
        "anchor_task_index": anchor_task_index if anchor_task_index is not None else -1,
        "semantic_labels": int(semantic_labels),
        "option_order": option_order,
        "display_A_alt": option_order.split("|")[0],
        "display_B_alt": option_order.split("|")[1],
        "display_C_alt": option_order.split("|")[2],
        "scenario_id": str(base_row["scenario_id"]),
        "complexity_score": float(base_row["complexity_score"]),
        "CAR_AVAILABLE": int(base_row["CAR_AVAILABLE"]),
        "TimePT": float(base_row["TimePT"]),
        "WaitingTimePT": float(base_row["WaitingTimePT"]),
        "MarginalCostPT": float(base_row["MarginalCostPT"]),
        "TimeCar": float(base_row["TimeCar"]),
        "CostCarCHF": float(base_row["CostCarCHF"]),
        "distance_km": float(base_row["distance_km"]),
        "TimePT_scaled": float(base_row["TimePT_scaled"]),
        "WaitingTimePT_scaled": float(base_row["WaitingTimePT_scaled"]),
        "MarginalCostPT_scaled": float(base_row["MarginalCostPT_scaled"]),
        "TimeCar_scaled": float(base_row["TimeCar_scaled"]),
        "CostCarCHF_scaled": float(base_row["CostCarCHF_scaled"]),
        "distance_km_scaled": float(base_row["distance_km_scaled"]),
        "target_alternative_name": "",
        "dominated_alternative_name": "",
        "dominance_reason": "",
    }
    return payload


def worsen_task(row: dict, multiplier: float) -> dict:
    updated = dict(row)
    proxy = alternative_proxies(pd.Series(updated))
    target = min(proxy, key=proxy.get)
    updated["target_alternative_name"] = target
    if target == "PT":
        updated["TimePT"] *= multiplier
        updated["WaitingTimePT"] *= multiplier
        updated["MarginalCostPT"] *= multiplier
        updated["TimePT_scaled"] *= multiplier
        updated["WaitingTimePT_scaled"] *= multiplier
        updated["MarginalCostPT_scaled"] *= multiplier
    elif target == "CAR":
        updated["TimeCar"] *= multiplier
        updated["CostCarCHF"] *= multiplier
        updated["TimeCar_scaled"] *= multiplier
        updated["CostCarCHF_scaled"] *= multiplier
    else:
        updated["distance_km"] *= multiplier
        updated["distance_km_scaled"] *= multiplier
    return updated


def dominance_task(row: dict, config: dict) -> dict:
    updated = dict(row)
    penalty_time = float(config["dominance_time_penalty_min"])
    penalty_wait = float(config["dominance_wait_penalty_min"])
    penalty_cost = float(config["dominance_cost_penalty_chf"])
    if int(updated["CAR_AVAILABLE"]) == 1:
        pt_proxy = float(updated["TimePT_scaled"] + updated["WaitingTimePT_scaled"] + updated["MarginalCostPT_scaled"])
        car_proxy = float(updated["TimeCar_scaled"] + updated["CostCarCHF_scaled"])
        if car_proxy <= pt_proxy:
            updated["dominated_alternative_name"] = "PT"
            updated["dominance_reason"] = "PT is clearly worse than CAR on the displayed burden attributes."
            updated["TimePT"] = max(float(updated["TimePT"]), float(updated["TimeCar"]) + penalty_time)
            updated["WaitingTimePT"] = max(float(updated["WaitingTimePT"]), penalty_wait)
            updated["MarginalCostPT"] = max(float(updated["MarginalCostPT"]), float(updated["CostCarCHF"]) + penalty_cost)
            updated["TimePT_scaled"] = updated["TimePT"] / 200.0
            updated["WaitingTimePT_scaled"] = updated["WaitingTimePT"] / 60.0
            updated["MarginalCostPT_scaled"] = updated["MarginalCostPT"] / 10.0
        else:
            updated["dominated_alternative_name"] = "CAR"
            updated["dominance_reason"] = "CAR is clearly worse than PT on the displayed burden attributes."
            updated["TimeCar"] = max(float(updated["TimeCar"]), float(updated["TimePT"]) + float(updated["WaitingTimePT"]) + penalty_time)
            updated["CostCarCHF"] = max(float(updated["CostCarCHF"]), float(updated["MarginalCostPT"]) + penalty_cost)
            updated["TimeCar_scaled"] = updated["TimeCar"] / 200.0
            updated["CostCarCHF_scaled"] = updated["CostCarCHF"] / 10.0
    else:
        updated["dominated_alternative_name"] = "PT"
        updated["dominance_reason"] = "PT is deliberately made much worse than the other displayed options."
        updated["TimePT"] = max(float(updated["TimePT"]), 90.0)
        updated["WaitingTimePT"] = max(float(updated["WaitingTimePT"]), 25.0)
        updated["MarginalCostPT"] = max(float(updated["MarginalCostPT"]), 12.0)
        updated["TimePT_scaled"] = updated["TimePT"] / 200.0
        updated["WaitingTimePT_scaled"] = updated["WaitingTimePT"] / 60.0
        updated["MarginalCostPT_scaled"] = updated["MarginalCostPT"] / 10.0
    return updated


def build_model_blocks(model_config: dict, profiles: pd.DataFrame, scenario_bank: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(CONFIG["master_seed"]) + 1000 * (1 + list(model["key"] for model in CONFIG["llm_models"]).index(model_config["key"])))
    total_respondents = int(CONFIG["n_respondents_per_model"])
    prompt_arms = list(CONFIG["survey_design"]["prompt_arms"])
    option_orders = list(CONFIG["survey_design"]["option_orders"])

    shuffled_profiles = profiles.sample(frac=1.0, random_state=int(CONFIG["master_seed"]) + len(model_config["key"])).reset_index(drop=True)
    if len(shuffled_profiles) < total_respondents:
        repeats = int(np.ceil(total_respondents / len(shuffled_profiles)))
        shuffled_profiles = pd.concat([shuffled_profiles] * repeats, ignore_index=True)
    selected_profiles = shuffled_profiles.head(total_respondents).copy().reset_index(drop=True)

    block_rows: list[dict] = []
    task_rows: list[dict] = []

    for block_index, (_, respondent_row) in enumerate(selected_profiles.iterrows(), start=1):
        respondent_id = f"{model_config['respondent_prefix']}{block_index:04d}"
        prompt_arm = str(rng.choice(prompt_arms))
        semantic_labels = prompt_arm == "semantic_arm"
        scenario_indices = rng.choice(len(scenario_bank), size=int(CONFIG["survey_design"]["n_core_tasks"]), replace=False)
        core_scenarios = scenario_bank.iloc[scenario_indices].reset_index(drop=True)

        block_payload = respondent_row.to_dict()
        block_payload.update(
            {
                "model_key": model_config["key"],
                "respondent_id": respondent_id,
                "prompt_arm": prompt_arm,
                "semantic_arm": int(semantic_labels),
                "block_complexity_mean": float(core_scenarios["complexity_score"].mean()),
                "n_core_tasks": int(len(core_scenarios)),
            }
        )
        block_rows.append(block_payload)

        core_rows: list[dict] = []
        for task_position, (_, scenario_row) in enumerate(core_scenarios.iterrows(), start=1):
            core_rows.append(
                build_task_row(
                    scenario_row,
                    respondent_row,
                    model_config["key"],
                    prompt_arm,
                    respondent_id,
                    task_position,
                    "core",
                    f"CORE_{task_position}",
                    "core",
                    None,
                    semantic_labels,
                    option_orders[0],
                )
            )
        task_rows.extend(core_rows)

        for anchor_index, task_position in enumerate([9, 10], start=1):
            anchor_row = dict(core_rows[anchor_index - 1])
            anchor_row["task_index"] = task_position
            anchor_row["task_role"] = "label_mask_twin"
            anchor_row["pair_id"] = f"LABEL_{anchor_index}"
            anchor_row["manipulation_type"] = "label_mask"
            anchor_row["anchor_task_index"] = anchor_index
            anchor_row["semantic_labels"] = int(not semantic_labels)
            task_rows.append(anchor_row)

        for anchor_index, task_position in zip([3, 4], [11, 12]):
            anchor_row = dict(core_rows[anchor_index - 1])
            alternative_orders = [order for order in option_orders if order != anchor_row["option_order"]]
            anchor_row["task_index"] = task_position
            anchor_row["task_role"] = "order_twin"
            anchor_row["pair_id"] = f"ORDER_{anchor_index - 2}"
            anchor_row["manipulation_type"] = "order_randomization"
            anchor_row["anchor_task_index"] = anchor_index
            anchor_row["option_order"] = str(rng.choice(alternative_orders))
            anchor_row["display_A_alt"] = anchor_row["option_order"].split("|")[0]
            anchor_row["display_B_alt"] = anchor_row["option_order"].split("|")[1]
            anchor_row["display_C_alt"] = anchor_row["option_order"].split("|")[2]
            task_rows.append(anchor_row)

        for anchor_index, task_position in zip([5, 6], [13, 14]):
            anchor_row = worsen_task(dict(core_rows[anchor_index - 1]), float(CONFIG["survey_design"]["monotonicity_multiplier"]))
            anchor_row["task_index"] = task_position
            anchor_row["task_role"] = "monotonicity"
            anchor_row["pair_id"] = f"MONO_{anchor_index - 4}"
            anchor_row["manipulation_type"] = "monotonicity"
            anchor_row["anchor_task_index"] = anchor_index
            task_rows.append(anchor_row)

        for anchor_index, task_position in zip([7, 8], [15, 16]):
            anchor_row = dominance_task(dict(core_rows[anchor_index - 1]), CONFIG["survey_design"])
            anchor_row["task_index"] = task_position
            anchor_row["task_role"] = "dominance"
            anchor_row["pair_id"] = f"DOM_{anchor_index - 6}"
            anchor_row["manipulation_type"] = "dominance"
            anchor_row["anchor_task_index"] = anchor_index
            task_rows.append(anchor_row)

    block_frame = pd.DataFrame(block_rows)
    task_frame = pd.DataFrame(task_rows).sort_values(["respondent_id", "task_index"]).reset_index(drop=True)
    return block_frame, task_frame


def main() -> None:
    args = parse_args()
    archive_experiment_config()
    ensure_dir(EXPERIMENT_DIR)
    ensure_dir(OUTPUT_DIR)
    source_human = pd.read_csv(SOURCE_DATA_DIR / "human_cleaned_wide.csv")
    source_profiles = pd.read_csv(SOURCE_DATA_DIR / "human_respondent_profiles.csv")

    scenario_bank = scenario_bank_from_human_data(source_human)
    profile_bank = source_profiles[PROFILE_COLUMNS].copy()

    scenario_bank.to_csv(experiment_artifact_path("scenario_bank.csv"), index=False)
    profile_bank.to_csv(experiment_artifact_path("respondent_profile_bank.csv"), index=False)

    model_config = llm_config_for(args.model_key)
    collection_dir = ensure_dir(OUTPUT_DIR)
    block_frame, task_frame = build_model_blocks(model_config, profile_bank, scenario_bank)
    block_frame.to_csv(experiment_artifact_path("block_assignments.csv"), index=False)
    task_frame.to_csv(experiment_artifact_path("panel_tasks.csv"), index=False)
    model_summaries = [
        {
            "model_key": model_config["key"],
            "n_blocks": int(len(block_frame)),
            "n_tasks": int(len(task_frame)),
            "semantic_arm_share": float(block_frame["semantic_arm"].mean()),
            "mean_block_complexity": float(block_frame["block_complexity_mean"].mean()),
        }
    ]
    for filename in [
        "raw_interactions.jsonl",
        "respondent_transcripts.json",
        "run_respondents.json",
        "ai_collection_summary.json",
    ]:
        target = collection_dir / filename
        if not target.exists():
            if filename.endswith(".jsonl"):
                target.write_text("", encoding="utf-8")
            elif filename.endswith(".json"):
                write_json(target, {})

    write_json(
        experiment_artifact_path("latent_regime_data_summary.json"),
        {
            "experiment_name": CONFIG["experiment_name"],
            "source_data_dir": str(SOURCE_DATA_DIR.relative_to(Path.cwd())) if SOURCE_DATA_DIR.is_relative_to(Path.cwd()) else str(SOURCE_DATA_DIR),
            "scenario_bank_rows": int(len(scenario_bank)),
            "profile_bank_rows": int(len(profile_bank)),
            "llm_models": model_summaries,
        },
    )
    print(
        f"[prepare_optima_latent_regime_data] model={model_config['key']} scenarios={len(scenario_bank)} "
        f"profiles={len(profile_bank)}"
    )


if __name__ == "__main__":
    main()
