from __future__ import annotations

import math

import numpy as np
import pandas as pd

from optima_common import CONFIG, DATA_DIR, SOURCE_DATA_DIR, archive_experiment_config, ensure_dir, llm_models, write_json


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


def alternative_proxy(row: pd.Series) -> dict[str, float]:
    return {
        "PT": float(row["TimePT_scaled"] + row["WaitingTimePT_scaled"] + row["MarginalCostPT_scaled"]),
        "CAR": float(row["TimeCar_scaled"] + row["CostCarCHF_scaled"]) if int(row["CAR_AVAILABLE"]) == 1 else 1.0e6,
        "SLOW_MODES": float(row["distance_km_scaled"]),
    }


def scenario_bank_from_human(frame: pd.DataFrame) -> pd.DataFrame:
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
    profile_row: pd.Series,
    model_key: str,
    block_template_id: str,
    respondent_id: str,
    run_repeat: int,
    prompt_arm: str,
    prompt_family: str,
    task_index: int,
    task_role: str,
    pair_id: str,
    manipulation_type: str,
    anchor_task_index: int | None,
    semantic_labels: bool,
    option_order: str,
    paraphrase_variant: str,
) -> dict:
    return {
        "model_key": model_key,
        "respondent_id": respondent_id,
        "block_template_id": block_template_id,
        "run_repeat": int(run_repeat),
        "human_respondent_id": str(profile_row["respondent_id"]),
        "human_id": int(profile_row["human_id"]),
        "normalized_weight": float(profile_row["normalized_weight"]),
        "prompt_arm": prompt_arm,
        "semantic_arm": int(prompt_arm == "semantic_arm"),
        "prompt_family": prompt_family,
        "prompt_family_naturalistic": int(prompt_family == "naturalistic"),
        "task_index": int(task_index),
        "task_role": task_role,
        "pair_id": pair_id,
        "manipulation_type": manipulation_type,
        "anchor_task_index": anchor_task_index if anchor_task_index is not None else -1,
        "is_utility_equivalent_intervention": int(manipulation_type in {"paraphrase", "label_mask", "order_randomization"}),
        "semantic_labels": int(semantic_labels),
        "option_order": option_order,
        "display_A_alt": option_order.split("|")[0],
        "display_B_alt": option_order.split("|")[1],
        "display_C_alt": option_order.split("|")[2],
        "paraphrase_variant": paraphrase_variant,
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


def worsen_task(row: dict, multiplier: float) -> dict:
    updated = dict(row)
    proxy = alternative_proxy(pd.Series(updated))
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


def dominance_task(row: dict, survey_config: dict) -> dict:
    updated = dict(row)
    penalty_time = float(survey_config["dominance_time_penalty_min"])
    penalty_wait = float(survey_config["dominance_wait_penalty_min"])
    penalty_cost = float(survey_config["dominance_cost_penalty_chf"])
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


def build_model_data(model_config: dict, profiles: pd.DataFrame, scenario_bank: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    survey_config = CONFIG["survey_design"]
    model_index = [m["key"] for m in llm_models()].index(model_config["key"])
    rng = np.random.default_rng(int(CONFIG["master_seed"]) + 1000 * (model_index + 1))
    n_templates = int(CONFIG["n_block_templates_per_model"])
    n_repeats = int(CONFIG["n_repeats_per_template"])
    prompt_arms = list(survey_config["prompt_arms"])
    prompt_families = list(survey_config["prompt_families"])
    option_orders = list(survey_config["option_orders"])

    shuffled_profiles = profiles.sample(frac=1.0, random_state=int(CONFIG["master_seed"]) + len(model_config["key"])).reset_index(drop=True)
    if len(shuffled_profiles) < n_templates:
        repeats = int(math.ceil(n_templates / len(shuffled_profiles)))
        shuffled_profiles = pd.concat([shuffled_profiles] * repeats, ignore_index=True)
    selected_profiles = shuffled_profiles.head(n_templates).copy().reset_index(drop=True)

    block_rows: list[dict] = []
    task_rows: list[dict] = []

    for template_index, (_, profile_row) in enumerate(selected_profiles.iterrows(), start=1):
        block_template_id = f"{model_config['respondent_prefix']}T{template_index:04d}"
        prompt_arm = str(rng.choice(prompt_arms))
        prompt_family = str(rng.choice(prompt_families))
        semantic_labels = prompt_arm == "semantic_arm"
        scenario_indices = rng.choice(len(scenario_bank), size=int(survey_config["n_core_tasks"]), replace=False)
        core_scenarios = scenario_bank.iloc[scenario_indices].reset_index(drop=True)

        template_rows: list[dict] = []
        for task_position, (_, scenario_row) in enumerate(core_scenarios.iterrows(), start=1):
            template_rows.append(
                build_task_row(
                    scenario_row,
                    profile_row,
                    model_config["key"],
                    block_template_id,
                    "TEMPLATE",
                    0,
                    prompt_arm,
                    prompt_family,
                    task_position,
                    "core",
                    f"CORE_{task_position}",
                    "core",
                    None,
                    semantic_labels,
                    option_orders[0],
                    "default",
                )
            )

        paraphrase_map = {1: 7, 2: 8}
        for anchor_index, task_position in paraphrase_map.items():
            row = dict(template_rows[anchor_index - 1])
            row["task_index"] = task_position
            row["task_role"] = "paraphrase_twin"
            row["pair_id"] = f"PARA_{anchor_index}"
            row["manipulation_type"] = "paraphrase"
            row["anchor_task_index"] = anchor_index
            row["paraphrase_variant"] = "alternate"
            template_rows.append(row)

        label_map = {3: 9, 4: 10}
        for anchor_index, task_position in label_map.items():
            row = dict(template_rows[anchor_index - 1])
            row["task_index"] = task_position
            row["task_role"] = "label_mask_twin"
            row["pair_id"] = f"LABEL_{anchor_index - 2}"
            row["manipulation_type"] = "label_mask"
            row["anchor_task_index"] = anchor_index
            row["semantic_labels"] = int(not semantic_labels)
            template_rows.append(row)

        order_map = {5: 11, 6: 12}
        for anchor_index, task_position in order_map.items():
            row = dict(template_rows[anchor_index - 1])
            row["task_index"] = task_position
            row["task_role"] = "order_twin"
            row["pair_id"] = f"ORDER_{anchor_index - 4}"
            row["manipulation_type"] = "order_randomization"
            row["anchor_task_index"] = anchor_index
            order_candidates = [order for order in option_orders if order != row["option_order"]]
            row["option_order"] = str(rng.choice(order_candidates))
            row["display_A_alt"] = row["option_order"].split("|")[0]
            row["display_B_alt"] = row["option_order"].split("|")[1]
            row["display_C_alt"] = row["option_order"].split("|")[2]
            template_rows.append(row)

        mono_map = {1: 13, 2: 14}
        for anchor_index, task_position in mono_map.items():
            row = worsen_task(dict(template_rows[anchor_index - 1]), float(survey_config["monotonicity_multiplier"]))
            row["task_index"] = task_position
            row["task_role"] = "monotonicity"
            row["pair_id"] = f"MONO_{anchor_index}"
            row["manipulation_type"] = "monotonicity"
            row["anchor_task_index"] = anchor_index
            template_rows.append(row)

        dom_map = {3: 15, 4: 16}
        for anchor_index, task_position in dom_map.items():
            row = dominance_task(dict(template_rows[anchor_index - 1]), survey_config)
            row["task_index"] = task_position
            row["task_role"] = "dominance"
            row["pair_id"] = f"DOM_{anchor_index - 2}"
            row["manipulation_type"] = "dominance"
            row["anchor_task_index"] = anchor_index
            template_rows.append(row)

        template_rows = sorted(template_rows, key=lambda item: item["task_index"])
        template_complexities = [row["complexity_score"] for row in template_rows if row["task_role"] == "core"]

        for repeat_index in range(1, n_repeats + 1):
            respondent_id = f"{model_config['respondent_prefix']}{template_index:04d}_R{repeat_index}"
            block_row = profile_row.to_dict()
            block_row.update(
                {
                    "model_key": model_config["key"],
                    "respondent_id": respondent_id,
                    "block_template_id": block_template_id,
                    "run_repeat": int(repeat_index),
                    "prompt_arm": prompt_arm,
                    "semantic_arm": int(semantic_labels),
                    "prompt_family": prompt_family,
                    "prompt_family_naturalistic": int(prompt_family == "naturalistic"),
                    "block_complexity_mean": float(np.mean(template_complexities)),
                    "block_complexity_sd": float(np.std(template_complexities)),
                    "task_count": int(survey_config["total_tasks"]),
                    "core_task_count": int(survey_config["n_core_tasks"]),
                    "paraphrase_pair_count": int(survey_config["n_paraphrase_twins"]),
                    "label_pair_count": int(survey_config["n_label_mask_twins"]),
                    "order_pair_count": int(survey_config["n_order_twins"]),
                    "monotonicity_task_count": int(survey_config["n_monotonicity_tasks"]),
                    "dominance_task_count": int(survey_config["n_dominance_tasks"]),
                }
            )
            block_rows.append(block_row)
            for row in template_rows:
                updated = dict(row)
                updated["respondent_id"] = respondent_id
                updated["run_repeat"] = int(repeat_index)
                task_rows.append(updated)

    return pd.DataFrame(block_rows), pd.DataFrame(task_rows)


def main() -> None:
    archive_experiment_config()
    ensure_dir(DATA_DIR)
    source_human = pd.read_csv(SOURCE_DATA_DIR / "human_cleaned_wide.csv")
    source_profiles = pd.read_csv(SOURCE_DATA_DIR / "human_respondent_profiles.csv")

    scenario_bank = scenario_bank_from_human(source_human)
    profile_bank = source_profiles[PROFILE_COLUMNS].copy()
    scenario_bank.to_csv(DATA_DIR / "scenario_bank.csv", index=False)
    profile_bank.to_csv(DATA_DIR / "respondent_profile_bank.csv", index=False)

    pooled_blocks = []
    summaries = []
    for model_config in llm_models():
        collection_dir = ensure_dir(DATA_DIR / model_config["collection_subdir"])
        block_frame, task_frame = build_model_data(model_config, profile_bank, scenario_bank)
        block_frame.to_csv(DATA_DIR / f"block_assignments_{model_config['key']}.csv", index=False)
        task_frame.to_csv(DATA_DIR / f"panel_tasks_{model_config['key']}.csv", index=False)
        pooled_blocks.append(block_frame)
        summaries.append(
            {
                "model_key": model_config["key"],
                "collection_subdir": model_config["collection_subdir"],
                "n_block_templates": int(block_frame["block_template_id"].nunique()),
                "n_runs": int(len(block_frame)),
                "n_tasks": int(len(task_frame)),
                "semantic_arm_share": float(block_frame["semantic_arm"].mean()),
                "naturalistic_prompt_share": float(block_frame["prompt_family_naturalistic"].mean()),
                "mean_block_complexity": float(block_frame["block_complexity_mean"].mean()),
            }
        )
        for filename in [
            "persona_samples.csv",
            "parsed_attitudes.csv",
            "parsed_task_responses.csv",
            "ai_panel_long.csv",
            "ai_panel_block.csv",
            "raw_interactions.jsonl",
            "respondent_transcripts.json",
            "run_respondents.json",
        ]:
            target = collection_dir / filename
            if not target.exists():
                if filename.endswith(".jsonl"):
                    target.write_text("", encoding="utf-8")
                elif filename.endswith(".json"):
                    write_json(target, {})

    pd.concat(pooled_blocks, ignore_index=True).to_csv(DATA_DIR / "pooled_block_assignments.csv", index=False)
    write_json(
        DATA_DIR / "intervention_regime_data_summary.json",
        {
            "experiment_name": CONFIG["experiment_name"],
            "source_data_dir": str(SOURCE_DATA_DIR),
            "scenario_bank_rows": int(len(scenario_bank)),
            "profile_bank_rows": int(len(profile_bank)),
            "llm_models": summaries,
        },
    )
    total_runs = sum(item["n_runs"] for item in summaries)
    print(
        f"[prepare_optima_intervention_regime_data] scenarios={len(scenario_bank)} profiles={len(profile_bank)} total_runs={total_runs}"
    )


if __name__ == "__main__":
    main()
