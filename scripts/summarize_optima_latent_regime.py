from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from optima_common import CONFIG, DATA_DIR, EXPERIMENT_DIR, ai_collection_dir_for, llm_models, total_variation_distance, write_json


OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
ANALYSIS_CONFIG = CONFIG.get("analysis_v2", {})
KAPPA = float(ANALYSIS_CONFIG.get("randomness_tolerance_kappa", 1.25))
PAIRED_INTERVENTIONS = list(ANALYSIS_CONFIG.get("paired_interventions", ["label_mask", "order_randomization"]))
EXACT_REPEAT_SIGNATURE_FIELDS = list(ANALYSIS_CONFIG.get("exact_repeat_signature_fields", []))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def available_model_configs() -> list[dict]:
    configs = []
    for model in llm_models():
        if (ai_collection_dir_for(model["key"]) / "ai_panel_long.csv").exists():
            configs.append(model)
    return configs


def collection_summary(model_key: str) -> dict:
    base_dir = ai_collection_dir_for(model_key)
    attitudes = pd.read_csv(base_dir / "parsed_attitudes.csv") if (base_dir / "parsed_attitudes.csv").exists() else pd.DataFrame()
    tasks = pd.read_csv(base_dir / "parsed_task_responses.csv") if (base_dir / "parsed_task_responses.csv").exists() else pd.DataFrame()
    blocks = pd.read_csv(base_dir / "ai_panel_block.csv") if (base_dir / "ai_panel_block.csv").exists() else pd.DataFrame()
    progress = read_json(base_dir / "run_respondents.json") if (base_dir / "run_respondents.json").exists() else {}
    summary = {
        "model_key": model_key,
        "completed_respondents": int(progress.get("completed_respondents", 0)),
        "target_respondents": int(progress.get("target_respondents", 0)),
        "valid_attitude_rate": float(attitudes["is_valid_indicator"].mean()) if not attitudes.empty else None,
        "valid_task_rate": float(tasks["is_valid_task_response"].mean()) if not tasks.empty else None,
        "mean_label_flip_rate": float(blocks["label_flip_rate"].mean()) if not blocks.empty else None,
        "mean_order_flip_rate": float(blocks["order_flip_rate"].mean()) if not blocks.empty else None,
        "mean_monotonicity_compliance_rate": float(blocks["monotonicity_compliance_rate"].mean()) if not blocks.empty else None,
        "mean_dominance_violation_rate": float(blocks["dominance_violation_rate"].mean()) if not blocks.empty else None,
        "mean_confidence": float(blocks["confidence_mean"].mean()) if not blocks.empty else None,
    }
    if model_key.startswith("qwen"):
        target_name = "qwen_ai_collection_summary"
    elif model_key.startswith("deepseek"):
        target_name = "deepseek_ai_collection_summary"
    else:
        target_name = f"{model_key}_ai_collection_summary"
    target_dir = OUTPUT_DIR / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    Path(target_dir / "collection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    clean_weights = weights.loc[mask].astype(float)
    total = float(clean_weights.sum())
    if total <= 1e-12:
        return np.nan
    return float(np.average(values.loc[mask].astype(float), weights=clean_weights))


def weighted_distribution(labels: pd.Series, weights: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"label": labels.astype(str), "weight": weights.astype(float)})
    grouped = frame.groupby("label", dropna=False)["weight"].sum()
    total = float(grouped.sum())
    if total <= 1e-12:
        return pd.Series(dtype=float)
    return grouped / total


def response_entropy(distribution: pd.Series) -> float:
    if distribution.empty:
        return np.nan
    probs = distribution.to_numpy(dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return np.nan
    return float(-(probs * np.log(probs)).sum())


def python_scalar(value):
    return value.item() if hasattr(value, "item") else value


def task_frame_for_model(model_key: str) -> pd.DataFrame:
    base_dir = ai_collection_dir_for(model_key)
    task_path = base_dir / "parsed_task_responses.csv"
    panel_path = DATA_DIR / f"panel_tasks_{model_key}.csv"
    persona_path = base_dir / "persona_samples.csv"
    if not task_path.exists() or not panel_path.exists():
        return pd.DataFrame()

    task_frame = pd.read_csv(task_path)
    panel_frame = pd.read_csv(panel_path)
    merge_cols = ["respondent_id", "task_index", "task_role", "pair_id", "manipulation_type", "anchor_task_index"]
    design_cols = merge_cols + [
        "human_id",
        "normalized_weight",
        "prompt_arm",
        "semantic_arm",
        "scenario_id",
        "semantic_labels",
        "option_order",
        "TimePT_scaled",
        "WaitingTimePT_scaled",
        "MarginalCostPT_scaled",
        "TimeCar_scaled",
        "CostCarCHF_scaled",
        "distance_km_scaled",
        "complexity_score",
        "target_alternative_name",
        "dominated_alternative_name",
        "dominance_reason",
    ]
    merged = task_frame.merge(panel_frame[design_cols], on=merge_cols, how="left", suffixes=("_response", "_design"))
    if persona_path.exists():
        persona_cols = ["respondent_id", "human_id", "normalized_weight", "prompt_arm", "semantic_arm", "block_complexity_mean"]
        merged = merged.merge(pd.read_csv(persona_path)[persona_cols], on="respondent_id", how="left", suffixes=("", "_persona"))
    else:
        merged["block_complexity_mean"] = np.nan

    for base_name in ["human_id", "normalized_weight", "prompt_arm", "semantic_arm", "target_alternative_name", "dominated_alternative_name"]:
        if base_name not in merged.columns:
            candidates = [column for column in merged.columns if column.startswith(base_name)]
            for candidate in candidates:
                merged[base_name] = merged[candidate]
                break
    if "semantic_labels" not in merged.columns:
        for candidate in ["semantic_labels_response", "semantic_labels_design"]:
            if candidate in merged.columns:
                merged["semantic_labels"] = merged[candidate]
                break
    if "option_order" not in merged.columns:
        for candidate in ["option_order_response", "option_order_design"]:
            if candidate in merged.columns:
                merged["option_order"] = merged[candidate]
                break

    merged["model_key"] = model_key
    merged["is_valid_task_response"] = merged["is_valid_task_response"].fillna(0).astype(int)
    return merged


def pooled_task_frame(model_configs: list[dict]) -> pd.DataFrame:
    frames = [task_frame_for_model(model["key"]) for model in model_configs]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def exact_repeat_group_rows(task_frame: pd.DataFrame, scope_key: str) -> pd.DataFrame:
    valid = task_frame.loc[task_frame["is_valid_task_response"] == 1].copy()
    if valid.empty:
        return pd.DataFrame()
    available_cols = [column for column in EXACT_REPEAT_SIGNATURE_FIELDS if column in valid.columns]
    if not available_cols:
        return pd.DataFrame()

    rows = []
    grouped = valid.groupby(available_cols, dropna=False)
    for signature, group in grouped:
        n_repeats = int(len(group))
        if n_repeats <= 1:
            continue
        counts = group["chosen_alternative_name"].value_counts().to_dict()
        pairwise_count = int(n_repeats * (n_repeats - 1) / 2)
        same_pairs = sum(int(count * (count - 1) / 2) for count in counts.values())
        signature_values = signature if isinstance(signature, tuple) else (signature,)
        rows.append(
            {
                "scope_key": scope_key,
                "repeat_signature": json.dumps({key: python_scalar(value) for key, value in zip(available_cols, signature_values)}, ensure_ascii=False),
                "n_repeats": n_repeats,
                "pairwise_comparisons": pairwise_count,
                "exact_repeat_flip_rate": float(1.0 - (same_pairs / max(pairwise_count, 1))),
                "response_entropy": response_entropy(pd.Series(counts, dtype=float) / max(n_repeats, 1)),
            }
        )
    return pd.DataFrame(rows)


def exact_repeat_summary_frame(task_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    summary_rows = []
    for scope_key, frame in task_frames.items():
        group_frame = exact_repeat_group_rows(frame, scope_key)
        if not group_frame.empty:
            group_rows.append(group_frame)
            summary_rows.append(
                {
                    "scope_key": scope_key,
                    "exact_repeat_available": True,
                    "n_repeat_groups": int(len(group_frame)),
                    "exact_repeat_flip_rate": weighted_mean(group_frame["exact_repeat_flip_rate"], group_frame["pairwise_comparisons"]),
                    "exact_repeat_response_entropy": weighted_mean(group_frame["response_entropy"], group_frame["pairwise_comparisons"]),
                }
            )
        else:
            summary_rows.append(
                {
                    "scope_key": scope_key,
                    "exact_repeat_available": False,
                    "n_repeat_groups": 0,
                    "exact_repeat_flip_rate": np.nan,
                    "exact_repeat_response_entropy": np.nan,
                }
            )
    group_frame = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame(
        columns=["scope_key", "repeat_signature", "n_repeats", "pairwise_comparisons", "exact_repeat_flip_rate", "response_entropy"]
    )
    summary_frame = pd.DataFrame(summary_rows)
    return group_frame, summary_frame


def paired_intervention_frame(task_frame: pd.DataFrame) -> pd.DataFrame:
    valid = task_frame.loc[task_frame["is_valid_task_response"] == 1].copy()
    if valid.empty:
        return pd.DataFrame()
    twin = valid.loc[valid["manipulation_type"].isin(PAIRED_INTERVENTIONS) & valid["anchor_task_index"].notna()].copy()
    if twin.empty:
        return pd.DataFrame()
    anchor = valid[
        [
            "respondent_id",
            "task_index",
            "chosen_alternative_name",
            "choice_code",
            "confidence",
            "top_attribute_1",
            "top_attribute_2",
            "normalized_weight",
        ]
    ].rename(
        columns={
            "task_index": "anchor_task_index",
            "chosen_alternative_name": "anchor_choice_name",
            "choice_code": "anchor_choice_code",
            "confidence": "anchor_confidence",
            "top_attribute_1": "anchor_top_attribute_1",
            "top_attribute_2": "anchor_top_attribute_2",
            "normalized_weight": "anchor_weight",
        }
    )
    merged = twin.merge(anchor, on=["respondent_id", "anchor_task_index"], how="left")
    merged["flip"] = (merged["chosen_alternative_name"] != merged["anchor_choice_name"]).astype(float)
    merged["confidence_shift"] = merged["confidence"].astype(float) - merged["anchor_confidence"].astype(float)
    return merged


def paired_effect_rows(pair_frame: pd.DataFrame, scope_key: str, randomness_row: pd.Series | None) -> list[dict]:
    rows = []
    if pair_frame.empty:
        return rows
    exact_repeat_available = bool(randomness_row["exact_repeat_available"]) if randomness_row is not None else False
    exact_repeat_flip = float(randomness_row["exact_repeat_flip_rate"]) if exact_repeat_available else np.nan
    exact_repeat_entropy = float(randomness_row["exact_repeat_response_entropy"]) if exact_repeat_available else np.nan

    for manipulation_type, group in pair_frame.groupby("manipulation_type", dropna=False):
        weights = group["normalized_weight"].fillna(1.0).astype(float)
        anchor_dist = weighted_distribution(group["anchor_choice_name"], weights)
        intervention_dist = weighted_distribution(group["chosen_alternative_name"], weights)
        tv_distance = total_variation_distance(anchor_dist, intervention_dist)
        flip_rate = weighted_mean(group["flip"], weights)
        confidence_shift = weighted_mean(group["confidence_shift"], weights)
        rows.append(
            {
                "scope_key": scope_key,
                "effect_name": str(manipulation_type),
                "effect_family": "paired_intervention",
                "n_observations": int(len(group)),
                "tv_distance": float(tv_distance),
                "flip_rate": float(flip_rate),
                "mean_confidence_shift": float(confidence_shift),
                "diagnostic_rate": np.nan,
                "exact_repeat_available": exact_repeat_available,
                "exact_repeat_flip_rate": exact_repeat_flip,
                "exact_repeat_response_entropy": exact_repeat_entropy,
                "kappa": KAPPA,
                "tv_excess_over_randomness": float(tv_distance - KAPPA * exact_repeat_flip) if exact_repeat_available else np.nan,
                "flip_excess_over_randomness": float(flip_rate - KAPPA * exact_repeat_flip) if exact_repeat_available else np.nan,
            }
        )
    return rows


def prompt_arm_effect_row(task_frame: pd.DataFrame, scope_key: str, randomness_row: pd.Series | None) -> dict | None:
    valid = task_frame.loc[(task_frame["is_valid_task_response"] == 1) & (task_frame["task_role"] == "core")].copy()
    if valid.empty or "prompt_arm" not in valid.columns:
        return None
    semantic = valid.loc[valid["prompt_arm"] == "semantic_arm"].copy()
    neutral = valid.loc[valid["prompt_arm"] == "neutral_arm"].copy()
    if semantic.empty or neutral.empty:
        return None

    semantic_dist = weighted_distribution(semantic["chosen_alternative_name"], semantic["normalized_weight"].fillna(1.0))
    neutral_dist = weighted_distribution(neutral["chosen_alternative_name"], neutral["normalized_weight"].fillna(1.0))
    exact_repeat_available = bool(randomness_row["exact_repeat_available"]) if randomness_row is not None else False
    exact_repeat_flip = float(randomness_row["exact_repeat_flip_rate"]) if exact_repeat_available else np.nan
    exact_repeat_entropy = float(randomness_row["exact_repeat_response_entropy"]) if exact_repeat_available else np.nan
    tv_distance = total_variation_distance(semantic_dist, neutral_dist)
    return {
        "scope_key": scope_key,
        "effect_name": "prompt_arm_semantic_vs_neutral",
        "effect_family": "between_block_intervention",
        "n_observations": int(len(valid)),
        "tv_distance": float(tv_distance),
        "flip_rate": np.nan,
        "mean_confidence_shift": np.nan,
        "diagnostic_rate": np.nan,
        "exact_repeat_available": exact_repeat_available,
        "exact_repeat_flip_rate": exact_repeat_flip,
        "exact_repeat_response_entropy": exact_repeat_entropy,
        "kappa": KAPPA,
        "tv_excess_over_randomness": float(tv_distance - KAPPA * exact_repeat_flip) if exact_repeat_available else np.nan,
        "flip_excess_over_randomness": np.nan,
    }


def diagnostic_effect_rows(task_frame: pd.DataFrame, scope_key: str) -> list[dict]:
    rows = []
    valid = task_frame.loc[task_frame["is_valid_task_response"] == 1].copy()
    if valid.empty:
        return rows

    monotonicity = valid.loc[(valid["task_role"] == "monotonicity") & valid["target_alternative_name"].notna()].copy()
    if not monotonicity.empty:
        monotonicity["compliance"] = (monotonicity["chosen_alternative_name"] == monotonicity["target_alternative_name"]).astype(float)
        rows.append(
            {
                "scope_key": scope_key,
                "effect_name": "monotonicity_diagnostic",
                "effect_family": "diagnostic_rate",
                "n_observations": int(len(monotonicity)),
                "tv_distance": np.nan,
                "flip_rate": np.nan,
                "mean_confidence_shift": np.nan,
                "diagnostic_rate": weighted_mean(monotonicity["compliance"], monotonicity["normalized_weight"].fillna(1.0)),
                "exact_repeat_available": False,
                "exact_repeat_flip_rate": np.nan,
                "exact_repeat_response_entropy": np.nan,
                "kappa": KAPPA,
                "tv_excess_over_randomness": np.nan,
                "flip_excess_over_randomness": np.nan,
            }
        )

    dominance = valid.loc[(valid["task_role"] == "dominance") & valid["dominated_alternative_name"].notna()].copy()
    if not dominance.empty:
        dominance["violation"] = (dominance["chosen_alternative_name"] == dominance["dominated_alternative_name"]).astype(float)
        rows.append(
            {
                "scope_key": scope_key,
                "effect_name": "dominance_diagnostic",
                "effect_family": "diagnostic_rate",
                "n_observations": int(len(dominance)),
                "tv_distance": np.nan,
                "flip_rate": np.nan,
                "mean_confidence_shift": np.nan,
                "diagnostic_rate": weighted_mean(dominance["violation"], dominance["normalized_weight"].fillna(1.0)),
                "exact_repeat_available": False,
                "exact_repeat_flip_rate": np.nan,
                "exact_repeat_response_entropy": np.nan,
                "kappa": KAPPA,
                "tv_excess_over_randomness": np.nan,
                "flip_excess_over_randomness": np.nan,
            }
        )
    return rows


def intervention_effects_frame(task_frames: dict[str, pd.DataFrame], repeat_summary: pd.DataFrame) -> pd.DataFrame:
    repeat_lookup = repeat_summary.set_index("scope_key") if not repeat_summary.empty else pd.DataFrame()
    rows = []
    for scope_key, frame in task_frames.items():
        randomness_row = repeat_lookup.loc[scope_key] if scope_key in repeat_lookup.index else None
        pair_frame = paired_intervention_frame(frame)
        rows.extend(paired_effect_rows(pair_frame, scope_key, randomness_row))
        prompt_row = prompt_arm_effect_row(frame, scope_key, randomness_row)
        if prompt_row is not None:
            rows.append(prompt_row)
        rows.extend(diagnostic_effect_rows(frame, scope_key))
    return pd.DataFrame(rows)


def posterior_long_frame(path: Path) -> pd.DataFrame:
    posterior = pd.read_csv(path)
    value_cols = [column for column in posterior.columns if column.startswith("posterior_")]
    long = posterior.melt(id_vars=["respondent_id"], value_vars=value_cols, var_name="state_id", value_name="posterior_weight")
    extracted = long["state_id"].str.extract(r"posterior_c(?P<preference_class>\d+)_s(?P<scale_class>\d+)")
    long["preference_class"] = extracted["preference_class"].astype(int)
    long["scale_class"] = extracted["scale_class"].astype(int)
    long["state_id"] = "C" + long["preference_class"].astype(str) + "_S" + long["scale_class"].astype(str)
    return long


def state_reference_repeat(block_with_weights: pd.DataFrame, repeat_summary: pd.DataFrame) -> tuple[float, float, bool]:
    repeat_lookup = repeat_summary.set_index("scope_key") if not repeat_summary.empty else pd.DataFrame()
    if repeat_lookup.empty or "model_key" not in block_with_weights.columns:
        return np.nan, np.nan, False
    mass_by_model = block_with_weights.groupby("model_key")["posterior_weight"].sum()
    available = []
    for model_key, mass in mass_by_model.items():
        if model_key not in repeat_lookup.index:
            continue
        row = repeat_lookup.loc[model_key]
        if bool(row["exact_repeat_available"]):
            available.append((float(mass), float(row["exact_repeat_flip_rate"]), float(row["exact_repeat_response_entropy"])))
    if not available:
        return np.nan, np.nan, False
    total_mass = sum(mass for mass, _, _ in available)
    if total_mass <= 1e-12:
        return np.nan, np.nan, False
    flip = sum(mass * flip for mass, flip, _ in available) / total_mass
    entropy = sum(mass * ent for mass, _, ent in available) / total_mass
    return float(flip), float(entropy), True


def weighted_pair_metrics(pair_with_weights: pd.DataFrame, manipulation_type: str) -> tuple[float, float, float]:
    subset = pair_with_weights.loc[pair_with_weights["manipulation_type"] == manipulation_type].copy()
    if subset.empty:
        return np.nan, np.nan, np.nan
    weights = subset["posterior_weight"].fillna(0.0) * subset["normalized_weight"].fillna(1.0)
    tv_distance = total_variation_distance(
        weighted_distribution(subset["anchor_choice_name"], weights),
        weighted_distribution(subset["chosen_alternative_name"], weights),
    )
    flip_rate = weighted_mean(subset["flip"], weights)
    confidence_shift = weighted_mean(subset["confidence_shift"], weights)
    return float(tv_distance), float(flip_rate), float(confidence_shift)


def weighted_prompt_arm_tv(task_with_weights: pd.DataFrame) -> float:
    subset = task_with_weights.loc[(task_with_weights["task_role"] == "core") & (task_with_weights["is_valid_task_response"] == 1)].copy()
    if subset.empty:
        return np.nan
    semantic = subset.loc[subset["prompt_arm"] == "semantic_arm"].copy()
    neutral = subset.loc[subset["prompt_arm"] == "neutral_arm"].copy()
    if semantic.empty or neutral.empty:
        return np.nan
    semantic_weights = semantic["posterior_weight"].fillna(0.0) * semantic["normalized_weight"].fillna(1.0)
    neutral_weights = neutral["posterior_weight"].fillna(0.0) * neutral["normalized_weight"].fillna(1.0)
    return float(
        total_variation_distance(
            weighted_distribution(semantic["chosen_alternative_name"], semantic_weights),
            weighted_distribution(neutral["chosen_alternative_name"], neutral_weights),
        )
    )


def positive_part(value: float) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(max(float(value), 0.0))


def assign_v2_regime_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["regime_label_v2"] = "distorted_tradeoff"
    if frame.empty:
        return frame

    human_like_idx = frame["normalized_coefficient_distance"].idxmin()
    frame.loc[human_like_idx, "regime_label_v2"] = "human_like_coherent"

    remaining = frame.loc[frame.index != human_like_idx].copy()
    if not remaining.empty:
        label_score = (
            remaining["label_tv_distance"].fillna(remaining["label_flip_rate_v2"]).fillna(0.0)
            + remaining["prompt_arm_tv_distance"].fillna(0.0)
            - remaining["dominance_violation_rate"].fillna(0.0)
            + remaining["monotonicity_compliance_rate"].fillna(0.0)
        )
        label_idx = label_score.idxmax()
        frame.loc[label_idx, "regime_label_v2"] = "label_sensitive_coherent"

    remaining = frame.loc[frame["regime_label_v2"] == "distorted_tradeoff"].copy()
    if not remaining.empty:
        consistency_score = (
            (1.0 / np.clip(remaining["scale_value"].astype(float), 1e-6, None))
            + remaining["exact_repeat_flip_rate_reference"].fillna(0.0)
            + remaining["order_tv_distance"].fillna(remaining["order_flip_rate_v2"]).fillna(0.0)
            + (1.0 - remaining["monotonicity_compliance_rate"].fillna(1.0))
        )
        consistency_idx = consistency_score.idxmax()
        frame.loc[consistency_idx, "regime_label_v2"] = "low_consistency_distorted"
    return frame


def regime_v2_summary(
    regime_frame: pd.DataFrame,
    block_frame: pd.DataFrame,
    task_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    repeat_summary: pd.DataFrame,
) -> pd.DataFrame:
    posterior_long = posterior_long_frame(OUTPUT_DIR / "pooled_ai_salcm" / "salcm_posterior_membership.csv")
    block_with_weights = block_frame.merge(posterior_long, on="respondent_id", how="left")
    task_with_weights = task_frame.merge(posterior_long, on="respondent_id", how="left")
    pair_with_weights = pair_frame.merge(posterior_long, on="respondent_id", how="left")

    v2_rows = []
    for _, regime_row in regime_frame.iterrows():
        state_id = regime_row.get("state_id", f"C{int(regime_row['preference_class'])}_S{int(regime_row['scale_class'])}")
        state_blocks = block_with_weights.loc[block_with_weights["state_id"] == state_id].copy()
        state_tasks = task_with_weights.loc[task_with_weights["state_id"] == state_id].copy()
        state_pairs = pair_with_weights.loc[pair_with_weights["state_id"] == state_id].copy()

        label_tv, label_flip, _ = weighted_pair_metrics(state_pairs, "label_mask")
        order_tv, order_flip, _ = weighted_pair_metrics(state_pairs, "order_randomization")
        prompt_tv = weighted_prompt_arm_tv(state_tasks)
        repeat_flip, repeat_entropy, repeat_available = state_reference_repeat(state_blocks, repeat_summary)

        block_weights = state_blocks["posterior_weight"].fillna(0.0) * state_blocks["normalized_weight"].fillna(1.0)
        label_excess = label_tv - KAPPA * repeat_flip if repeat_available else np.nan
        order_excess = order_tv - KAPPA * repeat_flip if repeat_available else np.nan
        intervention_signature = (
            positive_part(label_excess if repeat_available else label_tv)
            + positive_part(order_excess if repeat_available else order_tv)
            + positive_part(prompt_tv)
            + float(regime_row["dominance_violation_rate"])
            + float(1.0 - regime_row["monotonicity_compliance_rate"])
        )

        v2_rows.append(
            {
                "state_id": state_id,
                "preference_class": int(regime_row["preference_class"]),
                "scale_class": int(regime_row["scale_class"]),
                "posterior_mass": float(regime_row["posterior_mass"]),
                "scale_value": float(regime_row["scale_value"]),
                "sign_mismatches": int(regime_row["sign_mismatches"]),
                "normalized_coefficient_distance": float(regime_row["normalized_coefficient_distance"]),
                "mode_share_deviation": float(regime_row["mode_share_deviation"]),
                "exact_repeat_available": bool(repeat_available),
                "exact_repeat_flip_rate_reference": repeat_flip,
                "exact_repeat_response_entropy_reference": repeat_entropy,
                "label_tv_distance": label_tv,
                "label_flip_rate_v2": label_flip,
                "label_tv_excess_over_randomness": label_excess,
                "order_tv_distance": order_tv,
                "order_flip_rate_v2": order_flip,
                "order_tv_excess_over_randomness": order_excess,
                "prompt_arm_tv_distance": prompt_tv,
                "monotonicity_compliance_rate": float(regime_row["monotonicity_compliance_rate"]),
                "dominance_violation_rate": float(regime_row["dominance_violation_rate"]),
                "confidence_mean": float(regime_row["confidence_mean"]),
                "top_attr_share_travel_time": weighted_mean(state_blocks["top_attr_share_travel_time"], block_weights),
                "top_attr_share_cost": weighted_mean(state_blocks["top_attr_share_cost"], block_weights),
                "top_attr_share_mode_label": weighted_mean(state_blocks["top_attr_share_mode_label"], block_weights),
                "intervention_signature_score": float(intervention_signature),
                "choice_share_PT": float(regime_row["choice_share_PT"]),
                "choice_share_CAR": float(regime_row["choice_share_CAR"]),
                "choice_share_SLOW_MODES": float(regime_row["choice_share_SLOW_MODES"]),
            }
        )

    return assign_v2_regime_labels(pd.DataFrame(v2_rows))


def block_distortion_v2(block_scores: pd.DataFrame, regime_v2: pd.DataFrame) -> pd.DataFrame:
    posterior_long = posterior_long_frame(OUTPUT_DIR / "pooled_ai_salcm" / "salcm_posterior_membership.csv")
    posterior_long = posterior_long.merge(
        regime_v2[
            [
                "state_id",
                "normalized_coefficient_distance",
                "mode_share_deviation",
                "intervention_signature_score",
                "dominance_violation_rate",
                "monotonicity_compliance_rate",
            ]
        ],
        on="state_id",
        how="left",
    )
    posterior_long["state_distortion_score_v2"] = (
        posterior_long["normalized_coefficient_distance"].fillna(0.0)
        + posterior_long["mode_share_deviation"].fillna(0.0)
        + posterior_long["intervention_signature_score"].fillna(0.0)
        + posterior_long["dominance_violation_rate"].fillna(0.0)
        + (1.0 - posterior_long["monotonicity_compliance_rate"].fillna(1.0))
    )
    block_v2 = posterior_long.groupby("respondent_id").apply(
        lambda frame: float((frame["posterior_weight"] * frame["state_distortion_score_v2"]).sum()),
        include_groups=False,
    ).reset_index(name="posterior_distortion_score_v2")
    return block_scores.merge(block_v2, on="respondent_id", how="left")


def main() -> None:
    model_configs = available_model_configs()
    human_mnl = read_json(OUTPUT_DIR / "human_baseline_mnl" / "mnl_summary.json")
    pooled_ai_mnl = read_json(OUTPUT_DIR / "pooled_ai_panel_mnl" / "mnl_summary.json")
    salcm_summary = read_json(OUTPUT_DIR / "pooled_ai_salcm" / "salcm_summary.json")
    regime_frame = pd.read_csv(OUTPUT_DIR / "pooled_ai_salcm" / "salcm_regime_summaries.csv")
    block_scores = pd.read_csv(OUTPUT_DIR / "pooled_ai_salcm" / "salcm_block_distortion_scores.csv")
    block_frame = pd.read_csv(OUTPUT_DIR / "pooled_ai_salcm" / "estimation_input_block.csv")

    task_frames = {model["key"]: task_frame_for_model(model["key"]) for model in model_configs}
    task_frames["pooled"] = pooled_task_frame(model_configs)
    pair_frames = {scope_key: paired_intervention_frame(frame) for scope_key, frame in task_frames.items()}

    collection_rows = [collection_summary(model["key"]) for model in model_configs]

    repeat_groups, repeat_summary = exact_repeat_summary_frame(task_frames)
    intervention_effects = intervention_effects_frame(task_frames, repeat_summary)
    regime_v2 = regime_v2_summary(regime_frame, block_frame, task_frames["pooled"], pair_frames["pooled"], repeat_summary)
    block_scores_v2 = block_distortion_v2(block_scores, regime_v2)

    regime_dir = OUTPUT_DIR / "regime_diagnostics"
    regime_dir.mkdir(parents=True, exist_ok=True)
    regime_frame.to_csv(regime_dir / "salcm_regime_summaries.csv", index=False)
    block_scores.to_csv(regime_dir / "salcm_block_distortion_scores.csv", index=False)
    repeat_groups.to_csv(regime_dir / "exact_repeat_group_metrics.csv", index=False)
    repeat_summary.to_csv(regime_dir / "exact_repeat_randomness_summary.csv", index=False)
    intervention_effects.to_csv(regime_dir / "intervention_effects_vs_randomness.csv", index=False)
    regime_v2.to_csv(regime_dir / "salcm_regime_v2_summaries.csv", index=False)
    block_scores_v2.to_csv(regime_dir / "salcm_block_distortion_scores_v2.csv", index=False)

    analysis_status = {
        "paired_interventions": PAIRED_INTERVENTIONS,
        "randomness_tolerance_kappa": KAPPA,
        "exact_repeat_choice_available_anywhere": bool(repeat_summary["exact_repeat_available"].any()) if not repeat_summary.empty else False,
        "exact_repeat_choice_available_pooled": bool(
            repeat_summary.loc[repeat_summary["scope_key"] == "pooled", "exact_repeat_available"].astype(bool).iloc[0]
        )
        if not repeat_summary.empty and (repeat_summary["scope_key"] == "pooled").any()
        else False,
        "paraphrase_intervention_available": False,
        "formal_h0_test_feasible_at_choice_level": bool(
            not repeat_summary.empty
            and (repeat_summary["scope_key"] == "pooled").any()
            and bool(repeat_summary.loc[repeat_summary["scope_key"] == "pooled", "exact_repeat_available"].astype(bool).iloc[0])
        ),
    }
    write_json(regime_dir / "analysis_identification_status.json", analysis_status)

    pooled_repeat = repeat_summary.loc[repeat_summary["scope_key"] == "pooled"].iloc[0] if not repeat_summary.empty and (repeat_summary["scope_key"] == "pooled").any() else None
    lines = [
        "# Optima Latent Response Regime Experiment Summary",
        "",
        "This archive records the current intervention-anchored latent response regime experiment built on the retained Optima benchmark. The analysis is layered. It first asks whether exact-repeat randomness is empirically identified, then asks whether the observed intervention effects exceed that randomness envelope, and only then interprets the scale-adjusted latent class choice model (SALCM) as a summary of latent response regimes rather than as a stand-alone mixture fit.",
        "",
        "## Data collection quality",
    ]
    for row in collection_rows:
        lines.extend(
            [
                f"### {row['model_key']}",
                f"The completed respondent count is `{row['completed_respondents']}` out of `{row['target_respondents']}`. "
                f"The valid-attitude rate is `{fmt(row['valid_attitude_rate'])}`, and the valid-task rate is `{fmt(row['valid_task_rate'])}`. "
                f"The mean label-flip rate is `{fmt(row['mean_label_flip_rate'])}`, the mean order-flip rate is `{fmt(row['mean_order_flip_rate'])}`, "
                f"the mean monotonicity-compliance rate is `{fmt(row['mean_monotonicity_compliance_rate'])}`, and the mean dominance-violation rate is `{fmt(row['mean_dominance_violation_rate'])}`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Human benchmark and pooled artificial-intelligence baseline",
            f"The human benchmark multinomial logit model uses `{human_mnl['n_respondents']}` respondents and `{human_mnl['n_tasks']}` tasks, with a final log likelihood of `{fmt(human_mnl['final_loglikelihood'], 3)}`. "
            f"The pooled artificial-intelligence panel multinomial logit model uses `{pooled_ai_mnl['n_respondents']}` respondents and `{pooled_ai_mnl['n_tasks']}` tasks, with a final log likelihood of `{fmt(pooled_ai_mnl['final_loglikelihood'], 3)}`.",
            "",
            "## Randomness envelope",
        ]
    )
    if pooled_repeat is not None and bool(pooled_repeat["exact_repeat_available"]):
        lines.append(
            f"The pooled data contain `{int(pooled_repeat['n_repeat_groups'])}` exact-repeat choice signatures. "
            f"The repeat flip rate is `{fmt(pooled_repeat['exact_repeat_flip_rate'])}`, and the repeat response entropy is `{fmt(pooled_repeat['exact_repeat_response_entropy'])}`."
        )
    else:
        lines.append(
            "The current archive does not contain literal exact-repeat choice signatures under fixed model, fixed persona, fixed prompt arm, and fixed task card. Therefore, the exact-repeat randomness envelope is not empirically identified at the choice level in this archive, and the formal null hypothesis comparing intervention effects to exact-repeat randomness cannot yet be tested directly."
        )

    lines.extend(["", "## Intervention effects", "The current analysis reports intervention effects for paired paraphrase tasks, paired label-mask tasks, paired order-randomization tasks, the between-block prompt-arm contrast, and the diagnostic monotonicity and dominance tasks."])
    pooled_effects = intervention_effects.loc[intervention_effects["scope_key"] == "pooled"].copy()
    for _, row in pooled_effects.iterrows():
        if row["effect_family"] == "paired_intervention":
            lines.append(
                f"The pooled `{row['effect_name']}` intervention yields a total-variation distance of `{fmt(row['tv_distance'])}` and a paired flip rate of `{fmt(row['flip_rate'])}`. The mean confidence shift is `{fmt(row['mean_confidence_shift'])}`."
            )
        elif row["effect_family"] == "between_block_intervention":
            lines.append(
                f"The pooled semantic-versus-neutral prompt-arm comparison yields a total-variation distance of `{fmt(row['tv_distance'])}` across core-task choice distributions."
            )
        elif row["effect_name"] == "monotonicity_diagnostic":
            lines.append(f"The pooled monotonicity-compliance rate is `{fmt(row['diagnostic_rate'])}`.")
        elif row["effect_name"] == "dominance_diagnostic":
            lines.append(f"The pooled dominance-violation rate is `{fmt(row['diagnostic_rate'])}`.")

    lines.extend(
        [
            "",
            "## Scale-adjusted latent class choice model",
            f"The pooled artificial-intelligence SALCM is estimated with `{salcm_summary['n_preference_classes']}` preference classes and `{salcm_summary['n_scale_classes']}` scale classes. "
            f"The final log likelihood is `{fmt(salcm_summary['final_loglikelihood'], 3)}`, and `{salcm_summary.get('n_nonempty_states', 'NA')}` states have non-negligible posterior mass.",
            "",
            "## Regime interpretation under the intervention-anchored framework",
        ]
    )
    for _, row in regime_v2.sort_values(["intervention_signature_score", "normalized_coefficient_distance"]).iterrows():
        lines.append(
            f"The regime `{row['regime_label_v2']}` corresponds to `{row['state_id']}` with posterior mass `{fmt(row['posterior_mass'])}`. "
            f"Its normalized coefficient distance from the human benchmark is `{fmt(row['normalized_coefficient_distance'])}`, and its mode-share deviation is `{fmt(row['mode_share_deviation'])}`. "
            f"The label-mask total-variation gap is `{fmt(row['label_tv_distance'])}`, the order-randomization total-variation gap is `{fmt(row['order_tv_distance'])}`, "
            f"the prompt-arm total-variation gap is `{fmt(row['prompt_arm_tv_distance'])}`, the monotonicity-compliance rate is `{fmt(row['monotonicity_compliance_rate'])}`, "
            f"the dominance-violation rate is `{fmt(row['dominance_violation_rate'])}`, and the composite intervention-signature score is `{fmt(row['intervention_signature_score'])}`."
        )

    lines.extend(
        [
            "",
            "## Distortion score",
            f"The original posterior distortion score has mean `{fmt(block_scores['posterior_distortion_score'].mean())}`, minimum `{fmt(block_scores['posterior_distortion_score'].min())}`, and maximum `{fmt(block_scores['posterior_distortion_score'].max())}`. "
            f"The v2 intervention-anchored distortion score has mean `{fmt(block_scores_v2['posterior_distortion_score_v2'].mean())}`, minimum `{fmt(block_scores_v2['posterior_distortion_score_v2'].min())}`, and maximum `{fmt(block_scores_v2['posterior_distortion_score_v2'].max())}`.",
        ]
    )

    (OUTPUT_DIR / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
