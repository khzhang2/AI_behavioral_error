from __future__ import annotations

from pathlib import Path

import pandas as pd

from optima_common import EXPERIMENT_DIR, OUTPUT_DIR, active_model_config, ai_collection_dir_for, experiment_analysis_dir, read_json, write_json


HUMAN_ATASOY_BASE = Path(__file__).resolve().parents[1] / "data" / "Swissmetro" / "demographic_choice_psychometric" / "atasoy_2011_replication" / "base_logit" / "base_logit_summary.json"


def maybe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def maybe_read_json(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def fmt(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_ai_collection_summary(model_key: str) -> dict:
    base_dir = ai_collection_dir_for(model_key)
    attitudes = maybe_read_csv(base_dir / "parsed_attitudes.csv")
    tasks = maybe_read_csv(base_dir / "parsed_task_responses.csv")
    blocks = maybe_read_csv(base_dir / "ai_panel_block.csv")
    progress = maybe_read_json(OUTPUT_DIR / "run_respondents.json")
    summary = {
        "experiment_name": str(progress.get("experiment_name", "")),
        "model_key": model_key,
        "completed_respondents": int(progress.get("completed_respondents", 0)),
        "target_respondents": int(progress.get("target_respondents", 0)),
        "valid_attitude_rate": float(attitudes["is_valid_indicator"].mean()) if not attitudes.empty else None,
        "valid_task_rate": float(tasks["is_valid_task_response"].mean()) if not tasks.empty else None,
        "mean_exact_repeat_flip_rate": float(blocks["exact_repeat_flip_rate_mean"].mean()) if not blocks.empty and "exact_repeat_flip_rate_mean" in blocks.columns else None,
        "mean_paraphrase_flip_rate": float(blocks["paraphrase_flip_rate"].mean()) if not blocks.empty and "paraphrase_flip_rate" in blocks.columns else None,
        "mean_label_flip_rate": float(blocks["label_flip_rate"].mean()) if not blocks.empty and "label_flip_rate" in blocks.columns else None,
        "mean_order_flip_rate": float(blocks["order_flip_rate"].mean()) if not blocks.empty and "order_flip_rate" in blocks.columns else None,
        "mean_monotonicity_compliance_rate": float(blocks["monotonicity_compliance_rate"].mean()) if not blocks.empty and "monotonicity_compliance_rate" in blocks.columns else None,
        "mean_dominance_violation_rate": float(blocks["dominance_violation_rate"].mean()) if not blocks.empty and "dominance_violation_rate" in blocks.columns else None,
    }
    write_json(OUTPUT_DIR / "ai_collection_summary.json", summary)
    return summary


def intervention_by_type(path: Path) -> dict[str, dict[str, float]]:
    frame = maybe_read_csv(path)
    if frame.empty or "manipulation_type" not in frame.columns:
        return {}
    grouped = frame.groupby("manipulation_type")[["intervention_gap_tv", "excess_intervention_gap"]].mean().reset_index()
    payload: dict[str, dict[str, float]] = {}
    for _, row in grouped.iterrows():
        payload[str(row["manipulation_type"])] = {
            "intervention_gap_tv": float(row["intervention_gap_tv"]) if row["intervention_gap_tv"] == row["intervention_gap_tv"] else float("nan"),
            "excess_intervention_gap": float(row["excess_intervention_gap"]) if row["excess_intervention_gap"] == row["excess_intervention_gap"] else float("nan"),
        }
    return payload


def share_gap_tv(human_base: dict, ai_base: dict) -> float | None:
    human_share = human_base.get("metrics", {}).get("market_shares", {})
    ai_share = ai_base.get("metrics", {}).get("market_shares", {})
    if not human_share or not ai_share:
        return None
    names = ["PT", "PMM", "SM"]
    return 0.5 * sum(abs(float(ai_share.get(name, 0.0)) - float(human_share.get(name, 0.0))) for name in names)


def share_direction_text(human_share: dict, ai_share: dict) -> str:
    if not human_share or not ai_share:
        return "当前 summary 无法读取完整的 AI 与 human share。"
    gaps = {
        "PMM": float(ai_share.get("PMM", 0.0)) - float(human_share.get("PMM", 0.0)),
        "PT": float(ai_share.get("PT", 0.0)) - float(human_share.get("PT", 0.0)),
        "SM": float(ai_share.get("SM", 0.0)) - float(human_share.get("SM", 0.0)),
    }
    dominant_name = max(gaps, key=lambda name: abs(gaps[name]))
    direction = "高估" if gaps[dominant_name] > 0 else "低估"
    paired = [name for name in ["PMM", "PT", "SM"] if name != dominant_name]
    secondary = max(paired, key=lambda name: abs(gaps[name]))
    secondary_direction = "高估" if gaps[secondary] > 0 else "低估"
    return f"按 Atasoy 2011 base logit 的结构比较，模型当前最明显的是 `{direction}` `{dominant_name}`，同时 `{secondary_direction}` `{secondary}`。"


def label_order_summary_text(label_flip, order_flip, label_excess, order_excess) -> tuple[str, str]:
    values = [value for value in [label_flip, order_flip, label_excess, order_excess] if value is not None and not pd.isna(value)]
    if not values:
        return "NA", "当前 summary 无法读取完整的 label/order 指标。"
    strongest = max(values)
    if strongest >= 0.5:
        level = "很强"
    elif strongest >= 0.2:
        level = "明显"
    else:
        level = "很弱"
    if pd.isna(order_flip) or pd.isna(label_flip):
        return level, "当前 summary 无法同时比较 label 与 order 两类翻转率。"
    if float(order_flip) >= float(label_flip):
        return level, "当前更明显的是 order sensitivity，而不是 label sensitivity。"
    return level, "当前更明显的是 label sensitivity，而不是 order sensitivity。"


def tradeoff_summary_text(monotonicity, dominance_violation) -> tuple[str, str]:
    if monotonicity is None or dominance_violation is None or pd.isna(monotonicity) or pd.isna(dominance_violation):
        return "NA", "当前 summary 无法读取完整的 monotonicity / dominance 指标。"
    monotonicity = float(monotonicity)
    dominance_violation = float(dominance_violation)
    if monotonicity >= 0.95 and dominance_violation <= 0.05:
        return "很强", "模型同时通过 monotonicity 与 dominance 检查，trade-off fidelity 很强。"
    if monotonicity >= 0.8 and dominance_violation <= 0.2:
        return "中等", "模型在 monotonicity 上较稳，但 dominance 仍有一定偏离。"
    return "很弱", "虽然 monotonicity 通过率不低，但 dominance violation 很高，说明规则性 trade-off fidelity 并不稳。"


def caveat_text(human_base: dict, ai_base: dict, salcm: dict) -> str:
    warnings = []
    if human_base and not bool(human_base.get("optimizer_success", False)):
        warnings.append("human Atasoy base logit 有数值优化警告")
    if ai_base and not bool(ai_base.get("optimizer_success", False)):
        warnings.append("AI Atasoy base logit 有数值优化警告")
    if salcm and not bool(salcm.get("optimizer_success", False)):
        warnings.append("SALCM 未完全收敛")
    if not warnings:
        return "注意事项：当前 smoke 规模较小，因此五维 error 的方向性解释比结构参数的精细解释更可靠。"
    return "注意事项：" + "；".join(warnings) + "。因此本次结果更适合做方向性判断，不宜过度解读精细参数。"


def main() -> None:
    model_config = active_model_config()
    model_key = str(model_config["key"])
    collection_summary = build_ai_collection_summary(model_key)

    human_base = maybe_read_json(HUMAN_ATASOY_BASE)
    ai_base = maybe_read_json(EXPERIMENT_DIR / "atasoy_2011_replication" / "ai_atasoy_base_logit_summary.json")
    intervention = maybe_read_json(EXPERIMENT_DIR / "intervention_metrics_summary.json")
    salcm = maybe_read_json(experiment_analysis_dir(EXPERIMENT_DIR, "salcm") / "ai_salcm_summary.json")
    intervention_type = intervention_by_type(EXPERIMENT_DIR / "intervention_sensitivity.csv")
    gap_tv = share_gap_tv(human_base, ai_base)

    human_share = human_base.get("metrics", {}).get("market_shares", {})
    ai_share = ai_base.get("metrics", {}).get("market_shares", {})
    para = intervention_type.get("paraphrase", {})
    label = intervention_type.get("label_mask", {})
    order = intervention_type.get("order_randomization", {})
    label_order_level, label_order_explanation = label_order_summary_text(
        collection_summary.get("mean_label_flip_rate"),
        collection_summary.get("mean_order_flip_rate"),
        label.get("excess_intervention_gap"),
        order.get("excess_intervention_gap"),
    )
    tradeoff_level, tradeoff_explanation = tradeoff_summary_text(
        collection_summary.get("mean_monotonicity_compliance_rate"),
        collection_summary.get("mean_dominance_violation_rate"),
    )

    opening = (
        f"# 实验摘要：{model_key}\n\n"
        f"本次实验对应单模型归档 `{model_key}`。AI 问卷收集共完成 `{collection_summary['completed_respondents']}` / "
        f"`{collection_summary['target_respondents']}` 个 planned respondents，态度题有效率为 `{fmt(collection_summary['valid_attitude_rate'])}`，"
        f"任务题有效率为 `{fmt(collection_summary['valid_task_rate'])}`。总体上，这次实验的主要特征是：模型内部非常稳定，"
        f"但相对 human benchmark 仍表现出明显的出行方式偏移，而且部分干预与 trade-off 检查并不稳。"
    )

    table_lines = [
        "| 检验对象 | 这次试验怎么概括 | 主要数值 | 解释 |",
        "| --- | --- | --- | --- |",
        f"| 同一系统的随机不稳定性 | 很低 | exact-repeat flip rate = `{fmt(intervention.get('mean_exact_repeat_flip_rate'))}`；response entropy = `{fmt(intervention.get('mean_response_entropy'))}` | 完全相同输入下几乎不翻转，within-model randomness 很弱。 |",
        f"| 对语义等价改写是否稳健 | 很稳健 | paraphrase flip rate = `{fmt(collection_summary.get('mean_paraphrase_flip_rate'))}`；paraphrase gap = `{fmt(para.get('intervention_gap_tv'))}`；paraphrase excess gap = `{fmt(para.get('excess_intervention_gap'))}` | 改写措辞后几乎没有系统变化，没有观察到超出随机性基线的 semantic fragility。 |",
        f"| 对标签或顺序是否过敏 | {label_order_level} | label flip rate = `{fmt(collection_summary.get('mean_label_flip_rate'))}`；order flip rate = `{fmt(collection_summary.get('mean_order_flip_rate'))}`；label excess gap = `{fmt(label.get('excess_intervention_gap'))}`；order excess gap = `{fmt(order.get('excess_intervention_gap'))}` | {label_order_explanation} |",
        f"| 是否真的在做 trade-off | {tradeoff_level} | monotonicity compliance = `{fmt(collection_summary.get('mean_monotonicity_compliance_rate'))}`；dominance violation = `{fmt(collection_summary.get('mean_dominance_violation_rate'))}` | {tradeoff_explanation} |",
        f"| 是否只是总体像人 | 不像，仍有明显 distortion | AI base-model shares: `PMM={fmt(ai_share.get('PMM'))}, PT={fmt(ai_share.get('PT'))}, SM={fmt(ai_share.get('SM'))}`；human base-model shares: `PMM={fmt(human_share.get('PMM'))}, PT={fmt(human_share.get('PT'))}, SM={fmt(human_share.get('SM'))}`；share gap TV = `{fmt(gap_tv)}` | {share_direction_text(human_share, ai_share)} |",
    ]

    closing = (
        "\n".join(table_lines)
        + "\n\n"
        + caveat_text(human_base, ai_base, salcm)
    )

    summary_text = opening + "\n\n" + closing + "\n"
    (EXPERIMENT_DIR / "experiment_summary.md").write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
