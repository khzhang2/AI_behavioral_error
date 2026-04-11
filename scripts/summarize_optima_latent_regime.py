from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from optima_common import CONFIG, DATA_DIR, EXPERIMENT_DIR, ai_collection_dir_for, llm_models


OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


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


def main() -> None:
    human_mnl = read_json(OUTPUT_DIR / "human_baseline_mnl" / "mnl_summary.json")
    pooled_ai_mnl = read_json(OUTPUT_DIR / "pooled_ai_panel_mnl" / "mnl_summary.json")
    salcm_summary = read_json(OUTPUT_DIR / "pooled_ai_salcm" / "salcm_summary.json")
    regime_frame = pd.read_csv(OUTPUT_DIR / "pooled_ai_salcm" / "salcm_regime_summaries.csv")
    block_scores = pd.read_csv(OUTPUT_DIR / "pooled_ai_salcm" / "salcm_block_distortion_scores.csv")

    collection_rows = [collection_summary(model["key"]) for model in llm_models()]

    regime_dir = OUTPUT_DIR / "regime_diagnostics"
    regime_dir.mkdir(parents=True, exist_ok=True)
    regime_frame.to_csv(regime_dir / "salcm_regime_summaries.csv", index=False)
    block_scores.to_csv(regime_dir / "salcm_block_distortion_scores.csv", index=False)

    lines = [
        "# Optima Latent Response Regime Experiment Summary",
        "",
        "This archive records the first latent response regime experiment built on the retained Optima benchmark. The experiment is artificial-intelligence-first: the human Optima data are used only to estimate a normative benchmark multinomial logit model and to seed respondent profiles and scenario attributes, whereas the repeated-task panel data are newly collected from the artificial-intelligence survey sessions.",
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
            "## Scale-adjusted latent class choice model",
            f"The pooled artificial-intelligence scale-adjusted latent class choice model is estimated with `{salcm_summary['n_preference_classes']}` preference classes and `{salcm_summary['n_scale_classes']}` scale classes. "
            f"The final log likelihood is `{fmt(salcm_summary['final_loglikelihood'], 3)}`. "
            f"The posterior probabilities sum to one up to the numerical range `{fmt(salcm_summary['posterior_probability_min'], 6)}` to `{fmt(salcm_summary['posterior_probability_max'], 6)}`.",
            "",
            "## Regime interpretation",
        ]
    )

    regime_sorted = regime_frame.sort_values(["normalized_coefficient_distance", "dominance_violation_rate"]).reset_index(drop=True)
    for _, row in regime_sorted.iterrows():
        lines.append(
            f"The regime `{row['regime_label']}` corresponds to preference class `{int(row['preference_class'])}` and scale class `{int(row['scale_class'])}`. "
            f"It has posterior mass `{fmt(row['posterior_mass'])}`, normalized coefficient distance `{fmt(row['normalized_coefficient_distance'])}`, "
            f"`{int(row['sign_mismatches'])}` sign mismatches relative to the human benchmark, mode-share deviation `{fmt(row['mode_share_deviation'])}`, "
            f"label-flip rate `{fmt(row['label_flip_rate'])}`, order-flip rate `{fmt(row['order_flip_rate'])}`, monotonicity-compliance rate `{fmt(row['monotonicity_compliance_rate'])}`, "
            f"and dominance-violation rate `{fmt(row['dominance_violation_rate'])}`."
        )
    lines.extend(
        [
            "",
            "## Distortion score",
            f"The posterior respondent-level distortion score has mean `{fmt(block_scores['posterior_distortion_score'].mean())}`, "
            f"minimum `{fmt(block_scores['posterior_distortion_score'].min())}`, and maximum `{fmt(block_scores['posterior_distortion_score'].max())}`.",
        ]
    )

    (OUTPUT_DIR / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
