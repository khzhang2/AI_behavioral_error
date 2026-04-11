from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from optima_common import EXPERIMENT_DIR, ai_collection_dir_for, llm_models


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
        "mean_exact_repeat_flip_rate": float(blocks["exact_repeat_flip_rate_mean"].mean()) if not blocks.empty and "exact_repeat_flip_rate_mean" in blocks.columns else None,
        "mean_paraphrase_flip_rate": float(blocks["paraphrase_flip_rate"].mean()) if not blocks.empty and "paraphrase_flip_rate" in blocks.columns else None,
        "mean_label_flip_rate": float(blocks["label_flip_rate"].mean()) if not blocks.empty else None,
        "mean_order_flip_rate": float(blocks["order_flip_rate"].mean()) if not blocks.empty else None,
        "mean_monotonicity_compliance_rate": float(blocks["monotonicity_compliance_rate"].mean()) if not blocks.empty else None,
        "mean_dominance_violation_rate": float(blocks["dominance_violation_rate"].mean()) if not blocks.empty else None,
    }
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
    intervention_summary = read_json(OUTPUT_DIR / "intervention_diagnostics" / "intervention_metrics_summary.json")
    intervention_frame = pd.read_csv(OUTPUT_DIR / "intervention_diagnostics" / "intervention_sensitivity.csv")
    repeat_frame = pd.read_csv(OUTPUT_DIR / "intervention_diagnostics" / "exact_repeat_randomness.csv")

    collection_rows = [collection_summary(model["key"]) for model in llm_models()]
    regime_dir = OUTPUT_DIR / "regime_diagnostics"
    regime_dir.mkdir(parents=True, exist_ok=True)
    regime_frame.to_csv(regime_dir / "salcm_regime_summaries.csv", index=False)
    block_scores.to_csv(regime_dir / "salcm_block_distortion_scores.csv", index=False)

    lines = [
        "# Optima Intervention-Anchored Latent Response Regime Experiment Summary",
        "",
        "This archive records the second-stage intervention-anchored latent response regime experiment built on the retained Optima benchmark. The experiment first measures exact-repeat randomness, then evaluates whether controlled intervention effects exceed that stochastic envelope, and only then estimates a Scale-Adjusted Latent Class Choice Model (SALCM) on the pooled artificial-intelligence panel data. The human benchmark remains a normative comparison target rather than a participant in the latent-regime model.",
        "",
        "## Data collection quality",
    ]
    for row in collection_rows:
        lines.extend(
            [
                f"### {row['model_key']}",
                f"The completed respondent count is `{row['completed_respondents']}` out of `{row['target_respondents']}`. The valid-attitude rate is `{fmt(row['valid_attitude_rate'])}`, and the valid-task rate is `{fmt(row['valid_task_rate'])}`. The mean exact-repeat flip rate is `{fmt(row['mean_exact_repeat_flip_rate'])}`, the mean label-flip rate is `{fmt(row['mean_label_flip_rate'])}`, the mean order-flip rate is `{fmt(row['mean_order_flip_rate'])}`, the mean monotonicity-compliance rate is `{fmt(row['mean_monotonicity_compliance_rate'])}`, and the mean dominance-violation rate is `{fmt(row['mean_dominance_violation_rate'])}`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Randomness envelope and intervention effects",
            f"The mean exact-repeat flip rate is `{fmt(intervention_summary['mean_exact_repeat_flip_rate'])}`, and the mean response entropy is `{fmt(intervention_summary['mean_response_entropy'])}`. Across the utility-equivalent intervention pairs, the mean total-variation intervention gap is `{fmt(intervention_summary['mean_intervention_gap_tv'])}` and the mean excess intervention gap relative to the stochastic envelope is `{fmt(intervention_summary['mean_excess_intervention_gap'])}`.",
            f"The block-bootstrap test of the pure stochastic-instability null reports a 95% interval from `{fmt(intervention_summary['h0_test']['bootstrap_ci_lower'])}` to `{fmt(intervention_summary['h0_test']['bootstrap_ci_upper'])}` for the mean excess intervention gap. The null is rejected: `{intervention_summary['h0_test']['rejects_pure_randomness_h0']}`.",
            "",
            "## Human benchmark and pooled artificial-intelligence baseline",
            f"The human benchmark multinomial logit model uses `{human_mnl['n_respondents']}` respondents and `{human_mnl['n_tasks']}` tasks, with a final log likelihood of `{fmt(human_mnl['final_loglikelihood'], 3)}`. The pooled artificial-intelligence panel multinomial logit model uses `{pooled_ai_mnl['n_respondents']}` respondents and `{pooled_ai_mnl['n_tasks']}` tasks, with a final log likelihood of `{fmt(pooled_ai_mnl['final_loglikelihood'], 3)}`.",
            "",
            "## Scale-adjusted latent class choice model",
            f"The pooled artificial-intelligence SALCM is estimated with `{salcm_summary['n_preference_classes']}` preference classes and `{salcm_summary['n_scale_classes']}` scale classes. The final log likelihood is `{fmt(salcm_summary['final_loglikelihood'], 3)}`. The posterior probabilities sum to one up to the numerical range `{fmt(salcm_summary['posterior_probability_min'], 6)}` to `{fmt(salcm_summary['posterior_probability_max'], 6)}`.",
            "",
            "## Regime interpretation",
        ]
    )

    regime_sorted = regime_frame.sort_values(["normalized_coefficient_distance", "dominance_violation_rate"]).reset_index(drop=True)
    for _, row in regime_sorted.iterrows():
        lines.append(
            f"The regime `{row['regime_label']}` corresponds to preference class `{int(row['preference_class'])}` and scale class `{int(row['scale_class'])}`. It has posterior mass `{fmt(row['posterior_mass'])}`, normalized coefficient distance `{fmt(row['normalized_coefficient_distance'])}`, `{int(row['sign_mismatches'])}` sign mismatches relative to the human benchmark, mode-share deviation `{fmt(row['mode_share_deviation'])}`, label-flip rate `{fmt(row['label_flip_rate'])}`, order-flip rate `{fmt(row['order_flip_rate'])}`, monotonicity-compliance rate `{fmt(row['monotonicity_compliance_rate'])}`, and dominance-violation rate `{fmt(row['dominance_violation_rate'])}`."
        )

    if not intervention_frame.empty:
        manipulation_summary = intervention_frame.groupby("manipulation_type")["excess_intervention_gap"].mean().sort_values(ascending=False)
        lines.extend(["", "## Intervention signature", "The mean excess intervention gap by manipulation type is reported below."])
        for manipulation, value in manipulation_summary.items():
            lines.append(f"The manipulation `{manipulation}` has mean excess intervention gap `{fmt(value)}`.")

    lines.extend(
        [
            "",
            "## Distortion score",
            f"The posterior respondent-level distortion score has mean `{fmt(block_scores['posterior_distortion_score'].mean())}`, minimum `{fmt(block_scores['posterior_distortion_score'].min())}`, and maximum `{fmt(block_scores['posterior_distortion_score'].max())}`.",
            "",
            "## Identification logic",
            "This experiment treats artificial-intelligence response error as the combination of a stochastic envelope, intervention signatures that exceed that envelope, latent response regimes recovered by the SALCM, and distortions relative to the human benchmark. The latent classes are therefore interpreted after estimation as response regimes rather than pre-labeled error sources.",
        ]
    )

    (OUTPUT_DIR / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
