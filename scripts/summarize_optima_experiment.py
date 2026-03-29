from __future__ import annotations

import pandas as pd

from optima_common import AI_COLLECTION_DIR, CONFIG, INDICATOR_NAMES, OUTPUT_DIR, read_json


AGGREGATE_DIR = OUTPUT_DIR / "aggregate"


def fmt(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def main() -> None:
    ai_indicators = pd.read_csv(AI_COLLECTION_DIR / "parsed_indicators.csv")
    ai_choice = pd.read_csv(AI_COLLECTION_DIR / "parsed_choice.csv")
    progress = read_json(AI_COLLECTION_DIR / "run_respondents.json")
    human_biogeme32 = read_json(OUTPUT_DIR / "human_biogeme_32" / "biogeme_hcm_summary.json")
    ai_biogeme32 = read_json(OUTPUT_DIR / "ai_biogeme_32" / "biogeme_hcm_summary.json")
    human_torch500 = read_json(OUTPUT_DIR / "human_torch_500" / "torch_hcm_summary.json")
    ai_torch500 = read_json(OUTPUT_DIR / "ai_torch_500" / "torch_hcm_summary.json")
    comparison_summary = read_json(AGGREGATE_DIR / "comparison_summary.json")
    choice_shares = pd.read_csv(AGGREGATE_DIR / "human_vs_ai_choice_shares.csv")
    indicator_distribution = pd.read_csv(AGGREGATE_DIR / "human_vs_ai_indicator_distributions.csv")

    valid_indicator_rate = float(ai_indicators["is_valid_indicator"].mean())
    valid_choice_rate = float(ai_choice["is_valid_choice"].mean())
    avg_indicator_duration = float(ai_indicators["duration_sec"].mean())
    avg_choice_duration = float(ai_choice["duration_sec"].mean())

    lines = [
        "# Optima Reduced Official-Style HCM Summary",
        "",
        "## Run",
        f"- `experiment_name`: `{CONFIG['experiment_name']}`",
        f"- `model`: `{CONFIG['llm']['model']}`",
        f"- `completed_respondents`: `{progress['completed_respondents']}` / `{progress['target_respondents']}`",
        "",
        "## AI Survey Quality",
        f"- `valid_indicator_rate`: `{valid_indicator_rate:.4f}`",
        f"- `valid_choice_rate`: `{valid_choice_rate:.4f}`",
        f"- `avg_indicator_duration_sec`: `{avg_indicator_duration:.2f}`",
        f"- `avg_choice_duration_sec`: `{avg_choice_duration:.2f}`",
        "",
        "## Biogeme 32",
        f"- `human_final_loglikelihood`: `{fmt(human_biogeme32.get('final_loglikelihood'), 3)}`",
        f"- `ai_final_loglikelihood`: `{fmt(ai_biogeme32.get('final_loglikelihood'), 3)}`",
        "",
        "## Torch 500",
        f"- `human_final_loglikelihood`: `{fmt(human_torch500.get('final_loglikelihood'), 3)}`",
        f"- `ai_final_loglikelihood`: `{fmt(ai_torch500.get('final_loglikelihood'), 3)}`",
        "",
        "## Torch32 vs Biogeme32 Alignment",
        f"- `human_sign_match_rate`: `{fmt(comparison_summary['human_torch32_vs_biogeme32']['sign_match_rate'])}`",
        f"- `ai_sign_match_rate`: `{fmt(comparison_summary['ai_torch32_vs_biogeme32']['sign_match_rate'])}`",
        f"- `human_same_point_diff_at_biogeme`: `{fmt(comparison_summary['same_point_objective_checks']['human']['same_point_diff_at_biogeme'], 3)}`",
        f"- `ai_same_point_diff_at_biogeme`: `{fmt(comparison_summary['same_point_objective_checks']['ai']['same_point_diff_at_biogeme'], 3)}`",
        "",
        "## Choice Shares",
    ]
    for _, row in choice_shares.iterrows():
        lines.append(
            f"- `choice_{int(row['choice_code'])}`: human=`{row['human_share']:.4f}`, ai=`{row['ai_share']:.4f}`, diff=`{row['difference']:.4f}`"
        )
    lines.extend(["", "## Indicator Means"])
    for _, row in indicator_distribution.iterrows():
        lines.append(
            f"- `{row['indicator_name']}`: human_mean=`{row['human_mean']:.3f}`, ai_mean=`{row['ai_mean']:.3f}`, tvd=`{row['total_variation_distance']:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- `questionnaire_template_file`: `scripts/questionnaire_template.py`",
            "- `grounding`: compact JSON",
            "- `conversation_style`: one growing conversation per respondent",
            "- `indicators`: " + ", ".join(INDICATOR_NAMES),
        ]
    )
    (AGGREGATE_DIR / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
