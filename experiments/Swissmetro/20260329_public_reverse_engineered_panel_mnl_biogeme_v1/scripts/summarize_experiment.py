from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import OUTPUT_DIR, read_json


CONFIG = read_json(Path(__file__).resolve().parents[1] / "data" / "experiment_config.json")


def fmt(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def format_share_line(frame: pd.DataFrame, share_col: str) -> list[str]:
    ordered = frame.sort_values(share_col, ascending=False)
    return [
        f"- `{row['alternative_name']}`: {row[share_col]:.4f}"
        for _, row in ordered.iterrows()
    ]


def significant_parameters(estimates: pd.DataFrame) -> list[str]:
    if "p_value" not in estimates.columns:
        return []
    sig = estimates.loc[pd.to_numeric(estimates["p_value"], errors="coerce") < 0.05, "parameter"].tolist()
    return [f"- `{name}`" for name in sig]


def main() -> None:
    parsed = pd.read_csv(OUTPUT_DIR / "parsed_choices.csv")
    human_summary = read_json(OUTPUT_DIR / "human_benchmark_biogeme_mnl_summary.json")
    ai_summary = read_json(OUTPUT_DIR / "ai_biogeme_mnl_summary.json")
    comparison_summary = read_json(OUTPUT_DIR / "ai_vs_human_summary.json")
    shares = pd.read_csv(OUTPUT_DIR / "ai_vs_human_choice_shares.csv")
    ai_estimates = pd.read_csv(OUTPUT_DIR / "ai_biogeme_mnl_estimates.csv")
    progress = read_json(OUTPUT_DIR / "run_respondents.json")

    valid_choice_rate = float(parsed["is_valid_choice"].mean()) if not parsed.empty else 0.0
    grounding_rate = float(parsed["grounding_is_valid"].mean()) if not parsed.empty else 0.0
    avg_choice_duration = float(parsed["duration_sec"].mean()) if not parsed.empty else 0.0
    avg_total_duration = (
        float(parsed.groupby("respondent_id")["duration_sec"].sum().mean())
        if not parsed.empty
        else 0.0
    )

    lines = [
        "# Swissmetro Experiment Summary",
        "",
        "## Run",
        f"- `experiment_name`: `{CONFIG['experiment_name']}`",
        f"- `model`: `{CONFIG['backend']['model_name']}`",
        f"- `completed_respondents`: `{progress['completed_respondents']}` / `{progress['target_respondents']}`",
        f"- `tasks_per_respondent`: `{CONFIG['tasks_per_respondent']}`",
        "",
        "## Survey Quality",
        f"- `valid_choice_rate`: `{valid_choice_rate:.4f}`",
        f"- `grounding_parse_rate`: `{grounding_rate:.4f}`",
        f"- `avg_choice_duration_sec`: `{avg_choice_duration:.2f}`",
        f"- `avg_respondent_total_duration_sec`: `{avg_total_duration:.2f}`",
        "",
        "## Choice Shares",
        "Human:",
        *format_share_line(shares, "human_share"),
        "AI:",
        *format_share_line(shares, "ai_share"),
        "",
        "## Human Benchmark",
        f"- `final_loglikelihood`: `{fmt(human_summary.get('final_loglikelihood'), 3)}`",
        f"- `rho_square`: `{fmt(human_summary.get('rho_square'))}`",
        f"- `number_of_threads`: `{human_summary['number_of_threads']}`",
        "",
        "## AI MNL",
        f"- `final_loglikelihood`: `{fmt(ai_summary.get('final_loglikelihood'), 3)}`",
        f"- `rho_square`: `{fmt(ai_summary.get('rho_square'))}`",
        f"- `number_of_threads`: `{ai_summary['number_of_threads']}`",
        "",
        "## AI vs Human",
        f"- `sign_match_rate`: `{fmt(comparison_summary.get('sign_match_rate'))}`",
        f"- `mean_abs_difference`: `{fmt(comparison_summary.get('mean_abs_difference'))}`",
        f"- `human_time_cost_ratio`: `{fmt(comparison_summary.get('human_time_cost_ratio'))}`",
        f"- `ai_time_cost_ratio`: `{fmt(comparison_summary.get('ai_time_cost_ratio'))}`",
        "",
        "## Significant AI Parameters",
        *(significant_parameters(ai_estimates) or ["- none"]),
        "",
        "## Notes",
        "- `questionnaire_template_file`: `scripts/questionnaire_template.py`",
        "- `grounding`: compact JSON to avoid truncation",
        "- `conversation_style`: one growing multi-turn conversation per respondent",
        "- `estimator`: Biogeme 4-parameter MNL",
    ]
    (OUTPUT_DIR / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
