from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import DATA_DIR, OUTPUT_DIR, read_json


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        table.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return table


def format_metric(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def write_run_summary(run_dir: Path) -> None:
    raw_path = run_dir / "raw_interactions.jsonl"
    parsed_path = run_dir / "parsed_choices.csv"
    model_summary_path = run_dir / "biogeme_mnl_model_summary.json"
    comparison_summary_path = run_dir / "ai_vs_human_summary.json"
    estimates_path = run_dir / "biogeme_mnl_estimates.csv"
    if not (raw_path.exists() and parsed_path.exists() and model_summary_path.exists() and comparison_summary_path.exists() and estimates_path.exists()):
        return

    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    parsed = pd.read_csv(parsed_path)
    model_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
    comparison_summary = json.loads(comparison_summary_path.read_text(encoding="utf-8"))
    estimates = pd.read_csv(estimates_path)

    valid_choice_rate = float(parsed["is_valid_choice"].mean()) if not parsed.empty else 0.0
    choice_counts = parsed["choice_name"].value_counts().to_dict() if not parsed.empty else {}
    choice_durations = [
        float(row["total_duration"]) / 1e9
        for row in raw_rows
        if row.get("phase") == "choice" and row.get("total_duration")
    ]
    grounding_rows = [row for row in raw_rows if row.get("phase") == "grounding"]
    grounding_parse_rate = (
        sum(1 for row in grounding_rows if row.get("parsed_ok")) / len(grounding_rows)
        if grounding_rows
        else 0.0
    )
    respondent_seconds: dict[int, float] = {}
    for row in raw_rows:
        if row.get("phase") != "choice" or not row.get("total_duration"):
            continue
        respondent_id = int(row["synthetic_respondent_id"])
        respondent_seconds[respondent_id] = respondent_seconds.get(respondent_id, 0.0) + (float(row["total_duration"]) / 1e9)

    p_column = "robust_p_value" if "robust_p_value" in estimates.columns else "p_value"
    significant = estimates.loc[estimates[p_column].fillna(1.0) <= 0.05, "parameter_name"].tolist()

    lines = [
        "# 实验结果摘要",
        "",
        "## 运行摘要",
        "",
        f"- synthetic respondents：`{model_summary['n_respondents']}`",
        f"- choice tasks per respondent：`9`",
        f"- 总 choices：`{model_summary['n_observations']}`",
        f"- valid choice rate：`{valid_choice_rate:.4f}`",
        f"- grounding parse rate：`{grounding_parse_rate:.4f}`",
        (
            f"- 平均每题调用时长：`{sum(choice_durations) / len(choice_durations):.2f}` 秒"
            if choice_durations
            else "- 平均每题调用时长：`n/a`"
        ),
        (
            f"- 平均每位 respondent 的 9 题总时长：`{sum(respondent_seconds.values()) / len(respondent_seconds):.2f}` 秒"
            if respondent_seconds
            else "- 平均每位 respondent 的 9 题总时长：`n/a`"
        ),
        "",
        "## 选择分布",
        "",
    ]
    for alternative_name in ["TRAIN", "SWISSMETRO", "CAR"]:
        if alternative_name in choice_counts:
            lines.append(f"- {alternative_name}：`{choice_counts[alternative_name]}`")

    lines.extend(
        [
            "",
            "## Biogeme MNL 拟合",
            "",
            f"- final loglikelihood：`{model_summary['final_loglikelihood']:.3f}`",
            f"- null loglikelihood：`{model_summary['null_loglikelihood']:.3f}`",
            f"- rho_square：`{model_summary['rho_square']:.4f}`",
            f"- 参数数：`{model_summary['n_parameters']}`",
            f"- 线程数：`{model_summary['number_of_threads']}`",
            f"- 5% 显著参数数：`{int((estimates[p_column].fillna(1.0) <= 0.05).sum())}`",
            "",
            "## AI 与人类对比",
            "",
            f"- 可比较参数数：`{comparison_summary['n_compared_parameters']}`",
            f"- 符号一致数：`{comparison_summary['n_sign_matches']}`",
            f"- sign match rate：`{comparison_summary['sign_match_rate']:.4f}`",
            f"- human time-cost ratio：`{comparison_summary['human_time_cost_ratio']:.4f}`",
            f"- AI time-cost ratio：`{comparison_summary['ai_time_cost_ratio']:.4f}`",
            f"- ratio difference：`{comparison_summary['time_cost_ratio_difference']:.4f}`",
            "",
            "## 5% 显著参数",
            "",
        ]
    )

    if significant:
        for parameter_name in significant:
            row = estimates.loc[estimates["parameter_name"] == parameter_name].iloc[0]
            se_col = "robust_std_error" if "robust_std_error" in estimates.columns else "std_error"
            z_col = "robust_z_value" if "robust_z_value" in estimates.columns else "z_value"
            lines.append(
                f"- `{parameter_name}`：estimate=`{row['estimate']:.4f}`，se=`{row[se_col]:.4f}`，z=`{row[z_col]:.3f}`，p=`{row[p_column]:.4g}`"
            )
    else:
        lines.append("- 没有 5% 显著参数。")

    lines.extend(
        [
            "",
            "## 复现实验标签",
            "",
            "- `replication_standard = public_reverse_engineering`",
            "- `design_provenance = public data reverse engineering`",
            "- `estimation_backend = biogeme`",
            "- `questionnaire_style = multi_turn_one_task_at_a_time`",
        ]
    )

    (run_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    aggregate_dir = OUTPUT_DIR / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(DATA_DIR / "experiment_config.json")
    design_spec = read_json(DATA_DIR / "swissmetro_design_spec.json")
    benchmark_targets = read_json(DATA_DIR / "pylogit_benchmark_targets.json")
    human_summary = json.loads((OUTPUT_DIR / "human_benchmark" / "biogeme_mnl_model_summary.json").read_text(encoding="utf-8"))
    human_estimates = pd.read_csv(OUTPUT_DIR / "human_benchmark" / "biogeme_mnl_estimates.csv")
    comparison = pd.read_csv(aggregate_dir / "human_vs_ai_coefficients.csv")
    shares = pd.read_csv(aggregate_dir / "human_vs_ai_choice_shares.csv")
    run_stability = pd.read_csv(aggregate_dir / "run_stability_summary.csv")
    aggregate_summary = json.loads((aggregate_dir / "human_vs_ai_summary.json").read_text(encoding="utf-8"))

    for run_id in range(1, int(config["n_runs"]) + 1):
        write_run_summary(OUTPUT_DIR / f"ai_run_{run_id:02d}")

    report_lines = [
        "# Swissmetro AI vs Human Report",
        "",
        "## Experiment Positioning",
        "",
        "- `replication_standard = public_reverse_engineering`",
        "- This line is a public reverse engineering of Swissmetro, not an exact historical DOE recovery.",
        f"- LLM model: `{config['backend']['model_name']}`",
        f"- AI runs completed: `{aggregate_summary['n_ai_runs_completed']}`",
        "",
        "## Human Benchmark Validation",
        "",
        f"- cleaned observations: `{benchmark_targets['expected_cleaned_observations']}`",
        f"- cleaned respondents: `{benchmark_targets['expected_cleaned_respondents']}`",
        f"- target final loglikelihood: `{benchmark_targets['expected_final_loglikelihood']}`",
        f"- Biogeme final loglikelihood: `{human_summary['final_loglikelihood']:.3f}`",
        f"- benchmark LL delta: `{human_summary['benchmark_loglikelihood_delta']:.6f}`",
        "",
    ]

    human_table = human_estimates[["parameter_name", "estimate", "std_error", "robust_std_error"]].copy()
    human_table["estimate"] = human_table["estimate"].map(lambda value: format_metric(value))
    human_table["std_error"] = human_table["std_error"].map(lambda value: format_metric(value))
    human_table["robust_std_error"] = human_table["robust_std_error"].map(lambda value: format_metric(value))
    report_lines.extend(["### Human MNL Estimates", ""])
    report_lines.extend(markdown_table(human_table, ["parameter_name", "estimate", "std_error", "robust_std_error"]))
    report_lines.extend(["", "## Reverse-Engineered Design", ""])
    report_lines.extend(
        [
            f"- design type: `{design_spec['design_type']}`",
            f"- classification: `{design_spec['specific_classification']}`",
            f"- tasks per respondent: `{design_spec['tasks_per_respondent']}`",
            f"- `TRAIN_HE` levels: `{design_spec['exact_discrete_levels']['TRAIN_HE']}`",
            f"- `SM_HE` levels: `{design_spec['exact_discrete_levels']['SM_HE']}`",
            f"- `SM_SEATS` levels: `{design_spec['exact_discrete_levels']['SM_SEATS']}`",
            "",
        ]
    )

    run_rows = []
    for item in aggregate_summary["ai_runs"]:
        run_rows.append(
            {
                "run_label": item["run_label"],
                "final_loglikelihood": f"{item['final_loglikelihood']:.3f}",
                "sign_match_rate": f"{item['sign_match_rate']:.4f}",
                "ai_time_cost_ratio": f"{item['ai_time_cost_ratio']:.4f}",
                "ratio_diff": f"{item['time_cost_ratio_difference']:.4f}",
            }
        )
    if run_rows:
        report_lines.extend(["## AI Run Summary", ""])
        report_lines.extend(markdown_table(pd.DataFrame(run_rows), ["run_label", "final_loglikelihood", "sign_match_rate", "ai_time_cost_ratio", "ratio_diff"]))
        report_lines.append("")

    report_lines.extend(["## Coefficient Comparison", ""])
    compact = comparison[["run_label", "parameter_name", "human_estimate", "ai_estimate", "difference_ai_minus_human", "sign_match"]].copy()
    compact["human_estimate"] = compact["human_estimate"].map(lambda value: format_metric(value))
    compact["ai_estimate"] = compact["ai_estimate"].map(lambda value: format_metric(value))
    compact["difference_ai_minus_human"] = compact["difference_ai_minus_human"].map(lambda value: format_metric(value))
    report_lines.extend(markdown_table(compact.head(12), ["run_label", "parameter_name", "human_estimate", "ai_estimate", "difference_ai_minus_human", "sign_match"]))
    report_lines.append("")

    overall_shares = shares.loc[shares["group_type"] == "overall", ["source_label", "alternative_name", "share", "count"]].copy()
    overall_shares["share"] = overall_shares["share"].map(lambda value: format_metric(value))
    report_lines.extend(["## Overall Choice Shares", ""])
    report_lines.extend(markdown_table(overall_shares, ["source_label", "alternative_name", "count", "share"]))
    report_lines.append("")

    subgroup = shares.loc[
        shares["group_type"].isin(["GA", "CAR_AV"]),
        ["source_label", "group_type", "group_value", "alternative_name", "share"],
    ].copy()
    subgroup["share"] = subgroup["share"].map(lambda value: format_metric(value))
    report_lines.extend(["## Subgroup Choice Shares", ""])
    report_lines.extend(markdown_table(subgroup.head(24), ["source_label", "group_type", "group_value", "alternative_name", "share"]))
    report_lines.append("")

    if not run_stability.empty:
        stability_view = run_stability[["parameter_name", "ai_mean_estimate", "ai_std_estimate", "ai_min_estimate", "ai_max_estimate"]].copy()
        for column in ["ai_mean_estimate", "ai_std_estimate", "ai_min_estimate", "ai_max_estimate"]:
            stability_view[column] = stability_view[column].map(lambda value: format_metric(value))
        report_lines.extend(["## Run-to-Run Stability", ""])
        report_lines.extend(markdown_table(stability_view, ["parameter_name", "ai_mean_estimate", "ai_std_estimate", "ai_min_estimate", "ai_max_estimate"]))
        report_lines.append("")

    report_lines.extend(
        [
            "## Main Reading",
            "",
            "- The human Biogeme benchmark is expected to match the pylogit notebook closely because it estimates the same four-parameter MNL with the same cleaning and scaling rules.",
            "- The AI side should be read as a behavioral simulation benchmark: synthetic personas, reconstructed panel families, and one-task-at-a-time prompting.",
            "- The key comparison objects are coefficient signs and magnitudes, mode shares, subgroup share structure by `GA` and `CAR_AV`, and the implied time-cost ratio.",
            "",
        ]
    )

    (aggregate_dir / "experiment_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[summary] wrote {aggregate_dir / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
