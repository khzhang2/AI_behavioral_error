from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def resolve_output_dir(output_subdir: str | None) -> Path:
    if not output_subdir:
        return DEFAULT_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR / output_subdir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-subdir", type=str, default=None)
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_subdir)

    raw_rows = [json.loads(line) for line in open(DEFAULT_OUTPUT_DIR / "raw_interactions.jsonl", encoding="utf-8") if line.strip()]
    choices = pd.read_csv(DEFAULT_OUTPUT_DIR / "parsed_choices.csv")
    model_summary = json.loads((output_dir / "biogeme_mixed_model_summary.json").read_text(encoding="utf-8"))
    comparison_summary = json.loads((output_dir / "ai_vs_human_summary.json").read_text(encoding="utf-8"))
    estimates = pd.read_csv(output_dir / "biogeme_mixed_estimates.csv")

    valid_choice_rate = float(choices["is_valid_choice"].mean()) if not choices.empty else 0.0
    choice_shares = choices["chosen_alternative_name"].value_counts().to_dict() if not choices.empty else {}
    choice_call_durations = [
        row["total_duration"] / 1e9
        for row in raw_rows
        if row.get("phase") == "choice" and row.get("total_duration")
    ]
    respondent_session_durations: dict[int, float] = {}
    for row in raw_rows:
        if row.get("phase") != "choice" or not row.get("total_duration"):
            continue
        respondent_id = int(row["respondent_id"])
        respondent_session_durations[respondent_id] = respondent_session_durations.get(respondent_id, 0.0) + (
            row["total_duration"] / 1e9
        )

    p_column = "robust_p_value" if "robust_p_value" in estimates.columns else "p_value"
    significant = estimates.loc[estimates[p_column] <= 0.05, "parameter_name"].tolist()

    lines = [
        "# 实验结果摘要",
        "",
        "## 运行摘要",
        "",
        f"- synthetic respondents：`{model_summary['n_respondents']}`",
        f"- choice tasks per respondent：`6`",
        f"- 总 choices：`{model_summary['n_observations']}`",
        f"- valid choice rate：`{valid_choice_rate:.4f}`",
        f"- 平均每题调用时长：`{sum(choice_call_durations) / len(choice_call_durations):.2f}` 秒" if choice_call_durations else "- 平均每题调用时长：`n/a`",
        (
            f"- 平均每位 respondent 的 6 题总时长：`{sum(respondent_session_durations.values()) / len(respondent_session_durations):.2f}` 秒"
            if respondent_session_durations
            else "- 平均每位 respondent 的 6 题总时长：`n/a`"
        ),
        f"- Biogeme draws：`{model_summary['n_draws']}`",
        "",
        "## 选择分布",
        "",
    ]

    for name, count in choice_shares.items():
        lines.append(f"- {name}：`{count}`")

    lines.extend(
        [
            "",
            "## Biogeme mixed logit 拟合",
            "",
            f"- final loglikelihood：`{model_summary['final_loglikelihood']:.3f}`",
            f"- rho_square：`{model_summary['rho_square']:.4f}`" if model_summary["rho_square"] is not None else "- rho_square：`n/a`",
            f"- 参数数：`{model_summary['n_parameters']}`",
            f"- 5% 显著参数数：`{int((estimates[p_column] <= 0.05).sum())}`",
            "",
            "## AI 与人类对比",
            "",
            f"- 可比较参数数：`{comparison_summary['n_compared_parameters']}`",
            f"- 符号一致数：`{comparison_summary['n_sign_matches']}`",
            f"- sign match rate：`{comparison_summary['sign_match_rate']:.4f}`",
            "",
            "## 5% 显著参数",
            "",
        ]
    )

    if significant:
        for parameter_name in significant:
            row = estimates.loc[estimates["parameter_name"] == parameter_name].iloc[0]
            std_col = "robust_std_error" if "robust_std_error" in row else "std_error"
            z_col = "robust_z_value" if "robust_z_value" in row else "z_value"
            lines.append(
                f"- `{parameter_name}`：estimate=`{row['estimate']:.4f}`，se=`{row[std_col]:.4f}`，z=`{row[z_col]:.3f}`，p=`{row[p_column]:.4g}`"
            )
    else:
        lines.append("- 没有 5% 显著参数。")

    lines.extend(
        [
            "",
            "## 复现实验标签",
            "",
            "- `replication_standard = public_materials_high_fidelity`",
            "- `design_provenance = public_exact attribute levels + inferred_from_public block combinations`",
            "- `estimation_backend = biogeme`",
        ]
    )

    (output_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
