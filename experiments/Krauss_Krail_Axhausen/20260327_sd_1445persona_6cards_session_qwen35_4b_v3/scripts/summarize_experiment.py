from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def main() -> None:
    raw_rows = [json.loads(line) for line in open(OUTPUT_DIR / "raw_interactions.jsonl")]
    choices = pd.read_csv(OUTPUT_DIR / "parsed_choices.csv")
    model_summary = json.loads((OUTPUT_DIR / "mixed_choice_model_summary.json").read_text())
    comparison_summary = json.loads((OUTPUT_DIR / "ai_vs_human_summary.json").read_text())
    estimates = pd.read_csv(OUTPUT_DIR / "mixed_choice_estimates.csv")

    valid_choice_rate = float(choices["is_valid_choice"].mean())
    choice_shares = choices["chosen_alternative_name"].value_counts().to_dict()
    session_durations = [row["total_duration"] / 1e9 for row in raw_rows if row.get("total_duration")]
    significant = estimates.loc[estimates["significant_5pct"] == 1, "parameter_name"].tolist()

    lines = [
        "# 实验结果摘要",
        "",
        "## 运行摘要",
        "",
        f"- synthetic respondents：`{model_summary['n_respondents']}`",
        f"- choice tasks per respondent：`6`",
        f"- 总 choices：`{model_summary['n_observations']}`",
        f"- valid choice rate：`{valid_choice_rate:.4f}`",
        f"- 平均每个 session 时长：`{sum(session_durations) / len(session_durations):.2f}` 秒",
        f"- Sobol draws：`{model_summary['n_draws']}`",
        "",
        "## 选择分布",
        "",
    ]

    for name, count in choice_shares.items():
        lines.append(f"- {name}：`{count}`")

    lines.extend(
        [
            "",
            "## mixed logit 拟合",
            "",
            f"- final loglikelihood：`{model_summary['final_loglikelihood']:.3f}`",
            f"- rho_square：`{model_summary['rho_square']:.4f}`",
            f"- 参数数：`{model_summary['n_parameters']}`",
            f"- 5% 显著参数数：`{int(estimates['significant_5pct'].sum())}`",
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
            lines.append(
                f"- `{parameter_name}`：estimate=`{row['estimate']:.4f}`，se=`{row['std_error']:.4f}`，z=`{row['z_value']:.3f}`，p=`{row['p_value']:.4g}`"
            )
    else:
        lines.append("- 没有 5% 显著参数。")

    (OUTPUT_DIR / "experiment_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
