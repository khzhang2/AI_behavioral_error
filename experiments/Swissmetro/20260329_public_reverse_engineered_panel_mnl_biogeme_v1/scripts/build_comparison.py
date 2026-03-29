from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import CHOICE_CODE_TO_NAME, DATA_DIR, OUTPUT_DIR, read_json


def compute_share_rows(frame: pd.DataFrame, choice_col: str, source_label: str) -> list[dict]:
    rows: list[dict] = []

    def emit(group_type: str, group_value: str, subset: pd.DataFrame) -> None:
        counts = subset[choice_col].value_counts().sort_index().to_dict()
        total = int(len(subset))
        for code in [1, 2, 3]:
            count = int(counts.get(code, 0))
            rows.append(
                {
                    "source_label": source_label,
                    "group_type": group_type,
                    "group_value": group_value,
                    "alternative_code": code,
                    "alternative_name": CHOICE_CODE_TO_NAME[code],
                    "count": count,
                    "share": (count / total) if total else 0.0,
                    "n_observations": total,
                }
            )

    emit("overall", "all", frame)
    for ga_value, subset in frame.groupby("GA", sort=True):
        emit("GA", str(int(ga_value)), subset)
    for car_av_value, subset in frame.groupby("CAR_AV", sort=True):
        emit("CAR_AV", str(int(car_av_value)), subset)
    return rows


def plot_run_comparison(comparison: pd.DataFrame, output_path: Path, run_label: str) -> None:
    plotted = comparison.copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = range(len(plotted))
    ax.barh(
        [position + 0.2 for position in positions],
        plotted["human_estimate"],
        height=0.4,
        label="Human",
        color="#4c78a8",
    )
    ax.barh(
        [position - 0.2 for position in positions],
        plotted["ai_estimate"],
        height=0.4,
        label=run_label,
        color="#f58518",
    )
    ax.set_yticks(list(positions))
    ax.set_yticklabels(plotted["parameter_name"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient estimate")
    ax.set_title(f"{run_label} vs human MNL coefficients")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = read_json(DATA_DIR / "experiment_config.json")
    aggregate_dir = OUTPUT_DIR / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    human_estimates = pd.read_csv(OUTPUT_DIR / "human_benchmark" / "biogeme_mnl_estimates.csv")
    human_summary = json.loads((OUTPUT_DIR / "human_benchmark" / "biogeme_mnl_model_summary.json").read_text(encoding="utf-8"))
    human_wide = pd.read_csv(DATA_DIR / "human_cleaned_wide.csv")

    comparison_rows: list[pd.DataFrame] = []
    run_summaries: list[dict] = []
    share_rows = compute_share_rows(human_wide, "CHOICE", "human")
    stability_frames: list[pd.DataFrame] = []

    for run_id in range(1, int(config["n_runs"]) + 1):
        run_label = f"ai_run_{run_id:02d}"
        run_dir = OUTPUT_DIR / run_label
        ai_estimates_path = run_dir / "biogeme_mnl_estimates.csv"
        if not ai_estimates_path.exists():
            continue

        ai_estimates = pd.read_csv(ai_estimates_path)
        ai_summary = json.loads((run_dir / "biogeme_mnl_model_summary.json").read_text(encoding="utf-8"))
        ai_choices = pd.read_csv(run_dir / "parsed_choices.csv")
        ai_choices = ai_choices.rename(columns={"choice_code": "CHOICE", "ga": "GA", "car_av": "CAR_AV"})

        merged = human_estimates.merge(
            ai_estimates.rename(columns={"estimate": "ai_estimate"}),
            on="parameter_name",
            how="left",
            suffixes=("_human", "_ai"),
        )
        merged = merged.rename(columns={"estimate": "human_estimate"})
        merged["run_label"] = run_label
        merged["difference_ai_minus_human"] = merged["ai_estimate"] - merged["human_estimate"]
        merged["sign_match"] = ((merged["ai_estimate"] > 0) == (merged["human_estimate"] > 0)).astype(int)
        comparison_rows.append(merged)
        merged.to_csv(run_dir / "ai_vs_human_comparison.csv", index=False)
        plot_run_comparison(merged[["parameter_name", "human_estimate", "ai_estimate"]], run_dir / "ai_vs_human_comparison.png", run_label)

        share_rows.extend(compute_share_rows(ai_choices, "CHOICE", run_label))
        stability_frames.append(ai_estimates[["parameter_name", "estimate"]].rename(columns={"estimate": run_label}))

        ai_vot = abs(
            float(ai_estimates.loc[ai_estimates["parameter_name"] == "B_TIME", "estimate"].iloc[0])
            / float(ai_estimates.loc[ai_estimates["parameter_name"] == "B_COST", "estimate"].iloc[0])
        )
        human_vot = abs(
            float(human_estimates.loc[human_estimates["parameter_name"] == "B_TIME", "estimate"].iloc[0])
            / float(human_estimates.loc[human_estimates["parameter_name"] == "B_COST", "estimate"].iloc[0])
        )
        run_summaries.append(
            {
                "run_label": run_label,
                "final_loglikelihood": float(ai_summary["final_loglikelihood"]),
                "sign_match_rate": float(merged["sign_match"].mean()),
                "n_sign_matches": int(merged["sign_match"].sum()),
                "n_compared_parameters": int(len(merged)),
                "human_time_cost_ratio": human_vot,
                "ai_time_cost_ratio": ai_vot,
                "time_cost_ratio_difference": ai_vot - human_vot,
            }
        )
        (run_dir / "ai_vs_human_summary.json").write_text(
            json.dumps(
                {
                    "run_label": run_label,
                    "n_compared_parameters": int(len(merged)),
                    "n_sign_matches": int(merged["sign_match"].sum()),
                    "sign_match_rate": float(merged["sign_match"].mean()),
                    "human_time_cost_ratio": human_vot,
                    "ai_time_cost_ratio": ai_vot,
                    "time_cost_ratio_difference": ai_vot - human_vot,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    coefficient_comparison = pd.concat(comparison_rows, ignore_index=True)
    coefficient_comparison.to_csv(aggregate_dir / "human_vs_ai_coefficients.csv", index=False)

    choice_share_frame = pd.DataFrame(share_rows)
    choice_share_frame.to_csv(aggregate_dir / "human_vs_ai_choice_shares.csv", index=False)

    if stability_frames:
        stability = stability_frames[0]
        for frame in stability_frames[1:]:
            stability = stability.merge(frame, on="parameter_name", how="outer")
        value_columns = [column for column in stability.columns if column != "parameter_name"]
        stability["ai_mean_estimate"] = stability[value_columns].mean(axis=1)
        stability["ai_std_estimate"] = stability[value_columns].std(axis=1)
        stability["ai_min_estimate"] = stability[value_columns].min(axis=1)
        stability["ai_max_estimate"] = stability[value_columns].max(axis=1)
        stability.to_csv(aggregate_dir / "run_stability_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["parameter_name"]).to_csv(aggregate_dir / "run_stability_summary.csv", index=False)

    human_vot = abs(
        float(human_estimates.loc[human_estimates["parameter_name"] == "B_TIME", "estimate"].iloc[0])
        / float(human_estimates.loc[human_estimates["parameter_name"] == "B_COST", "estimate"].iloc[0])
    )
    summary = {
        "human_benchmark_final_loglikelihood": float(human_summary["final_loglikelihood"]),
        "human_time_cost_ratio": human_vot,
        "ai_runs": run_summaries,
        "n_ai_runs_completed": len(run_summaries),
    }
    (aggregate_dir / "human_vs_ai_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
