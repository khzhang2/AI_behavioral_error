from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import CHOICE_CODE_TO_NAME, DATA_DIR, OUTPUT_DIR, write_json


def load_choice_share_frame(name: str) -> pd.DataFrame:
    if name == "human":
        frame = pd.read_csv(DATA_DIR / "human_cleaned_wide.csv")
    else:
        panels = pd.read_csv(OUTPUT_DIR / "reconstructed_panels_wide.csv")
        choices = pd.read_csv(OUTPUT_DIR / "parsed_choices.csv")
        choices = choices.loc[choices["is_valid_choice"] == 1, ["respondent_id", "task_id", "choice_code"]]
        frame = panels.merge(choices, on=["respondent_id", "task_id"], how="inner")
        frame["CHOICE"] = frame["choice_code"]
    shares = (
        frame["CHOICE"]
        .map(CHOICE_CODE_TO_NAME)
        .value_counts(normalize=True)
        .rename_axis("alternative_name")
        .reset_index(name=f"{name}_share")
    )
    return shares


def plot_coefficients(comparison: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axvline(0.0, color="gray", linewidth=1, linestyle="--")
    y_positions = range(len(comparison))
    ax.scatter(comparison["human_value"], list(y_positions), label="Human", color="#1f77b4")
    ax.scatter(comparison["ai_value"], list(y_positions), label="AI", color="#d62728")
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(comparison["parameter"])
    ax.set_xlabel("Coefficient value")
    ax.set_title("AI vs Human Coefficients")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def ratio_from_estimates(frame: pd.DataFrame) -> float | None:
    lookup = frame.set_index("parameter")["value"].to_dict()
    cost = lookup.get("B_COST")
    time = lookup.get("B_TIME")
    if cost in (None, 0):
        return None
    return float(time / cost)


def main() -> None:
    human = pd.read_csv(OUTPUT_DIR / "human_benchmark_biogeme_mnl_estimates.csv").rename(
        columns={
            "value": "human_value",
            "std_err": "human_std_err",
            "p_value": "human_p_value",
        }
    )
    ai = pd.read_csv(OUTPUT_DIR / "ai_biogeme_mnl_estimates.csv").rename(
        columns={
            "value": "ai_value",
            "std_err": "ai_std_err",
            "p_value": "ai_p_value",
        }
    )
    comparison = human.merge(ai, on="parameter", how="inner")
    comparison["human_sign"] = comparison["human_value"].apply(lambda x: 0 if x == 0 else (1 if x > 0 else -1))
    comparison["ai_sign"] = comparison["ai_value"].apply(lambda x: 0 if x == 0 else (1 if x > 0 else -1))
    comparison["sign_match"] = comparison["human_sign"] == comparison["ai_sign"]
    comparison["abs_diff"] = (comparison["ai_value"] - comparison["human_value"]).abs()
    comparison.to_csv(OUTPUT_DIR / "ai_vs_human_comparison.csv", index=False)

    plot_coefficients(comparison, OUTPUT_DIR / "ai_vs_human_coefficients.png")

    human_summary = pd.read_json(OUTPUT_DIR / "human_benchmark_biogeme_mnl_summary.json", typ="series")
    ai_summary = pd.read_json(OUTPUT_DIR / "ai_biogeme_mnl_summary.json", typ="series")

    choice_shares = load_choice_share_frame("human").merge(
        load_choice_share_frame("ai"),
        on="alternative_name",
        how="outer",
    ).fillna(0.0)
    choice_shares["share_diff"] = choice_shares["ai_share"] - choice_shares["human_share"]
    choice_shares.to_csv(OUTPUT_DIR / "ai_vs_human_choice_shares.csv", index=False)

    summary = {
        "n_compared_parameters": int(len(comparison)),
        "n_sign_matches": int(comparison["sign_match"].sum()),
        "sign_match_rate": float(comparison["sign_match"].mean()),
        "mean_abs_difference": float(comparison["abs_diff"].mean()),
        "human_time_cost_ratio": ratio_from_estimates(human.rename(columns={"human_value": "value"})),
        "ai_time_cost_ratio": ratio_from_estimates(ai.rename(columns={"ai_value": "value"})),
        "human_final_loglikelihood": float(human_summary["final_loglikelihood"]),
        "ai_final_loglikelihood": float(ai_summary["final_loglikelihood"]),
    }
    if summary["human_time_cost_ratio"] is not None and summary["ai_time_cost_ratio"] is not None:
        summary["time_cost_ratio_difference"] = summary["ai_time_cost_ratio"] - summary["human_time_cost_ratio"]
    write_json(OUTPUT_DIR / "ai_vs_human_summary.json", summary)


if __name__ == "__main__":
    main()
