from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def main() -> None:
    human = pd.read_csv(DATA_DIR / "human_table4_sd_subset.csv")
    ai = pd.read_csv(OUTPUT_DIR / "biogeme_ai_estimates.csv")
    first_column = ai.columns[0]
    ai = ai.rename(columns={first_column: "parameter_name", "Value": "ai_estimate", "Rob. Std err": "ai_robust_std_error"})

    comparison = human.merge(ai[["parameter_name", "ai_estimate", "ai_robust_std_error"]], on="parameter_name", how="left")
    comparison["difference_ai_minus_human"] = comparison["ai_estimate"] - comparison["human_estimate"]
    comparison["sign_match"] = (
        (comparison["ai_estimate"] > 0) == (comparison["human_estimate"] > 0)
    ).astype(int)
    comparison.to_csv(OUTPUT_DIR / "ai_vs_human_comparison.csv", index=False)

    plotted = comparison.dropna(subset=["ai_estimate"]).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = range(len(plotted))
    ax.barh([p + 0.2 for p in positions], plotted["human_estimate"], height=0.4, label="Human (paper)", color="#4c78a8")
    ax.barh([p - 0.2 for p in positions], plotted["ai_estimate"], height=0.4, label="AI (qwen3.5:4b)", color="#f58518")
    ax.set_yticks(list(positions))
    ax.set_yticklabels(plotted["parameter_name"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient estimate")
    ax.set_title("AI vs human coefficient comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ai_vs_human_coefficients.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_compared_parameters": int(len(plotted)),
        "n_sign_matches": int(plotted["sign_match"].sum()),
        "sign_match_rate": float(plotted["sign_match"].mean()),
    }
    (OUTPUT_DIR / "ai_vs_human_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
