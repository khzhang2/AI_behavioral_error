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
    ai = pd.read_csv(OUTPUT_DIR / "mixed_choice_estimates.csv")

    comparison = human.merge(ai, on="parameter_name", how="left")
    comparison["difference_ai_minus_human"] = comparison["estimate"] - comparison["human_estimate"]
    comparison["sign_match"] = (
        (comparison["estimate"] > 0) == (comparison["human_estimate"] > 0)
    ).astype(int)
    comparison.to_csv(OUTPUT_DIR / "ai_vs_human_comparison.csv", index=False)

    plotted = comparison.dropna(subset=["estimate"]).copy()
    fig, ax = plt.subplots(figsize=(10, 10))
    positions = range(len(plotted))
    ax.barh([p + 0.2 for p in positions], plotted["human_estimate"], height=0.4, label="Human (paper transformed)", color="#4c78a8")
    ax.barh([p - 0.2 for p in positions], plotted["estimate"], height=0.4, label="AI (qwen3.5:9b)", color="#f58518")
    ax.set_yticks(list(positions))
    ax.set_yticklabels(plotted["parameter_name"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient estimate")
    ax.set_title("AI vs human mixed-logit coefficient comparison")
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
