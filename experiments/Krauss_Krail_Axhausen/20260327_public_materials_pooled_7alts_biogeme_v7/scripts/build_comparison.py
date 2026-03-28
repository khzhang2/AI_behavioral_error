from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXPERIMENT_DIR / "data"
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

    human = pd.read_csv(DATA_DIR / "human_table4_pooled_full.csv")
    ai = pd.read_csv(output_dir / "biogeme_mixed_estimates.csv")

    comparison = human.merge(ai, on="parameter_name", how="left")
    comparison["difference_ai_minus_human"] = comparison["estimate"] - comparison["human_estimate"]
    comparison["sign_match"] = (
        (comparison["estimate"] > 0) == (comparison["human_estimate"] > 0)
    ).astype(int)
    comparison.to_csv(output_dir / "ai_vs_human_comparison.csv", index=False)

    plotted = comparison.dropna(subset=["estimate"]).copy()
    fig, ax = plt.subplots(figsize=(10, 14))
    positions = range(len(plotted))
    ax.barh([p + 0.2 for p in positions], plotted["human_estimate"], height=0.4, label="Human (paper Table 4)", color="#4c78a8")
    ax.barh([p - 0.2 for p in positions], plotted["estimate"], height=0.4, label="AI (Biogeme)", color="#f58518")
    ax.set_yticks(list(positions))
    ax.set_yticklabels(plotted["parameter_name"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient estimate")
    ax.set_title("AI vs human pooled mixed-logit coefficient comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ai_vs_human_coefficients.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_compared_parameters": int(len(plotted)),
        "n_sign_matches": int(plotted["sign_match"].sum()),
        "sign_match_rate": float(plotted["sign_match"].mean()),
    }
    (output_dir / "ai_vs_human_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
