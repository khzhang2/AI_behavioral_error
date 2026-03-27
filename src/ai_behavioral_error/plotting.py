from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_choice_shares(choice_frame: pd.DataFrame, output_dir: Path) -> None:
    share_frame = (
        choice_frame["chosen_alternative_name"]
        .value_counts(normalize=True)
        .rename_axis("alternative_name")
        .reset_index(name="share")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(share_frame["alternative_name"], share_frame["share"], color="#4c78a8")
    ax.set_ylabel("Choice share")
    ax.set_xlabel("Chosen alternative")
    ax.set_ylim(0, 1)
    ax.set_title("Synthetic respondent choice shares")
    fig.tight_layout()
    fig.savefig(output_dir / "choice_shares.png", dpi=150)
    plt.close(fig)


def plot_position_bias(choice_frame: pd.DataFrame, output_dir: Path) -> None:
    share_frame = (
        choice_frame["chosen_display_label"]
        .value_counts(normalize=True)
        .reindex(["A", "B", "C", "D"], fill_value=0.0)
        .rename_axis("display_label")
        .reset_index(name="share")
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(share_frame["display_label"], share_frame["share"], color="#f58518")
    ax.set_ylabel("Choice share")
    ax.set_xlabel("Displayed position")
    ax.set_ylim(0, 1)
    ax.set_title("Choice share by displayed label")
    fig.tight_layout()
    fig.savefig(output_dir / "position_bias.png", dpi=150)
    plt.close(fig)


def plot_mnl_coefficients(coefficient_frame: pd.DataFrame, output_dir: Path) -> None:
    ordered = coefficient_frame.sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered["feature"], ordered["coefficient"], color="#54a24b")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    ax.set_title("Conditional logit coefficients")
    fig.tight_layout()
    fig.savefig(output_dir / "mnl_coefficients.png", dpi=150)
    plt.close(fig)
