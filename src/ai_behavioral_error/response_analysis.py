from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ai_behavioral_error.io import write_csv, write_json


def _mention_count(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(int(keyword in lowered) for keyword in keywords)


def _thinking_mentions_final_label(row: pd.Series) -> int:
    label = str(row["chosen_display_label"]).strip().lower()
    if not label:
        return 0
    thinking = str(row["thinking_text"]).lower()
    return int(f"{label}." in thinking or f"choice\": \"{label.upper()}\"" in thinking)


def analyze_response_composition(interaction_frame: pd.DataFrame, choice_frame: pd.DataFrame, output_dir: Path) -> dict:
    merged = interaction_frame.merge(
        choice_frame[["respondent_id", "repeat_id", "task_id", "chosen_alternative_name", "chosen_display_label"]],
        on=["respondent_id", "repeat_id", "task_id"],
        how="left",
    )

    merged["thinking_char_count"] = merged["thinking_text"].fillna("").str.len()
    merged["response_char_count"] = merged["response_text"].fillna("").str.len()
    merged["thinking_word_count"] = merged["thinking_text"].fillna("").str.split().str.len()
    merged["response_word_count"] = merged["response_text"].fillna("").str.split().str.len()
    merged["has_thinking"] = (merged["thinking_char_count"] > 0).astype(int)
    merged["thinking_mentions_multiple_alternatives"] = merged["thinking_text"].fillna("").apply(
        lambda text: int(
            _mention_count(
                text=text,
                keywords=["e-scooter", "bike sharing", "walking", "private car"],
            ) >= 2
        )
    )
    merged["thinking_mentions_final_label"] = merged.apply(_thinking_mentions_final_label, axis=1)

    summary = {
        "n_interactions": int(len(merged)),
        "has_thinking_rate": float(merged["has_thinking"].mean()),
        "mean_thinking_char_count": float(merged["thinking_char_count"].mean()),
        "median_thinking_char_count": float(merged["thinking_char_count"].median()),
        "mean_response_char_count": float(merged["response_char_count"].mean()),
        "median_response_char_count": float(merged["response_char_count"].median()),
        "mean_prompt_eval_count": float(merged["prompt_eval_count"].mean()),
        "mean_eval_count": float(merged["eval_count"].mean()),
        "mean_total_duration_sec": float(merged["total_duration"].mean() / 1_000_000_000),
        "thinking_mentions_multiple_alternatives_rate": float(merged["thinking_mentions_multiple_alternatives"].mean()),
        "thinking_mentions_final_label_rate": float(merged["thinking_mentions_final_label"].mean()),
    }

    write_csv(output_dir / "response_composition_rows.csv", merged)
    write_json(output_dir / "response_composition_summary.json", summary)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(merged["thinking_char_count"], merged["response_char_count"], color="#e45756")
    ax.set_xlabel("Thinking length (characters)")
    ax.set_ylabel("Response length (characters)")
    ax.set_title("Thinking vs response length")
    fig.tight_layout()
    fig.savefig(output_dir / "thinking_vs_response_chars.png", dpi=150)
    plt.close(fig)

    return summary
