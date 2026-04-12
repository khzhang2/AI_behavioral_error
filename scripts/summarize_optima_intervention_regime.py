from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from optima_common import EXPERIMENT_DIR, OUTPUT_DIR, active_model_config, ai_collection_dir_for, read_json, write_json


def maybe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def maybe_read_json(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def fmt(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_ai_collection_summary(model_key: str) -> dict:
    base_dir = ai_collection_dir_for(model_key)
    attitudes = maybe_read_csv(base_dir / "parsed_attitudes.csv")
    tasks = maybe_read_csv(base_dir / "parsed_task_responses.csv")
    blocks = maybe_read_csv(base_dir / "ai_panel_block.csv")
    progress = maybe_read_json(OUTPUT_DIR / "run_respondents.json")
    summary = {
        "experiment_name": str(progress.get("experiment_name", "")),
        "model_key": model_key,
        "completed_respondents": int(progress.get("completed_respondents", 0)),
        "target_respondents": int(progress.get("target_respondents", 0)),
        "valid_attitude_rate": float(attitudes["is_valid_indicator"].mean()) if not attitudes.empty else None,
        "valid_task_rate": float(tasks["is_valid_task_response"].mean()) if not tasks.empty else None,
        "mean_exact_repeat_flip_rate": float(blocks["exact_repeat_flip_rate_mean"].mean()) if not blocks.empty and "exact_repeat_flip_rate_mean" in blocks.columns else None,
        "mean_label_flip_rate": float(blocks["label_flip_rate"].mean()) if not blocks.empty and "label_flip_rate" in blocks.columns else None,
        "mean_order_flip_rate": float(blocks["order_flip_rate"].mean()) if not blocks.empty and "order_flip_rate" in blocks.columns else None,
        "mean_monotonicity_compliance_rate": float(blocks["monotonicity_compliance_rate"].mean()) if not blocks.empty and "monotonicity_compliance_rate" in blocks.columns else None,
        "mean_dominance_violation_rate": float(blocks["dominance_violation_rate"].mean()) if not blocks.empty and "dominance_violation_rate" in blocks.columns else None,
    }
    write_json(OUTPUT_DIR / "ai_collection_summary.json", summary)
    return summary


def main() -> None:
    model_config = active_model_config()
    model_key = str(model_config["key"])
    collection_summary = build_ai_collection_summary(model_key)

    human_mnl = maybe_read_json(EXPERIMENT_DIR / "human_baseline_mnl_summary.json")
    ai_mnl = maybe_read_json(EXPERIMENT_DIR / "ai_panel_mnl_summary.json")
    intervention = maybe_read_json(EXPERIMENT_DIR / "intervention_metrics_summary.json")
    salcm = maybe_read_json(EXPERIMENT_DIR / "ai_salcm_summary.json")

    lines = [
        f"# Experiment Summary: {model_key}",
        "",
        f"This archive records one model only: `{model_key}`. The AI collection completed `{collection_summary['completed_respondents']}` of `{collection_summary['target_respondents']}` planned respondent runs. The valid-attitude rate is `{fmt(collection_summary['valid_attitude_rate'])}` and the valid-task rate is `{fmt(collection_summary['valid_task_rate'])}`.",
    ]

    if intervention:
        lines.append(
            f"Exact-repeat randomness is summarized by mean flip rate `{fmt(intervention.get('mean_exact_repeat_flip_rate'))}` and mean response entropy `{fmt(intervention.get('mean_response_entropy'))}`. The mean intervention gap is `{fmt(intervention.get('mean_intervention_gap_tv'))}`, and the mean excess intervention gap is `{fmt(intervention.get('mean_excess_intervention_gap'))}`."
        )

    if human_mnl and ai_mnl:
        lines.append(
            f"The human baseline MNL uses `{human_mnl.get('n_respondents', 'NA')}` respondents and `{human_mnl.get('n_tasks', 'NA')}` tasks, with final log likelihood `{fmt(human_mnl.get('final_loglikelihood'), 3)}`. The AI panel MNL uses `{ai_mnl.get('n_respondents', 'NA')}` respondents and `{ai_mnl.get('n_tasks', 'NA')}` tasks, with final log likelihood `{fmt(ai_mnl.get('final_loglikelihood'), 3)}`."
        )

    if salcm:
        lines.append(
            f"The SALCM is estimated with `{salcm.get('n_preference_classes', 'NA')}` preference classes and `{salcm.get('n_scale_classes', 'NA')}` scale classes. The final log likelihood is `{fmt(salcm.get('final_loglikelihood'), 3)}`, and the number of nonempty posterior states is `{salcm.get('n_nonempty_states', 'NA')}`."
        )

    lines.append(
        "Raw AI conversation records are stored only under `outputs/`. Derived AI panels, diagnostics, MNL estimates, SALCM estimates, and comparison results are stored directly in the experiment root."
    )

    summary_text = "\n\n".join(lines) + "\n"
    (EXPERIMENT_DIR / "experiment_summary.md").write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
