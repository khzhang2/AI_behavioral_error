from __future__ import annotations

from optima_common import EXPERIMENT_DIR, OUTPUT_DIR, active_model_config, ai_collection_dir_for, read_json, write_json

import pandas as pd


def maybe_read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def maybe_read_json(path):
    return read_json(path) if path.exists() else {}


def fmt(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def main() -> None:
    model_config = active_model_config()
    model_key = str(model_config["key"])
    base_dir = ai_collection_dir_for(model_key)

    attitudes = maybe_read_csv(base_dir / "parsed_attitudes.csv")
    tasks = maybe_read_csv(base_dir / "parsed_task_responses.csv")
    progress = maybe_read_json(OUTPUT_DIR / "run_respondents.json")
    collection_summary = {
        "experiment_name": str(progress.get("experiment_name", "")),
        "model_key": model_key,
        "completed_respondents": int(progress.get("completed_respondents", 0)),
        "target_respondents": int(progress.get("target_respondents", 0)),
        "valid_attitude_rate": float(attitudes["is_valid_indicator"].mean()) if not attitudes.empty else None,
        "valid_task_rate": float(tasks["is_valid_task_response"].mean()) if not tasks.empty else None,
    }
    write_json(OUTPUT_DIR / "ai_collection_summary.json", collection_summary)

    human_mnl = maybe_read_json(EXPERIMENT_DIR / "human_baseline_mnl_summary.json")
    ai_mnl = maybe_read_json(EXPERIMENT_DIR / "ai_panel_mnl_summary.json")
    salcm = maybe_read_json(EXPERIMENT_DIR / "ai_salcm_summary.json")

    lines = [
        f"# Experiment Summary: {model_key}",
        "",
        f"This archive contains one model only: `{model_key}`. The AI collection completed `{collection_summary['completed_respondents']}` of `{collection_summary['target_respondents']}` planned runs. The valid-attitude rate is `{fmt(collection_summary['valid_attitude_rate'])}` and the valid-task rate is `{fmt(collection_summary['valid_task_rate'])}`.",
    ]
    if human_mnl and ai_mnl:
        lines.append(
            f"The human baseline MNL final log likelihood is `{fmt(human_mnl.get('final_loglikelihood'), 3)}`. The AI panel MNL final log likelihood is `{fmt(ai_mnl.get('final_loglikelihood'), 3)}`."
        )
    if salcm:
        lines.append(
            f"The SALCM uses `{salcm.get('n_preference_classes', 'NA')}` preference classes and `{salcm.get('n_scale_classes', 'NA')}` scale classes, with final log likelihood `{fmt(salcm.get('final_loglikelihood'), 3)}`."
        )
    lines.append("Raw AI conversation records are stored only in `outputs/`. Derived AI panels and all estimation results are stored in the experiment root.")
    (EXPERIMENT_DIR / "experiment_summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
