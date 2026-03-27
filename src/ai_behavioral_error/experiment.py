from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ai_behavioral_error.backends import build_backend
from ai_behavioral_error.design import ChoiceTask, load_choice_tasks
from ai_behavioral_error.io import ensure_dir, load_csv, load_json, resolve_path, write_csv, write_json
from ai_behavioral_error.mnl import fit_conditional_logit
from ai_behavioral_error.parsing import parse_choice_label
from ai_behavioral_error.plotting import plot_choice_shares, plot_mnl_coefficients, plot_position_bias
from ai_behavioral_error.prompting import build_system_prompt, build_user_prompt
from ai_behavioral_error.response_analysis import analyze_response_composition


def _choice_set_id(respondent_id: str, task_id: str, repeat_id: int) -> str:
    return f"{respondent_id}_{task_id}_rep{repeat_id:02d}"


def _shuffle_alternatives(task: ChoiceTask, master_seed: int, respondent_id: str, repeat_id: int) -> list:
    alternatives = list(task.alternatives)
    rng = random.Random(f"{master_seed}-{respondent_id}-{task.task_id}-{repeat_id}")
    rng.shuffle(alternatives)
    return alternatives


def _display_bundle(task: ChoiceTask, master_seed: int, respondent_id: str, repeat_id: int, randomize: bool) -> list[tuple[str, object]]:
    alternatives = _shuffle_alternatives(task, master_seed, respondent_id, repeat_id) if randomize else list(task.alternatives)
    display_labels = ["A", "B", "C", "D"]
    return list(zip(display_labels, alternatives))


def _alternative_features(alternative) -> dict:
    return {
        "travel_time_min": alternative.travel_time_min,
        "access_time_min": alternative.access_time_min,
        "egress_time_min": alternative.egress_time_min,
        "parking_search_time_min": alternative.parking_search_time_min,
        "cost_eur": alternative.cost_eur,
        "availability_share": alternative.availability_pct / 100.0,
        "is_e_scooter": int(alternative.alternative_id == "e_scooter"),
        "is_bikesharing": int(alternative.alternative_id == "bikesharing"),
        "is_private_car": int(alternative.alternative_id == "private_car"),
    }


def _build_long_format(choice_frame: pd.DataFrame, design_frame: pd.DataFrame) -> pd.DataFrame:
    merged = choice_frame.merge(
        design_frame,
        on=["task_id", "alternative_id"],
        how="left",
        suffixes=("", "_design"),
    )

    merged["choice"] = (merged["alternative_id"] == merged["chosen_alternative_id"]).astype(int)
    merged["choice_set_id"] = merged.apply(
        lambda row: _choice_set_id(row["respondent_id"], row["task_id"], int(row["repeat_id"])),
        axis=1,
    )
    return merged


def run_pipeline(config_path: str | Path) -> dict:
    config_file = Path(config_path).resolve()
    project_root = config_file.parents[1] if config_file.parent.name == "configs" else config_file.parent
    config = load_json(config_file)

    design_path = resolve_path(config["design_path"], project_root)
    output_dir = resolve_path(config["output_dir"], project_root)
    ensure_dir(output_dir)

    design_frame = load_csv(design_path)
    tasks = load_choice_tasks(design_frame)
    backend = build_backend(config["backend"])

    interaction_rows = []
    choice_rows = []
    expanded_rows = []
    system_prompt = build_system_prompt()

    for respondent_index in range(1, int(config["n_respondents"]) + 1):
        respondent_id = f"R{respondent_index:04d}"
        for repeat_id in range(1, int(config["repeats_per_task"]) + 1):
            for task in tasks:
                display_bundle = _display_bundle(
                    task=task,
                    master_seed=int(config["master_seed"]),
                    respondent_id=respondent_id,
                    repeat_id=repeat_id,
                    randomize=bool(config["randomize_alternative_order"]),
                )
                user_prompt = build_user_prompt(config["scenario_text"], task, display_bundle)

                request_state = {
                    "respondent_id": respondent_id,
                    "task_id": task.task_id,
                    "repeat_id": repeat_id,
                    "alternatives": [
                        {
                            "display_label": display_label,
                            "alternative_id": alternative.alternative_id,
                            "alternative_name": alternative.alternative_name,
                            "features": _alternative_features(alternative),
                        }
                        for display_label, alternative in display_bundle
                    ],
                }

                backend_response = backend.generate(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    request_state=request_state,
                )

                response_text = backend_response["response_text"]
                thinking_text = backend_response["thinking_text"]
                parsed_choice_label = parse_choice_label(response_text)
                label_map = {display_label: alternative for display_label, alternative in display_bundle}
                chosen_alternative = label_map.get(parsed_choice_label)

                interaction_rows.append(
                    {
                        "experiment_name": config["experiment_name"],
                        "questionnaire_version": config["questionnaire_version"],
                        "prompt_version": config["prompt_version"],
                        "respondent_id": respondent_id,
                        "repeat_id": repeat_id,
                        "task_id": task.task_id,
                        "task_index": task.task_index,
                        "backend_type": config["backend"]["type"],
                        "model_name": backend.model_name,
                        "temperature": config["backend"].get("temperature", 0.0),
                        "top_p": config["backend"].get("top_p", 1.0),
                        "top_k": config["backend"].get("top_k", 0),
                        "seed": config["backend"].get("seed", 0),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "raw_response": backend_response["raw_text"],
                        "thinking_text": thinking_text,
                        "response_text": response_text,
                        "parsed_choice_label": parsed_choice_label,
                        "thinking_char_count": len(thinking_text),
                        "response_char_count": len(response_text),
                        "alternative_order": json.dumps(
                            [
                                {
                                    "display_label": display_label,
                                    "alternative_id": alternative.alternative_id,
                                    "alternative_name": alternative.alternative_name,
                                }
                                for display_label, alternative in display_bundle
                            ]
                        ),
                        **backend_response["metadata"],
                    }
                )

                choice_rows.append(
                    {
                        "experiment_name": config["experiment_name"],
                        "questionnaire_version": config["questionnaire_version"],
                        "prompt_version": config["prompt_version"],
                        "respondent_id": respondent_id,
                        "repeat_id": repeat_id,
                        "task_id": task.task_id,
                        "task_index": task.task_index,
                        "chosen_display_label": parsed_choice_label,
                        "chosen_alternative_id": chosen_alternative.alternative_id if chosen_alternative else "",
                        "chosen_alternative_name": chosen_alternative.alternative_name if chosen_alternative else "",
                        "is_valid_choice": int(chosen_alternative is not None),
                    }
                )

                for display_label, alternative in display_bundle:
                    expanded_rows.append(
                        {
                            "respondent_id": respondent_id,
                            "repeat_id": repeat_id,
                            "task_id": task.task_id,
                            "task_index": task.task_index,
                            "display_label": display_label,
                            "alternative_id": alternative.alternative_id,
                        }
                    )

    interaction_frame = pd.DataFrame(interaction_rows)
    choice_frame = pd.DataFrame(choice_rows)
    expanded_frame = pd.DataFrame(expanded_rows)

    long_frame = _build_long_format(
        choice_frame=choice_frame.merge(expanded_frame, on=["respondent_id", "repeat_id", "task_id", "task_index"], how="left"),
        design_frame=design_frame[["task_id", "alternative_id", "alternative_name", "trip_length_km", "travel_time_min", "access_time_min", "egress_time_min", "parking_search_time_min", "availability_pct", "cost_eur", "scheme", "engine", "range_km"]],
    )

    interaction_frame.to_json(output_dir / "raw_interactions.jsonl", orient="records", lines=True)
    write_csv(output_dir / "choices.csv", choice_frame)
    write_csv(output_dir / "long_format.csv", long_frame)

    coefficient_frame = fit_conditional_logit(
        long_frame=long_frame,
        feature_names=list(config["analysis_features"]),
        output_dir=output_dir,
    )

    diagnostics = {
        "experiment_name": config["experiment_name"],
        "n_respondents": int(config["n_respondents"]),
        "repeats_per_task": int(config["repeats_per_task"]),
        "n_tasks": len(tasks),
        "n_choices": int(len(choice_frame)),
        "valid_choice_rate": float(choice_frame["is_valid_choice"].mean()),
        "choice_share_by_alternative": choice_frame["chosen_alternative_name"].value_counts(normalize=True).to_dict(),
        "choice_share_by_display_label": choice_frame["chosen_display_label"].value_counts(normalize=True).to_dict(),
    }

    response_composition = analyze_response_composition(interaction_frame, choice_frame, output_dir)
    write_json(output_dir / "diagnostics.json", diagnostics)
    write_json(output_dir / "response_composition_summary.json", response_composition)
    plot_choice_shares(choice_frame, output_dir)
    plot_position_bias(choice_frame, output_dir)
    plot_mnl_coefficients(coefficient_frame, output_dir)
    return diagnostics
