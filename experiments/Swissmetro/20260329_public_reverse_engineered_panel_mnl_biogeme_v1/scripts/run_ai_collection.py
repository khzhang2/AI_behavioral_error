from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_behavioral_error.backends.ollama import OllamaBackend  # noqa: E402
from ai_behavioral_error.parsing import parse_choice_label  # noqa: E402

from common import (  # noqa: E402
    CHOICE_CODE_TO_NAME,
    CHOICE_LABEL_TO_CODE,
    CHOICE_LABEL_TO_NAME,
    DATA_DIR,
    OUTPUT_DIR,
    ensure_dir,
    read_json,
    reconstructed_value,
    weighted_sample_one,
    write_json,
)
from questionnaire_template import (  # noqa: E402
    build_choice_prompt,
    build_grounding_prompt,
    build_system_prompt,
)


CONFIG = read_json(DATA_DIR / "experiment_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-respondents", type=int, default=CONFIG["n_respondents"])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_json_payload(raw_text: str) -> dict:
    stripped = raw_text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    if not stripped.startswith("{"):
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_backends() -> tuple[OllamaBackend, OllamaBackend]:
    backend_config = dict(CONFIG["backend"])
    grounding_backend = OllamaBackend(
        {
            **backend_config,
            "num_predict": int(backend_config["grounding_num_predict"]),
        }
    )
    choice_backend = OllamaBackend(
        {
            **backend_config,
            "num_predict": int(backend_config["choice_num_predict"]),
        }
    )
    return grounding_backend, choice_backend


def select_baseline(
    baselines: pd.DataFrame,
    survey_stratum: int,
    car_av: int,
    ga: int,
    rng: np.random.Generator,
) -> pd.Series:
    candidates = baselines.loc[
        (baselines["survey_stratum"] == survey_stratum)
        & (baselines["car_av"] == car_av)
        & (baselines["ga"] == ga)
    ]
    if candidates.empty:
        candidates = baselines.loc[
            (baselines["survey_stratum"] == survey_stratum) & (baselines["car_av"] == car_av)
        ]
    if candidates.empty:
        candidates = baselines.loc[baselines["survey_stratum"] == survey_stratum]
    if candidates.empty:
        candidates = baselines
    return candidates.iloc[int(rng.integers(len(candidates)))]


def select_template(catalog: pd.DataFrame, survey_stratum: int, rng: np.random.Generator) -> pd.DataFrame:
    scoped = catalog.loc[catalog["survey_stratum"] == survey_stratum].copy()
    if scoped.empty:
        scoped = catalog.copy()
    template_frame = scoped[["template_id", "template_weight"]].drop_duplicates().reset_index(drop=True)
    sampled = weighted_sample_one(template_frame, rng, "template_weight")
    return scoped.loc[scoped["template_id"] == sampled["template_id"]].sort_values("task_position").reset_index(drop=True)


def make_persona_samples(
    n_respondents: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles = pd.read_csv(DATA_DIR / "human_respondent_profiles.csv")
    baselines = pd.read_csv(DATA_DIR / "reconstructed_panel_baselines.csv")
    catalog = pd.read_csv(DATA_DIR / "reconstructed_panel_catalog.csv")

    sampled_indices = rng.integers(0, len(profiles), size=n_respondents)
    persona_rows: list[dict] = []
    wide_rows: list[dict] = []
    long_rows: list[dict] = []

    custom_id = 1
    for respondent_number, sampled_index in enumerate(sampled_indices, start=1):
        profile = profiles.iloc[int(sampled_index)].to_dict()
        respondent_id = f"AI{respondent_number:04d}"
        profile["respondent_id"] = respondent_id
        profile["persona_id"] = respondent_id

        survey_stratum = int(profile["SURVEY"])
        car_av = int(profile["CAR_AV"])
        ga = int(profile["GA"])

        template_rows = select_template(catalog, survey_stratum, rng)
        baseline = select_baseline(baselines, survey_stratum, car_av, ga, rng)

        persona_rows.append(
            {
                "respondent_id": respondent_id,
                "persona_id": respondent_id,
                "synthetic_numeric_id": respondent_number,
                "source_profile_id": int(profile["ID"]),
                "sampled_template_id": template_rows["template_id"].iloc[0],
                "sampled_baseline_source_human_id": int(baseline["source_human_id"]),
                **profile,
            }
        )

        for _, task in template_rows.iterrows():
            task_position = int(task["task_position"])
            train_tt = reconstructed_value(baseline["baseline_train_time"], task["train_time_multiplier"])
            train_co = reconstructed_value(baseline["baseline_train_cost"], task["train_cost_multiplier"])
            sm_tt = reconstructed_value(baseline["baseline_sm_time"], task["sm_time_multiplier"])
            sm_co = reconstructed_value(baseline["baseline_sm_cost"], task["sm_cost_multiplier"])
            car_tt = reconstructed_value(
                baseline["baseline_car_time"],
                task["car_time_multiplier"],
                zero_if_nonpositive=True,
            )
            car_co = reconstructed_value(
                baseline["baseline_car_cost"],
                task["car_cost_multiplier"],
                zero_if_nonpositive=True,
            )

            if ga == 1:
                train_co = 0
                sm_co = 0
            if car_av == 0:
                car_tt = 0
                car_co = 0

            wide_row = {
                "GROUP": int(profile["GROUP"]),
                "SURVEY": survey_stratum,
                "SP": 1,
                "ID": respondent_number,
                "respondent_id": respondent_id,
                "persona_id": respondent_id,
                "source_profile_id": int(profile["ID"]),
                "sampled_template_id": template_rows["template_id"].iloc[0],
                "PURPOSE": int(profile["PURPOSE"]),
                "FIRST": int(profile["FIRST"]),
                "TICKET": int(profile["TICKET"]),
                "WHO": int(profile["WHO"]),
                "LUGGAGE": int(profile["LUGGAGE"]),
                "AGE": int(profile["AGE"]),
                "MALE": int(profile["MALE"]),
                "INCOME": int(profile["INCOME"]),
                "GA": ga,
                "ORIGIN": int(profile["ORIGIN"]),
                "DEST": int(profile["DEST"]),
                "TRAIN_AV": 1,
                "CAR_AV": car_av,
                "SM_AV": 1,
                "TRAIN_TT": train_tt,
                "TRAIN_CO": train_co,
                "TRAIN_HE": int(task["train_headway"]),
                "SM_TT": sm_tt,
                "SM_CO": sm_co,
                "SM_HE": int(task["sm_headway"]),
                "SM_SEATS": int(task["sm_seats"]),
                "CAR_TT": car_tt,
                "CAR_CO": car_co,
                "CHOICE": 0,
                "custom_id": custom_id,
                "task_position": task_position,
                "task_id": f"{respondent_id}_T{task_position:02d}",
                "TRAIN_COST_SCALED": train_co / 100.0,
                "SM_COST_SCALED": sm_co / 100.0,
                "CAR_COST_SCALED": car_co / 100.0,
                "TRAIN_TIME_SCALED": train_tt / 100.0,
                "SM_TIME_SCALED": sm_tt / 100.0,
                "CAR_TIME_SCALED": car_tt / 100.0,
                "chosen_alternative_name": "",
                "sex_text": profile["sex_text"],
                "age_text": profile["age_text"],
                "income_text": profile["income_text"],
                "ga_text": profile["ga_text"],
                "car_av_text": profile["car_av_text"],
                "luggage_text": profile["luggage_text"],
                "purpose_text": profile["purpose_text"],
                "first_class_text": profile["first_class_text"],
                "ticket_text": profile["ticket_text"],
                "who_text": profile["who_text"],
                "origin_text": profile["origin_text"],
                "dest_text": profile["dest_text"],
                "survey_text": profile["survey_text"],
            }
            wide_rows.append(wide_row)

            for mode_id, alt_name in [(1, "TRAIN"), (2, "SWISSMETRO"), (3, "CAR")]:
                long_rows.append(
                    {
                        "custom_id": custom_id,
                        "respondent_id": respondent_id,
                        "ID": respondent_number,
                        "task_id": wide_row["task_id"],
                        "task_position": task_position,
                        "mode_id": mode_id,
                        "alternative_name": alt_name,
                        "choice": 0,
                        "availability": 0 if alt_name == "CAR" and car_av == 0 else 1,
                        "travel_time": wide_row[f"{alt_name if alt_name != 'SWISSMETRO' else 'SM'}_TT"],
                        "travel_cost": wide_row[f"{alt_name if alt_name != 'SWISSMETRO' else 'SM'}_CO"],
                        "headway": 0
                        if alt_name == "CAR"
                        else wide_row[f"{alt_name if alt_name != 'SWISSMETRO' else 'SM'}_HE"],
                        "seat_configuration": wide_row["SM_SEATS"] if alt_name == "SWISSMETRO" else 0,
                        "travel_time_hundredth": wide_row[
                            f"{alt_name if alt_name != 'SWISSMETRO' else 'SM'}_TIME_SCALED"
                        ],
                        "travel_cost_hundredth": wide_row[
                            f"{alt_name if alt_name != 'SWISSMETRO' else 'SM'}_COST_SCALED"
                        ],
                        "GROUP": wide_row["GROUP"],
                        "SURVEY": wide_row["SURVEY"],
                        "SP": 1,
                        "PURPOSE": wide_row["PURPOSE"],
                        "FIRST": wide_row["FIRST"],
                        "TICKET": wide_row["TICKET"],
                        "WHO": wide_row["WHO"],
                        "LUGGAGE": wide_row["LUGGAGE"],
                        "AGE": wide_row["AGE"],
                        "MALE": wide_row["MALE"],
                        "INCOME": wide_row["INCOME"],
                        "GA": wide_row["GA"],
                        "ORIGIN": wide_row["ORIGIN"],
                        "DEST": wide_row["DEST"],
                        "CAR_AV": wide_row["CAR_AV"],
                    }
                )
            custom_id += 1

    return pd.DataFrame(persona_rows), pd.DataFrame(wide_rows), pd.DataFrame(long_rows)


def write_manifest(n_respondents: int) -> None:
    manifest = {
        "experiment_name": CONFIG["experiment_name"],
        "n_respondents": n_respondents,
        "tasks_per_respondent": CONFIG["tasks_per_respondent"],
        "llm_model": CONFIG["backend"]["model_name"],
        "prompt_template_file": str((Path(__file__).resolve().parent / "questionnaire_template.py")),
        "compact_grounding": True,
        "growing_context_across_questions": True,
        "concept_explanations": [
            "GA covers both Train and Swissmetro fares",
            "headway means minutes between departures",
            "Swissmetro seat configuration is a binary comfort attribute",
        ],
    }
    write_json(OUTPUT_DIR / "questionnaire_manifest.json", manifest)


def initialize_output_files() -> None:
    ensure_dir(OUTPUT_DIR)
    (OUTPUT_DIR / "raw_interactions.jsonl").write_text("", encoding="utf-8")
    write_json(
        OUTPUT_DIR / "respondent_transcripts.json",
        {"experiment_name": CONFIG["experiment_name"], "respondents": {}},
    )
    write_json(
        OUTPUT_DIR / "run_respondents.json",
        {
            "experiment_name": CONFIG["experiment_name"],
            "target_respondents": 0,
            "completed_respondents": 0,
            "updated_at": now_iso(),
        },
    )


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_previous_answers(rows: list[dict]) -> list[str]:
    return [
        f"Q{int(row['task_position'])}={row['choice_label']} ({row['chosen_alternative_name']})"
        for row in rows
        if row.get("choice_label") in CHOICE_LABEL_TO_CODE
    ]


def load_transcripts() -> dict:
    path = OUTPUT_DIR / "respondent_transcripts.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"experiment_name": CONFIG["experiment_name"], "respondents": {}}


def save_transcripts(payload: dict) -> None:
    write_json(OUTPUT_DIR / "respondent_transcripts.json", payload)


def load_completed_respondents() -> set[str]:
    parsed_path = OUTPUT_DIR / "parsed_choices.csv"
    if not parsed_path.exists():
        return set()
    parsed = pd.read_csv(parsed_path)
    if parsed.empty:
        return set()
    counts = parsed.groupby("respondent_id")["task_position"].count()
    return set(counts[counts == CONFIG["tasks_per_respondent"]].index.tolist())


def update_progress(target_respondents: int, completed_ids: set[str]) -> None:
    payload = {
        "experiment_name": CONFIG["experiment_name"],
        "target_respondents": int(target_respondents),
        "completed_respondents": int(len(completed_ids)),
        "updated_at": now_iso(),
    }
    write_json(OUTPUT_DIR / "run_respondents.json", payload)


def load_or_generate_panels(n_respondents: int, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    persona_path = OUTPUT_DIR / "persona_samples.csv"
    wide_path = OUTPUT_DIR / "reconstructed_panels_wide.csv"
    long_path = OUTPUT_DIR / "reconstructed_panels_long.csv"

    if resume and persona_path.exists() and wide_path.exists() and long_path.exists():
        return pd.read_csv(persona_path), pd.read_csv(wide_path)

    rng = np.random.default_rng(int(CONFIG["master_seed"]))
    personas, wide_rows, long_rows = make_persona_samples(n_respondents, rng)
    personas.to_csv(persona_path, index=False)
    wide_rows.to_csv(wide_path, index=False)
    long_rows.to_csv(long_path, index=False)
    write_manifest(n_respondents)
    return personas, wide_rows


def save_parsed_rows(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "parsed_choices.csv", index=False)


def run_collection(personas: pd.DataFrame, panels: pd.DataFrame, resume: bool) -> None:
    grounding_backend, choice_backend = build_backends()
    parsed_rows = []
    parsed_path = OUTPUT_DIR / "parsed_choices.csv"
    if resume and parsed_path.exists():
        parsed_rows = pd.read_csv(parsed_path).to_dict(orient="records")

    completed_ids = load_completed_respondents() if resume else set()
    transcripts = load_transcripts() if resume else {"experiment_name": CONFIG["experiment_name"], "respondents": {}}
    raw_path = OUTPUT_DIR / "raw_interactions.jsonl"

    for _, persona in personas.iterrows():
        respondent_id = str(persona["respondent_id"])
        if respondent_id in completed_ids:
            continue

        persona_dict = persona.to_dict()
        task_rows = panels.loc[panels["respondent_id"] == respondent_id].sort_values("task_position")
        messages: list[dict] = [{"role": "system", "content": build_system_prompt(persona_dict)}]
        turns: list[dict] = []

        grounding_prompt = build_grounding_prompt(persona_dict)
        grounding_messages = messages + [{"role": "user", "content": grounding_prompt}]
        grounding_result = grounding_backend.generate(
            grounding_messages,
            request_state={"respondent_id": respondent_id, "stage": "grounding"},
        )
        grounding_payload = parse_json_payload(grounding_result["response_text"])
        messages.append({"role": "user", "content": grounding_prompt})
        messages.append({"role": "assistant", "content": grounding_result["response_text"]})
        turns.append(
            {
                "stage": "grounding",
                "prompt": grounding_prompt,
                "response_text": grounding_result["response_text"],
                "response_payload": grounding_payload,
                "metadata": grounding_result["metadata"],
                "is_valid_grounding": bool(grounding_payload.get("ready") is True),
            }
        )
        append_jsonl(
            raw_path,
            {
                "respondent_id": respondent_id,
                "stage": "grounding",
                "prompt": grounding_prompt,
                "messages_payload": grounding_messages,
                "message_count": len(grounding_messages),
                "response_text": grounding_result["response_text"],
                "metadata": grounding_result["metadata"],
            },
        )

        respondent_rows: list[dict] = []
        for _, task in task_rows.iterrows():
            task_dict = task.to_dict()
            previous_answers = summarize_previous_answers(respondent_rows)
            choice_prompt = build_choice_prompt(
                task_dict,
                int(task["task_position"]),
                CONFIG["tasks_per_respondent"],
                previous_answers=previous_answers,
            )
            choice_messages = messages + [{"role": "user", "content": choice_prompt}]
            choice_result = choice_backend.generate(
                choice_messages,
                request_state={
                    "respondent_id": respondent_id,
                    "stage": "choice",
                    "task_id": task_dict["task_id"],
                },
            )
            response_text = choice_result["response_text"]
            choice_label = parse_choice_label(response_text)

            if choice_label not in CHOICE_LABEL_TO_CODE:
                repair_prompt = (
                    "Your last reply was invalid. Reply again with JSON only in exactly this form: "
                    '{"choice_label":"A"}'
                )
                repair_messages = (
                    messages
                    + [{"role": "user", "content": choice_prompt}]
                    + [{"role": "assistant", "content": response_text}]
                    + [{"role": "user", "content": repair_prompt}]
                )
                repair_result = choice_backend.generate(
                    repair_messages,
                    request_state={
                        "respondent_id": respondent_id,
                        "stage": "choice_repair",
                        "task_id": task_dict["task_id"],
                    },
                )
                response_text = repair_result["response_text"]
                choice_result = repair_result
                choice_label = parse_choice_label(response_text)
                append_jsonl(
                    raw_path,
                    {
                        "respondent_id": respondent_id,
                        "stage": "choice_repair",
                        "task_id": task_dict["task_id"],
                        "prompt": repair_prompt,
                        "messages_payload": repair_messages,
                        "message_count": len(repair_messages),
                        "response_text": response_text,
                        "metadata": repair_result["metadata"],
                    },
                )

            is_valid_choice = choice_label in CHOICE_LABEL_TO_CODE
            chosen_code = CHOICE_LABEL_TO_CODE.get(choice_label, 0)
            chosen_name = CHOICE_CODE_TO_NAME.get(chosen_code, "")
            duration_sec = float(choice_result["metadata"].get("total_duration", 0)) / 1_000_000_000.0

            respondent_rows.append(
                {
                    "experiment_name": CONFIG["experiment_name"],
                    "respondent_id": respondent_id,
                    "persona_id": respondent_id,
                    "task_id": task_dict["task_id"],
                    "task_position": int(task_dict["task_position"]),
                    "choice_label": choice_label,
                    "choice_code": chosen_code,
                    "chosen_alternative_name": chosen_name,
                    "is_valid_choice": int(is_valid_choice),
                    "grounding_is_valid": int(turns[0]["is_valid_grounding"]),
                    "GA": int(persona_dict["GA"]),
                    "CAR_AV": int(persona_dict["CAR_AV"]),
                    "SEX_TEXT": persona_dict["sex_text"],
                    "AGE_TEXT": persona_dict["age_text"],
                    "INCOME_TEXT": persona_dict["income_text"],
                    "duration_sec": duration_sec,
                    "prompt_eval_count": int(choice_result["metadata"].get("prompt_eval_count", 0)),
                    "eval_count": int(choice_result["metadata"].get("eval_count", 0)),
                }
            )

            messages.append({"role": "user", "content": choice_prompt})
            messages.append({"role": "assistant", "content": response_text})
            turns.append(
                {
                    "stage": "choice",
                    "task_id": task_dict["task_id"],
                    "prompt": choice_prompt,
                    "response_text": response_text,
                    "choice_label": choice_label,
                    "is_valid_choice": is_valid_choice,
                    "metadata": choice_result["metadata"],
                }
            )
            append_jsonl(
                raw_path,
                {
                    "respondent_id": respondent_id,
                    "stage": "choice",
                    "task_id": task_dict["task_id"],
                    "prompt": choice_prompt,
                    "messages_payload": choice_messages,
                    "message_count": len(choice_messages),
                    "previous_answers": previous_answers,
                    "response_text": response_text,
                    "metadata": choice_result["metadata"],
                },
            )

        if len(respondent_rows) == CONFIG["tasks_per_respondent"]:
            parsed_rows = [row for row in parsed_rows if row["respondent_id"] != respondent_id]
            parsed_rows.extend(respondent_rows)
            save_parsed_rows(parsed_rows)
            transcripts["respondents"][respondent_id] = {
                "persona": persona_dict,
                "turns": turns,
            }
            save_transcripts(transcripts)
            completed_ids.add(respondent_id)
            update_progress(len(personas), completed_ids)
            print(
                f"[collection] completed respondent={respondent_id} "
                f"choices={len(respondent_rows)} valid={sum(row['is_valid_choice'] for row in respondent_rows)}"
            )


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)
    if not args.resume:
        initialize_output_files()
    personas, panels = load_or_generate_panels(args.n_respondents, args.resume)
    update_progress(len(personas), load_completed_respondents() if args.resume else set())
    run_collection(personas, panels, args.resume)


if __name__ == "__main__":
    main()
