from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_behavioral_error.backends.ollama import OllamaBackend  # noqa: E402
from ai_behavioral_error.parsing import parse_choice_label  # noqa: E402

from common import CHOICE_CODE_TO_LABEL, CHOICE_CODE_TO_NAME, DATA_DIR, OUTPUT_DIR, LABEL_TO_CHOICE_CODE, ensure_dir, read_json, utc_timestamp, write_json


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_grounding_payload(raw_text: str) -> dict | None:
    text = strip_code_fence(raw_text)
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def save_transcripts(path: Path, transcripts: dict) -> None:
    path.write_text(json.dumps(transcripts, indent=2), encoding="utf-8")


def load_transcripts(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_progress(path: Path, run_label: str, completed_ids: set[int]) -> None:
    sorted_ids = sorted(completed_ids)
    payload = {
        "run_label": run_label,
        "completed_respondent_count": len(sorted_ids),
        "last_completed_synthetic_respondent_id": sorted_ids[-1] if sorted_ids else 0,
        "completed_respondent_ids": sorted_ids,
        "updated_at_utc": utc_timestamp(),
    }
    write_json(path, payload)


def build_system_prompt(survey_briefing: str, persona: dict) -> str:
    dossier_lines = [
        "Respondent dossier:",
        f"- persona_id: {persona['persona_id']}",
        f"- sex: {persona['sex_text']}",
        f"- age: {persona['age_text']}",
        f"- income: {persona['income_text']}",
        f"- luggage: {persona['luggage_text']}",
        f"- GA travelcard: {persona['ga_text']}",
        f"- travel class: {persona['first_class_text']}",
        f"- ticket category: {persona['ticket_text']}",
        f"- who pays: {persona['who_text']}",
        f"- trip purpose: {persona['purpose_text']}",
        f"- origin: {persona['origin_text']}",
        f"- destination: {persona['dest_text']}",
        f"- survey stratum: {persona['survey_text']}",
        f"- car availability: {persona['car_av_text']}",
    ]
    anchor_lines = [
        "Trip anchor:",
        f"- typical train travel time: {int(persona['anchor_train_time'])} minutes",
        f"- typical train travel cost: {int(persona['anchor_train_cost'])}",
        f"- typical Swissmetro travel time: {int(persona['anchor_sm_time'])} minutes",
        f"- typical Swissmetro travel cost: {int(persona['anchor_sm_cost'])}",
        f"- typical car travel time: {int(persona['anchor_car_time'])} minutes",
        f"- typical car travel cost: {int(persona['anchor_car_cost'])}",
        f"- car availability reminder: {persona['car_av_text']}",
        "- all nine tasks refer to the same trip context",
    ]
    return (
        "You are one respondent in a stated-preference transport survey.\n\n"
        f"{survey_briefing}\n\n"
        f"{chr(10).join(dossier_lines)}\n\n"
        f"{chr(10).join(anchor_lines)}\n\n"
        "Stay in the same persona for the full conversation. "
        "Use only the dossier, the trip anchor, and the current card. "
        "Do not invent extra preferences or background facts. "
        "When asked for a grounding turn, return strict JSON only. "
        "When asked for a choice, return strict JSON only."
    )


def build_grounding_prompt() -> str:
    return (
        "Before the first choice task, restate the respondent dossier as strict JSON only.\n"
        "Required keys: persona_id, sex, age, income, luggage, ga_travelcard, travel_class, ticket_category, "
        "who_pays, trip_purpose, origin, destination, survey_stratum, car_availability, ready_for_survey.\n"
        "Set ready_for_survey to true. Do not add extra text."
    )


def build_task_prompt(task_frame: pd.DataFrame, task_position: int, total_tasks: int) -> str:
    header = [
        f"Choice task {task_position} of {total_tasks}",
        f"Task id: {task_frame['task_id'].iloc[0]}",
        "Labels: A = Train, B = Swissmetro, C = Car.",
        "",
        "Alternatives:",
    ]
    body: list[str] = []
    for _, row in task_frame.sort_values("alternative_id").iterrows():
        available_text = "available" if int(row["is_available"]) == 1 else "not available"
        body.extend(
            [
                f"{row['display_label']}. {row['alternative_name']}",
                f"  availability: {available_text}",
                f"  travel time: {int(row['travel_time'])} minutes",
                f"  travel cost: {int(row['travel_cost'])}",
                (
                    f"  headway: {int(row['headway'])} minutes"
                    if int(row["alternative_id"]) in (1, 2)
                    else "  headway: -"
                ),
                (
                    f"  seat configuration: {int(row['seat_configuration'])}"
                    if int(row["alternative_id"]) == 2
                    else "  seat configuration: -"
                ),
                "",
            ]
        )
    footer = [
        "Choose exactly one available option.",
        'Return strict JSON only, for example {"choice_label":"A"}.',
    ]
    return "\n".join(header + body + footer)


def valid_choice(choice_label: str, task_frame: pd.DataFrame) -> bool:
    if choice_label not in LABEL_TO_CHOICE_CODE:
        return False
    choice_code = LABEL_TO_CHOICE_CODE[choice_label]
    row = task_frame.loc[task_frame["alternative_id"] == choice_code]
    if row.empty:
        return False
    return int(row["is_available"].iloc[0]) == 1


def build_choice_row(persona: dict, task_frame: pd.DataFrame, task_position: int, choice_label: str) -> dict:
    choice_code = LABEL_TO_CHOICE_CODE.get(choice_label, 0)
    choice_name = CHOICE_CODE_TO_NAME.get(choice_code, "")
    return {
        "run_label": str(persona["run_label"]),
        "run_id": int(persona["run_id"]),
        "synthetic_respondent_id": int(persona["synthetic_respondent_id"]),
        "persona_id": str(persona["persona_id"]),
        "task_position": int(task_position),
        "task_id": str(task_frame["task_id"].iloc[0]),
        "survey_stratum": int(persona["survey_stratum"]),
        "ga": int(persona["ga"]),
        "car_av": int(persona["car_av"]),
        "choice_label": choice_label,
        "choice_code": choice_code,
        "choice_name": choice_name,
        "is_valid_choice": int(choice_code in CHOICE_CODE_TO_NAME),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--max-respondents", type=int, default=None)
    args = parser.parse_args()

    config = read_json(DATA_DIR / "experiment_config.json")
    survey_briefing = (DATA_DIR / "survey_briefing_en.md").read_text(encoding="utf-8")

    run_label = f"ai_run_{args.run_id:02d}"
    run_output_dir = OUTPUT_DIR / run_label
    ensure_dir(run_output_dir)

    persona_path = run_output_dir / "persona_samples.csv"
    panels_long_path = run_output_dir / "reconstructed_panels_long.csv"
    if not persona_path.exists() or not panels_long_path.exists():
        raise FileNotFoundError(f"Missing generated AI panels for {run_label}. Run generate_ai_panels.py first.")

    personas = pd.read_csv(persona_path)
    panels_long = pd.read_csv(panels_long_path)

    parsed_path = run_output_dir / "parsed_choices.csv"
    raw_path = run_output_dir / "raw_interactions.jsonl"
    transcript_path = run_output_dir / "respondent_transcripts.json"
    progress_path = run_output_dir / "run_respondents.json"

    if not parsed_path.exists():
        parsed_path.write_text("", encoding="utf-8")
    if not raw_path.exists():
        raw_path.write_text("", encoding="utf-8")
    transcripts = load_transcripts(transcript_path)

    completed: set[int] = set()
    if parsed_path.exists() and parsed_path.stat().st_size > 0:
        prior = pd.read_csv(parsed_path)
        counts = prior.groupby("synthetic_respondent_id").size()
        completed = set(counts[counts >= int(config["tasks_per_respondent"])].index.tolist())
    write_progress(progress_path, run_label, completed)

    backend = OllamaBackend(config["backend"])

    processed = 0
    for persona in personas.sort_values("synthetic_respondent_id").to_dict(orient="records"):
        synthetic_id = int(persona["synthetic_respondent_id"])
        if synthetic_id in completed:
            continue
        if args.max_respondents is not None and processed >= int(args.max_respondents):
            break

        system_prompt = build_system_prompt(survey_briefing, persona)
        messages = [{"role": "system", "content": system_prompt}]
        persona_transcript: list[dict] = [{"role": "system", "content": system_prompt}]

        grounding_prompt = build_grounding_prompt()
        grounding_response = backend.generate(messages + [{"role": "user", "content": grounding_prompt}], request_state={})
        grounding_payload = parse_grounding_payload(grounding_response["response_text"])
        if grounding_payload is None:
            repair_prompt = "Return the grounding payload again as strict JSON only, with no extra text."
            grounding_response = backend.generate(
                messages + [{"role": "user", "content": grounding_prompt}, {"role": "assistant", "content": grounding_response["response_text"]}, {"role": "user", "content": repair_prompt}],
                request_state={},
            )
            grounding_payload = parse_grounding_payload(grounding_response["response_text"])

        append_jsonl(
            raw_path,
            {
                "run_label": run_label,
                "synthetic_respondent_id": synthetic_id,
                "persona_id": persona["persona_id"],
                "phase": "grounding",
                "prompt_text": grounding_prompt,
                **grounding_response,
                **grounding_response["metadata"],
                "parsed_ok": grounding_payload is not None,
            },
        )

        messages.append({"role": "user", "content": grounding_prompt})
        messages.append({"role": "assistant", "content": grounding_response["response_text"]})
        persona_transcript.extend(messages[1:])
        base_messages = list(messages)

        choice_rows: list[dict] = []
        for task_position, (_, task_frame) in enumerate(
            panels_long.loc[panels_long["synthetic_respondent_id"] == synthetic_id].groupby("task_id", sort=False),
            start=1,
        ):
            task_prompt = build_task_prompt(task_frame, task_position, int(config["tasks_per_respondent"]))
            attempt_messages = base_messages + [{"role": "user", "content": task_prompt}]
            choice_response = backend.generate(attempt_messages, request_state={})
            choice_label = parse_choice_label(choice_response["response_text"])
            retries = 0
            while (not valid_choice(choice_label, task_frame)) and retries < 2:
                retries += 1
                repair_prompt = 'Return strict JSON only using one available label. Example: {"choice_label":"A"}.'
                attempt_messages = attempt_messages + [{"role": "assistant", "content": choice_response["response_text"]}, {"role": "user", "content": repair_prompt}]
                choice_response = backend.generate(attempt_messages, request_state={})
                choice_label = parse_choice_label(choice_response["response_text"])

            if not valid_choice(choice_label, task_frame):
                available_row = task_frame.loc[task_frame["is_available"] == 1].sort_values("alternative_id").iloc[0]
                choice_label = str(available_row["display_label"])

            append_jsonl(
                raw_path,
                {
                    "run_label": run_label,
                    "synthetic_respondent_id": synthetic_id,
                    "persona_id": persona["persona_id"],
                    "phase": "choice",
                    "task_id": str(task_frame["task_id"].iloc[0]),
                    "task_position": int(task_position),
                    "prompt_text": task_prompt,
                    **choice_response,
                    **choice_response["metadata"],
                    "parsed_choice_label": choice_label,
                    "retry_count": retries,
                },
            )

            choice_rows.append(build_choice_row(persona, task_frame, task_position, choice_label))
            persona_transcript.extend(
                [
                    {"role": "user", "content": task_prompt},
                    {"role": "assistant", "content": choice_response["response_text"]},
                ]
            )

        choice_frame = pd.DataFrame(choice_rows)
        header = not parsed_path.exists() or parsed_path.stat().st_size == 0
        choice_frame.to_csv(parsed_path, mode="a", header=header, index=False)

        transcripts[str(synthetic_id)] = persona_transcript
        save_transcripts(transcript_path, transcripts)
        processed += 1
        completed.add(synthetic_id)
        write_progress(progress_path, run_label, completed)

    write_json(
        run_output_dir / "questionnaire_manifest.json",
        {
            "experiment_name": config["experiment_name"],
            "run_label": run_label,
            "replication_standard": config["replication_standard"],
            "n_personas": int(len(personas)),
            "tasks_per_respondent": int(config["tasks_per_respondent"]),
            "backend": config["backend"],
            "questionnaire_style": "multi_turn_one_task_at_a_time",
            "generated_at_utc": utc_timestamp(),
        },
    )

    parsed = pd.read_csv(parsed_path) if parsed_path.exists() and parsed_path.stat().st_size > 0 else pd.DataFrame()
    valid_rate = float(parsed["is_valid_choice"].mean()) if not parsed.empty else 0.0
    if not parsed.empty:
        counts = parsed.groupby("synthetic_respondent_id").size()
        completed = set(counts[counts >= int(config["tasks_per_respondent"])].index.tolist())
        write_progress(progress_path, run_label, completed)
    print(f"[collect] {run_label}: choices={len(parsed)} valid_rate={valid_rate:.4f}")


if __name__ == "__main__":
    main()
