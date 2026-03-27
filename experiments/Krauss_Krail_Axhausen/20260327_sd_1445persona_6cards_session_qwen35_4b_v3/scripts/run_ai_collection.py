from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_behavioral_error.backends.ollama import OllamaBackend  # noqa: E402


AGE_BAND_RANGES = {
    "18_29": (18, 29),
    "30_39": (30, 39),
    "40_49": (40, 49),
    "50_59": (50, 59),
    "60_67": (60, 67),
    "68_plus": (68, 80),
}

ALTERNATIVE_CODE = {
    "e_scooter": 1,
    "bikesharing": 2,
    "walking": 3,
    "private_car": 4,
}

LABEL_TO_ALTERNATIVE_ID = {
    "A": "e_scooter",
    "B": "bikesharing",
    "C": "walking",
    "D": "private_car",
}

TEXT_TO_LABEL = {
    "a": "A",
    "b": "B",
    "c": "C",
    "d": "D",
    "e-scooter": "A",
    "e scooter": "A",
    "e_scooter": "A",
    "e-scooter sharing": "A",
    "bike sharing": "B",
    "bikesharing": "B",
    "bike": "B",
    "walking": "C",
    "walk": "C",
    "private car": "D",
    "car": "D",
    "private_car": "D",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sample_categorical(rng: random.Random, distribution: dict[str, float]) -> str:
    draw = rng.random()
    cumulative = 0.0
    last_key = next(reversed(distribution))
    for key, probability in distribution.items():
        cumulative += float(probability)
        if draw <= cumulative:
            return key
    return last_key


def sample_truncated_poisson(rng: random.Random, mean: float, min_value: int, max_value: int) -> int:
    threshold = 2.718281828459045 ** (-mean)
    k = 0
    product = 1.0
    while product > threshold:
        k += 1
        product *= rng.random()
    value = k - 1
    return max(min_value, min(max_value, value))


def clean_text(value: object) -> str:
    if pd.isna(value):
        return "-"
    text = str(value).strip()
    return text if text else "-"


def format_attribute(value: object, unit: str) -> str:
    if pd.isna(value):
        return "-"
    if value in ("", None):
        return "-"
    numeric = float(value)
    if numeric == 0:
        return "-"
    if unit == "EUR":
        return f"{numeric:g}"
    if unit == "%":
        return f"{numeric:g}"
    if unit == "km":
        return f"{numeric:g}"
    return f"{numeric:g}"


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def normalize_choice_label(value: object) -> str:
    text = str(value).strip().strip('"').strip("'")
    if not text:
        return ""
    lower = text.lower()
    if lower in TEXT_TO_LABEL:
        return TEXT_TO_LABEL[lower]
    if len(text) == 1 and text.upper() in LABEL_TO_ALTERNATIVE_ID:
        return text.upper()
    match = re.search(r"\b([ABCD])\b", text.upper())
    if match:
        return match.group(1)
    return ""


def parse_session_choices(response_text: str, task_ids: list[str]) -> dict[str, str]:
    stripped = strip_code_fence(response_text)
    parsed: dict[str, str] = {}

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        if isinstance(payload.get("choices"), dict):
            payload = payload["choices"]
        for index, task_id in enumerate(task_ids, start=1):
            for candidate_key in (task_id, str(index), f"task_{index}", f"task{index}"):
                if candidate_key in payload:
                    label = normalize_choice_label(payload[candidate_key])
                    if label:
                        parsed[task_id] = label
                        break

    if len(parsed) != len(task_ids):
        for index, task_id in enumerate(task_ids, start=1):
            if task_id in parsed:
                continue
            patterns = [
                rf'"{re.escape(task_id)}"\s*:\s*"([^"]+)"',
                rf'"{index}"\s*:\s*"([^"]+)"',
                rf"{re.escape(task_id)}\s*[:=]\s*([A-Da-d]|walking|walk|car|private car|bike sharing|bikesharing|e-scooter|e scooter)",
                rf"\b{index}\b\s*[:=]\s*([A-Da-d])",
            ]
            for pattern in patterns:
                match = re.search(pattern, stripped, flags=re.IGNORECASE)
                if match:
                    label = normalize_choice_label(match.group(1))
                    if label:
                        parsed[task_id] = label
                        break

    return {task_id: parsed.get(task_id, "") for task_id in task_ids}


def build_persona(persona_num: int, rng: random.Random, rules: dict) -> dict:
    exact = rules["exact_distributions"]
    approximations = rules["moment_matching_approximations"]
    assumptions = rules["assumed_distributions"]

    female = int(rng.random() < float(exact["female"]))
    age_band = sample_categorical(rng, exact["age_band"])
    age_low, age_high = AGE_BAND_RANGES[age_band]
    age_years = rng.randint(age_low, age_high)
    income_band = sample_categorical(rng, exact["income_band"])
    metropolis = int(rng.random() < float(exact["metropolis"]))
    pt_pass = int(rng.random() < float(exact["pt_pass"]))
    household_cars = sample_truncated_poisson(
        rng=rng,
        mean=float(approximations["household_cars"]["mean"]),
        min_value=int(approximations["household_cars"]["min"]),
        max_value=int(approximations["household_cars"]["max"]),
    )
    accessible_bikes = sample_truncated_poisson(
        rng=rng,
        mean=float(approximations["accessible_bikes"]["mean"]),
        min_value=int(approximations["accessible_bikes"]["min"]),
        max_value=int(approximations["accessible_bikes"]["max"]),
    )
    maas_subscription = int(rng.random() < float(assumptions["maas_subscription"]["probability"]))

    return {
        "respondent_id": persona_num,
        "persona_id": f"P{persona_num:04d}",
        "female": female,
        "gender_text": "female" if female else "male",
        "age_band": age_band,
        "age_years": age_years,
        "income_band": income_band,
        "metropolis": metropolis,
        "metropolis_text": "metropolis" if metropolis else "large city",
        "household_cars": household_cars,
        "accessible_bikes": accessible_bikes,
        "pt_pass": pt_pass,
        "pt_pass_text": "holds a public transport pass" if pt_pass else "does not hold a public transport pass",
        "maas_subscription": maas_subscription,
        "maas_text": "has an active MaaS subscription" if maas_subscription else "does not have a MaaS subscription",
    }


def persona_system_prompt(persona: dict) -> str:
    return (
        "Act as one respondent in a transport SP survey. "
        "Use only the profile and card attributes. "
        "No extra assumptions. JSON only. "
        f"profile: gender={persona['gender_text']}; age={persona['age_years']}; age_band={persona['age_band']}; "
        f"income={persona['income_band']}; city_type={persona['metropolis_text']}; "
        f"hh_cars={persona['household_cars']}; hh_bikes={persona['accessible_bikes']}; "
        f"pt_pass={persona['pt_pass']}; maas={persona['maas_subscription']}."
    )


def build_session_prompt(questionnaire: pd.DataFrame) -> tuple[str, list[str]]:
    task_ids = questionnaire["task_id"].drop_duplicates().tolist()
    lines = [
        "Complete one short-distance survey session.",
        "Scenario: leisure, intra-urban, no luggage, ignore membership constraints.",
        "Labels: A=e-scooter, B=bike sharing, C=walking, D=private car.",
        "Fields: tt=travel_time, at=access_time, et=egress_time, pk=parking_search, av=availability, c=cost, sch=scheme, eng=engine, rg=range.",
        'Return one-line JSON only: {"choices":{"SD01":"A","SD02":"B","SD03":"C","SD04":"D","SD05":"A","SD06":"B"}}',
        "",
    ]

    for task_id, task_frame in questionnaire.groupby("task_id", sort=False):
        task_index = int(task_frame["task_index"].iloc[0])
        trip_length_km = float(task_frame["trip_length_km"].iloc[0])
        lines.append(f"{task_id} task={task_index} dist={trip_length_km:g}")
        for _, row in task_frame.sort_values("label").iterrows():
            scheme_text = clean_text(row["scheme"]).lower().replace("station-based", "sb").replace("free-floating", "ff")
            engine_text = clean_text(row["engine"]).lower().replace("pedelec", "ped")
            lines.append(
                " | ".join(
                    [
                        str(row["label"]),
                        f"tt={format_attribute(row['travel_time_min'], 'min')}",
                        f"at={format_attribute(row['access_time_min'], 'min')}",
                        f"et={format_attribute(row['egress_time_min'], 'min')}",
                        f"park={format_attribute(row['parking_search_time_min'], 'min')}",
                        f"avail={format_attribute(row['availability_pct'], '%')}",
                        f"cost={format_attribute(row['cost_eur'], 'EUR')}",
                        f"sch={scheme_text}",
                        f"eng={engine_text}",
                        f"range={format_attribute(row['range_km'], 'km')}",
                    ]
                )
            )
        lines.append("")
    return "\n".join(lines).strip(), task_ids


def build_wide_row(persona: dict, task_frame: pd.DataFrame, choice_label: str) -> dict:
    row = {
        "respondent_id": persona["respondent_id"],
        "persona_id": persona["persona_id"],
        "task_id": str(task_frame["task_id"].iloc[0]),
        "task_index": int(task_frame["task_index"].iloc[0]),
        "age": persona["age_years"],
        "hhcar": persona["household_cars"],
        "hhbike": persona["accessible_bikes"],
        "ptpass": persona["pt_pass"],
        "maas": persona["maas_subscription"],
        "choice": ALTERNATIVE_CODE.get(LABEL_TO_ALTERNATIVE_ID.get(choice_label, ""), 0),
        "avail_es": 1,
        "avail_bs": 1,
        "avail_walk": 1,
        "avail_car": 1,
    }

    for _, alternative in task_frame.iterrows():
        suffix = {
            "e_scooter": "es",
            "bikesharing": "bs",
            "walking": "walk",
            "private_car": "car",
        }[alternative["alternative_id"]]
        row[f"time_{suffix}"] = float(alternative["travel_time_min"])
        row[f"access_{suffix}"] = float(alternative["access_time_min"])
        row[f"egress_{suffix}"] = float(alternative["egress_time_min"])
        row[f"parking_{suffix}"] = float(alternative["parking_search_time_min"])
        row[f"cost_{suffix}"] = float(alternative["cost_eur"])
        row[f"availability_{suffix}"] = float(alternative["availability_pct"])
        row[f"freefloat_{suffix}"] = int(clean_text(alternative["scheme"]).lower() == "free-floating")
        row[f"range_{suffix}"] = float(alternative["range_km"])
        row[f"pedelec_{suffix}"] = int(clean_text(alternative["engine"]).lower() == "pedelec")
    return row


def initialize_output_files(output_dir: Path) -> None:
    (output_dir / "raw_interactions.jsonl").write_text("")
    (output_dir / "ai_choices_wide.csv").write_text("")
    (output_dir / "parsed_warmups.csv").write_text("")
    pd.DataFrame(
        columns=[
            "experiment_name",
            "respondent_id",
            "persona_id",
            "task_id",
            "task_index",
            "choice_label",
            "chosen_alternative_id",
            "chosen_alternative_name",
            "is_valid_choice",
            "female",
            "gender_text",
            "age_band",
            "age_years",
            "income_band",
            "metropolis",
            "metropolis_text",
            "household_cars",
            "accessible_bikes",
            "pt_pass",
            "pt_pass_text",
            "maas_subscription",
            "maas_text",
        ]
    ).to_csv(output_dir / "parsed_choices.csv", index=False)
    (output_dir / "respondent_transcripts.json").write_text("{}")
    (output_dir / "run_progress.json").write_text(json.dumps({"completed_respondents": 0, "last_respondent_id": 0}, indent=2))


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def append_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    header = path.stat().st_size == 0
    frame.to_csv(path, mode="a", header=header, index=False)


def load_completed_respondents(path: Path, tasks_per_respondent: int) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    if frame.empty:
        return set()
    grouped = frame.groupby("respondent_id").size()
    return set(grouped[grouped >= tasks_per_respondent].index.tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-respondents", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-respondents", type=int, default=None)
    args = parser.parse_args()

    output_dir = EXPERIMENT_DIR / "outputs"
    ensure_dir(output_dir)

    config = load_json(EXPERIMENT_DIR / "data" / "experiment_config.json")
    rules = load_json(EXPERIMENT_DIR / "data" / "persona_sampling_rules.json")
    questionnaire = pd.read_csv(EXPERIMENT_DIR / "data" / "krauss_sd_6cards_reconstruction.csv")
    questionnaire = questionnaire.rename(columns={"default_label": "label"})
    backend = OllamaBackend(config["backend"])

    if not args.resume:
        initialize_output_files(output_dir)

    questionnaire_manifest = {
        "experiment_name": config["experiment_name"],
        "questionnaire_file": "data/krauss_sd_6cards_reconstruction.csv",
        "survey_mode": "single_session_prompt",
        "warmup_count": 0,
        "n_tasks": int(questionnaire["task_id"].nunique()),
        "alternatives": ["e_scooter", "bikesharing", "walking", "private_car"],
        "note": "Six-task short-distance reconstruction block delivered in one survey-session prompt per respondent.",
    }
    (output_dir / "questionnaire_manifest.json").write_text(json.dumps(questionnaire_manifest, indent=2))

    rng = random.Random(int(config["master_seed"]))
    total_personas = args.n_respondents if args.n_respondents is not None else int(config["n_respondents"])
    personas = [build_persona(i, rng, rules) for i in range(1, total_personas + 1)]
    pd.DataFrame(personas).to_csv(output_dir / "persona_samples.csv", index=False)

    tasks_per_respondent = int(config["tasks_per_respondent"])
    completed = load_completed_respondents(output_dir / "parsed_choices.csv", tasks_per_respondent) if args.resume else set()
    transcripts = {}
    transcript_path = output_dir / "respondent_transcripts.json"
    if args.resume and transcript_path.exists():
        transcripts = json.loads(transcript_path.read_text())

    session_prompt, task_ids = build_session_prompt(questionnaire)
    personas_to_run = [p for p in personas if p["respondent_id"] not in completed]
    if args.max_respondents is not None:
        personas_to_run = personas_to_run[: args.max_respondents]

    for counter, persona in enumerate(personas_to_run, start=1):
        system_prompt = persona_system_prompt(persona)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": session_prompt},
        ]

        backend_response = backend.generate(messages, request_state={})
        response_text = backend_response["response_text"]
        parsed_choices = parse_session_choices(response_text, task_ids)
        valid_choice_count = sum(bool(parsed_choices[task_id]) for task_id in task_ids)

        raw_row = {
            "experiment_name": config["experiment_name"],
            "phase": "session_choices",
            "respondent_id": persona["respondent_id"],
            "persona_id": persona["persona_id"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "system_prompt": system_prompt,
            "user_prompt": session_prompt,
            "messages_sent": json.dumps(messages),
            "raw_response": backend_response["raw_text"],
            "response_text": response_text,
            "thinking_text": backend_response["thinking_text"],
            "parsed_choices_json": json.dumps(parsed_choices),
            "n_valid_choices": valid_choice_count,
            "done_reason": backend_response["metadata"].get("done_reason", ""),
            "total_duration": backend_response["metadata"].get("total_duration", 0),
            "prompt_eval_count": backend_response["metadata"].get("prompt_eval_count", 0),
            "eval_count": backend_response["metadata"].get("eval_count", 0),
            **persona,
        }
        append_jsonl(output_dir / "raw_interactions.jsonl", raw_row)

        parsed_choice_rows: list[dict] = []
        wide_rows: list[dict] = []
        for task_id, task_frame in questionnaire.groupby("task_id", sort=False):
            choice_label = parsed_choices.get(task_id, "")
            chosen_alternative_id = LABEL_TO_ALTERNATIVE_ID.get(choice_label, "")
            chosen_name = ""
            if chosen_alternative_id:
                chosen_name = task_frame.loc[task_frame["alternative_id"] == chosen_alternative_id, "alternative_name"].iloc[0]

            parsed_choice_rows.append(
                {
                    "experiment_name": config["experiment_name"],
                    "respondent_id": persona["respondent_id"],
                    "persona_id": persona["persona_id"],
                    "task_id": task_id,
                    "task_index": int(task_frame["task_index"].iloc[0]),
                    "choice_label": choice_label,
                    "chosen_alternative_id": chosen_alternative_id,
                    "chosen_alternative_name": chosen_name,
                    "is_valid_choice": int(bool(chosen_alternative_id)),
                    **persona,
                }
            )
            wide_rows.append(build_wide_row(persona, task_frame, choice_label))

        append_csv_rows(output_dir / "parsed_choices.csv", parsed_choice_rows)
        append_csv_rows(output_dir / "ai_choices_wide.csv", wide_rows)

        transcripts[persona["persona_id"]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": session_prompt},
            {"role": "assistant", "content": response_text},
        ]
        transcript_path.write_text(json.dumps(transcripts, indent=2))
        progress_payload = {
            "completed_respondents": len(completed) + counter,
            "last_respondent_id": persona["respondent_id"],
            "target_respondents": total_personas,
            "valid_choices_last_session": valid_choice_count,
        }
        (output_dir / "run_progress.json").write_text(json.dumps(progress_payload, indent=2))

        if counter % 10 == 0 or counter == len(personas_to_run):
            print(
                f"[progress] completed={len(completed) + counter}/{total_personas} "
                f"last_persona={persona['persona_id']} valid_choices={valid_choice_count}/6",
                flush=True,
            )


if __name__ == "__main__":
    main()
