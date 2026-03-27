from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_behavioral_error.backends.ollama import OllamaBackend  # noqa: E402
from ai_behavioral_error.parsing import parse_choice_label  # noqa: E402


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
    l = 2.718281828459045 ** (-mean)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= rng.random()
    value = k - 1
    return max(min_value, min(max_value, value))


def build_persona(persona_id: str, rng: random.Random, rules: dict) -> dict:
    exact = rules["exact_distributions"]
    approximations = rules["moment_matching_approximations"]

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

    return {
        "persona_id": persona_id,
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
    }


def persona_behavior_note(persona: dict) -> str:
    notes = []
    if persona["income_band"] in {"lt_500_eur", "500_1999_eur"}:
        notes.append("You are usually cost-conscious about everyday travel spending.")
    elif persona["income_band"] == "4000_plus_eur":
        notes.append("You are less constrained by small travel cost differences.")

    if persona["household_cars"] == 0:
        notes.append("There is no household car readily available to you.")
    elif persona["household_cars"] >= 2:
        notes.append("Household cars are readily available to you.")

    if persona["accessible_bikes"] == 0:
        notes.append("Bicycles are not readily available in your household.")
    elif persona["accessible_bikes"] >= 2:
        notes.append("Bicycles are readily available in your household.")

    if persona["pt_pass"] == 1:
        notes.append("You are accustomed to using public transport in daily life.")

    return " ".join(notes)


def persona_system_prompt(persona: dict) -> str:
    behavior_note = persona_behavior_note(persona)
    return (
        "You are one respondent in a transport stated-preference survey. "
        "Remain in the same persona for the whole conversation. "
        "Choose as this person would choose for a real intra-urban leisure trip in their city of residence. "
        "Use only the persona and the attributes shown on each choice card. "
        "Let the persona's mobility resources and budget background influence what feels convenient or attractive. "
        "Do not explain your answer. Do not add assumptions beyond the stated scenario. "
        "Return exactly one line of JSON in the form {\"choice\":\"A\"}. "
        f"Persona: {persona['gender_text']}, age {persona['age_years']} ({persona['age_band']}), "
        f"household income {persona['income_band']}, lives in a {persona['metropolis_text']}, "
        f"{persona['household_cars']} household cars, {persona['accessible_bikes']} accessible household bikes, "
        f"and {persona['pt_pass_text']}. "
        f"{behavior_note}"
    )


def format_attribute(value: object, unit: str) -> str:
    if pd.isna(value):
        return "-"
    if value in ("", None):
        return "-"
    numeric = float(value)
    if numeric == 0:
        return "-"
    if unit == "EUR":
        return f"{numeric:g} EUR"
    if unit == "%":
        return f"{numeric:g}%"
    if unit == "km":
        return f"{numeric:g} km"
    return f"{numeric:g} {unit}"


def clean_text(value: object) -> str:
    if pd.isna(value):
        return "-"
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text else "-"


def build_task_prompt(task_frame: pd.DataFrame, task_index: int, total_tasks: int) -> str:
    trip_length_km = float(task_frame["trip_length_km"].iloc[0])
    lines = [
        "Scenario:",
        "- This is a leisure trip within your city of residence.",
        "- You carry no luggage.",
        "- Ignore membership constraints for shared services.",
        "",
        f"Choice task {task_index} of {total_tasks}",
        f"Trip length: {trip_length_km:g} km",
        "",
        "Alternatives:",
    ]

    for _, row in task_frame.sort_values("label").iterrows():
        lines.extend(
            [
                f"{row['label']}. {row['alternative_name']}",
                f"  travel_time: {format_attribute(row['travel_time_min'], 'min')}",
                f"  access_time: {format_attribute(row['access_time_min'], 'min')}",
                f"  egress_time: {format_attribute(row['egress_time_min'], 'min')}",
                f"  parking_search_time: {format_attribute(row['parking_search_time_min'], 'min')}",
                f"  availability: {format_attribute(row['availability_pct'], '%')}",
                f"  cost: {format_attribute(row['cost_eur'], 'EUR')}",
                f"  scheme: {clean_text(row['scheme'])}",
                f"  engine: {clean_text(row['engine'])}",
                f"  range: {format_attribute(row['range_km'], 'km')}",
                "",
            ]
        )

    lines.extend(
        [
            "Choose exactly one option.",
            "Return exactly one line of JSON like {\"choice\":\"B\"}.",
        ]
    )
    return "\n".join(lines)


def build_wide_row(persona: dict, task_frame: pd.DataFrame, choice_label: str) -> dict:
    row = {
        "persona_id": persona["persona_id"],
        "age_years": persona["age_years"],
        "female": persona["female"],
        "metropolis": persona["metropolis"],
        "household_cars": persona["household_cars"],
        "accessible_bikes": persona["accessible_bikes"],
        "pt_pass": persona["pt_pass"],
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
        row[f"freefloat_{suffix}"] = int(clean_text(alternative["scheme"]) == "free-floating")
        row[f"pedelec_{suffix}"] = int(clean_text(alternative["engine"]).lower() == "pedelec")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-respondents", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    ensure_dir(EXPERIMENT_DIR / "outputs")

    config = load_json(EXPERIMENT_DIR / "data" / "experiment_config.json")
    rules = load_json(EXPERIMENT_DIR / "data" / "persona_sampling_rules.json")
    questionnaire = pd.read_csv(EXPERIMENT_DIR / "data" / "krauss_sd_5cards_reconstruction.csv")
    backend = OllamaBackend(config["backend"])

    questionnaire_manifest = {
        "experiment_name": config["experiment_name"],
        "questionnaire_file": "data/krauss_sd_5cards_reconstruction.csv",
        "n_tasks": int(questionnaire["task_id"].nunique()),
        "alternatives": ["e_scooter", "bikesharing", "walking", "private_car"],
        "note": "Five-task reconstruction block derived from the paper's short-distance attribute universe.",
    }
    (EXPERIMENT_DIR / "outputs" / "questionnaire_manifest.json").write_text(json.dumps(questionnaire_manifest, indent=2))

    rng = random.Random(int(config["master_seed"]))
    total_personas = args.n_respondents if args.n_respondents is not None else int(config["n_respondents"])
    personas = [build_persona(f"P{i:03d}", rng, rules) for i in range(1, total_personas + 1)]
    personas_frame = pd.DataFrame(personas)
    personas_frame.to_csv(EXPERIMENT_DIR / "outputs" / "persona_samples.csv", index=False)

    raw_rows: list[dict] = []
    parsed_rows: list[dict] = []
    wide_rows: list[dict] = []
    transcripts: dict[str, list[dict]] = {}

    grouped_tasks = list(questionnaire.groupby("task_id", sort=False))
    if args.max_tasks is not None:
        grouped_tasks = grouped_tasks[: args.max_tasks]
    total_tasks = len(grouped_tasks)

    for persona in personas:
        messages = [{"role": "system", "content": persona_system_prompt(persona)}]
        transcripts[persona["persona_id"]] = [{"role": "system", "content": messages[0]["content"]}]

        for task_position, (task_id, task_frame) in enumerate(grouped_tasks, start=1):
            user_prompt = build_task_prompt(task_frame, task_position, total_tasks)
            call_messages = messages + [{"role": "user", "content": user_prompt}]

            backend_response = backend.generate(call_messages, request_state={})
            response_text = backend_response["response_text"]
            parsed_choice_label = parse_choice_label(response_text)
            chosen_alternative_id = LABEL_TO_ALTERNATIVE_ID.get(parsed_choice_label, "")
            chosen_name = ""
            if chosen_alternative_id:
                chosen_name = task_frame.loc[task_frame["alternative_id"] == chosen_alternative_id, "alternative_name"].iloc[0]

            raw_rows.append(
                {
                    "experiment_name": config["experiment_name"],
                    "persona_id": persona["persona_id"],
                    "task_id": task_id,
                    "task_index": int(task_frame["task_index"].iloc[0]),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "system_prompt": messages[0]["content"],
                    "user_prompt": user_prompt,
                    "messages_sent": json.dumps(call_messages),
                    "raw_response": backend_response["raw_text"],
                    "response_text": response_text,
                    "thinking_text": backend_response["thinking_text"],
                    "parsed_choice_label": parsed_choice_label,
                    "chosen_alternative_id": chosen_alternative_id,
                    "chosen_alternative_name": chosen_name,
                    "is_valid_choice": int(bool(chosen_alternative_id)),
                    "done_reason": backend_response["metadata"].get("done_reason", ""),
                    "total_duration": backend_response["metadata"].get("total_duration", 0),
                    "prompt_eval_count": backend_response["metadata"].get("prompt_eval_count", 0),
                    "eval_count": backend_response["metadata"].get("eval_count", 0),
                    **persona,
                }
            )

            parsed_rows.append(
                {
                    "experiment_name": config["experiment_name"],
                    "persona_id": persona["persona_id"],
                    "task_id": task_id,
                    "task_index": int(task_frame["task_index"].iloc[0]),
                    "choice_label": parsed_choice_label,
                    "chosen_alternative_id": chosen_alternative_id,
                    "chosen_alternative_name": chosen_name,
                    "is_valid_choice": int(bool(chosen_alternative_id)),
                    **persona,
                }
            )

            wide_rows.append(build_wide_row(persona, task_frame, parsed_choice_label))

            messages.append({"role": "user", "content": user_prompt})
            messages.append({"role": "assistant", "content": response_text})
            transcripts[persona["persona_id"]].append({"role": "user", "content": user_prompt})
            transcripts[persona["persona_id"]].append({"role": "assistant", "content": response_text})

    pd.DataFrame(raw_rows).to_json(EXPERIMENT_DIR / "outputs" / "raw_interactions.jsonl", orient="records", lines=True)
    pd.DataFrame(parsed_rows).to_csv(EXPERIMENT_DIR / "outputs" / "parsed_choices.csv", index=False)
    pd.DataFrame(wide_rows).to_csv(EXPERIMENT_DIR / "outputs" / "ai_choices_wide.csv", index=False)
    (EXPERIMENT_DIR / "outputs" / "respondent_transcripts.json").write_text(json.dumps(transcripts, indent=2))


if __name__ == "__main__":
    main()
