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
    "carsharing": 5,
    "ridepooling": 6,
    "public_transport": 7,
}

LABEL_TO_ALTERNATIVE_ID = {
    "SD": {
        "A": "e_scooter",
        "B": "bikesharing",
        "C": "walking",
        "D": "private_car",
    },
    "MD": {
        "A": "carsharing",
        "B": "ridepooling",
        "C": "public_transport",
        "D": "private_car",
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if pd.isna(value) or value in ("", None):
        return "-"
    numeric = float(value)
    if unit == "EUR":
        return f"{numeric:g} EUR"
    if unit == "%":
        return f"{numeric:g}%"
    if unit == "km":
        return f"{numeric:g} km"
    if unit == "count":
        return f"{numeric:g}"
    return f"{numeric:g} {unit}"


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
        "city_type_text": "metropolis" if metropolis else "large city",
        "household_cars": household_cars,
        "accessible_bikes": accessible_bikes,
        "pt_pass": pt_pass,
        "pt_pass_text": "holds a public transport pass" if pt_pass else "does not hold a public transport pass",
        "maas_subscription": maas_subscription,
        "maas_text": "has an active MaaS subscription" if maas_subscription else "does not have a MaaS subscription",
    }


def assign_subsamples(personas: list[dict], master_seed: int, blocks_per_subsample: int) -> list[dict]:
    respondent_ids = [persona["respondent_id"] for persona in personas]
    rng = random.Random(f"{master_seed}-subsample-balance")
    rng.shuffle(respondent_ids)

    n_sd = round(len(personas) * 0.5)
    sd_ids = set(respondent_ids[:n_sd])
    groups = {
        "SD": [persona for persona in personas if persona["respondent_id"] in sd_ids],
        "MD": [persona for persona in personas if persona["respondent_id"] not in sd_ids],
    }

    for subsample, members in groups.items():
        rng = random.Random(f"{master_seed}-{subsample}-blocks")
        rng.shuffle(members)
        for index, persona in enumerate(members):
            persona["subsample"] = subsample
            persona["assigned_block_id"] = (index % blocks_per_subsample) + 1

    personas.sort(key=lambda row: row["respondent_id"])
    return personas


def persona_system_prompt(survey_briefing: str, persona: dict) -> str:
    dossier = "\n".join(
        [
            "Respondent dossier:",
            f"- persona_id: {persona['persona_id']}",
            f"- gender: {persona['gender_text']}",
            f"- age_years: {persona['age_years']}",
            f"- age_band: {persona['age_band']}",
            f"- income_band: {persona['income_band']}",
            f"- city_type: {persona['city_type_text']}",
            f"- household_cars: {persona['household_cars']}",
            f"- accessible_bikes: {persona['accessible_bikes']}",
            f"- pt_pass: {persona['pt_pass']}",
            f"- maas_subscription: {persona['maas_subscription']}",
            f"- assigned_subsample: {persona['subsample']}",
        ]
    )
    return (
        "You are one respondent in a stated-preference transport survey.\n\n"
        f"{survey_briefing}\n\n"
        f"{dossier}\n\n"
        "Stay in the same persona for the full conversation. "
        "Use only the respondent dossier and the attributes displayed on each card. "
        "Do not invent unstated preferences, experiences, or background facts. "
        "Ignore whether this person is currently a member of any shared service. "
        "When asked for a grounding turn, return strict JSON only. "
        "When asked for a formal choice, return strict JSON only."
    )


def build_grounding_prompt() -> str:
    return (
        "Before the first choice task, restate the respondent dossier as strict JSON only.\n"
        "Required keys: persona_id, gender, age_years, age_band, income_band, city_type, "
        "household_cars, accessible_bikes, pt_pass, maas_subscription, assigned_subsample, ready_for_survey.\n"
        "Set ready_for_survey to true. Do not add extra text."
    )


def build_sd_task_prompt(task_frame: pd.DataFrame, task_position: int, total_tasks: int) -> str:
    lines = [
        "Scenario:",
        "- This is an intra-urban leisure trip in your city of residence.",
        "- You carry no luggage.",
        "- Ignore membership constraints for shared services.",
        "",
        f"Choice task {task_position} of {total_tasks}",
        f"Card id: {task_frame['task_id'].iloc[0]}",
        f"Trip length: {float(task_frame['trip_length_km'].iloc[0]):g} km",
        "",
        "Labels: A = e-scooter sharing, B = bike sharing, C = walking, D = private car.",
        "Alternatives:",
    ]

    for _, row in task_frame.sort_values("display_label").iterrows():
        lines.extend(
            [
                f"{row['display_label']}. {row['alternative_name']}",
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
            "Choose exactly one option for this task.",
            'Return exactly one line of JSON like {"choice":"B"}.',
        ]
    )
    return "\n".join(lines)


def build_md_task_prompt(task_frame: pd.DataFrame, task_position: int, total_tasks: int) -> str:
    lines = [
        "Scenario:",
        "- This is an intra-urban leisure trip in your city of residence.",
        "- You carry no luggage.",
        "- Ignore membership constraints for shared services.",
        "",
        f"Choice task {task_position} of {total_tasks}",
        f"Card id: {task_frame['task_id'].iloc[0]}",
        f"Trip length: {float(task_frame['trip_length_km'].iloc[0]):g} km",
        "",
        "Labels: A = carsharing, B = ridepooling, C = public transport, D = private car.",
        "Alternatives:",
    ]

    for _, row in task_frame.sort_values("display_label").iterrows():
        lines.extend(
            [
                f"{row['display_label']}. {row['alternative_name']}",
                f"  travel_time: {format_attribute(row['travel_time_min'], 'min')}",
                f"  access_time: {format_attribute(row['access_time_min'], 'min')}",
                f"  waiting_time: {format_attribute(row['waiting_time_min'], 'min')}",
                f"  egress_time: {format_attribute(row['egress_time_min'], 'min')}",
                f"  detour_time: {format_attribute(row['detour_time_min'], 'min')}",
                f"  parking_search_time: {format_attribute(row['parking_search_time_min'], 'min')}",
                f"  cost: {format_attribute(row['cost_eur'], 'EUR')}",
                f"  scheme: {clean_text(row['scheme'])}",
                f"  crowding: {format_attribute(row['crowding_pct'], '%')}",
                f"  transfers: {format_attribute(row['transfer_count'], 'count')}",
                "",
            ]
        )

    lines.extend(
        [
            "Choose exactly one option for this task.",
            'Return exactly one line of JSON like {"choice":"C"}.',
        ]
    )
    return "\n".join(lines)


def build_task_prompt(task_frame: pd.DataFrame, task_position: int, total_tasks: int) -> str:
    if str(task_frame["subsample"].iloc[0]) == "SD":
        return build_sd_task_prompt(task_frame, task_position, total_tasks)
    return build_md_task_prompt(task_frame, task_position, total_tasks)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def append_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    header = not path.exists() or path.stat().st_size == 0
    frame.to_csv(path, mode="a", header=header, index=False)


def initialize_output_files(output_dir: Path) -> None:
    (output_dir / "raw_interactions.jsonl").write_text("", encoding="utf-8")
    (output_dir / "parsed_choices.csv").write_text("", encoding="utf-8")
    (output_dir / "pooled_choices_long.csv").write_text("", encoding="utf-8")
    (output_dir / "respondent_transcripts.json").write_text("{}", encoding="utf-8")
    (output_dir / "run_progress.json").write_text(
        json.dumps({"completed_respondents": 0, "last_respondent_id": 0}, indent=2),
        encoding="utf-8",
    )


def load_completed_respondents(path: Path, tasks_per_respondent: int) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    if frame.empty or "respondent_id" not in frame.columns:
        return set()
    grouped = frame.groupby("respondent_id").size()
    return set(grouped[grouped >= tasks_per_respondent].index.tolist())


def task_sequence_for_persona(design_frame: pd.DataFrame, persona: dict, master_seed: int) -> list[tuple[str, pd.DataFrame]]:
    subset = design_frame.loc[
        (design_frame["subsample"] == persona["subsample"])
        & (design_frame["block_id"] == persona["assigned_block_id"])
    ].copy()
    active = subset.loc[subset["is_available"] == 1]
    ordered = list(active.groupby("task_id", sort=False))
    rng = random.Random(f"{master_seed}-{persona['respondent_id']}-task-order")
    rng.shuffle(ordered)
    return ordered


def build_parsed_choice_row(persona: dict, task_frame: pd.DataFrame, task_position: int, choice_label: str) -> dict:
    subsample = str(task_frame["subsample"].iloc[0])
    chosen_alternative_id = LABEL_TO_ALTERNATIVE_ID[subsample].get(choice_label, "")
    chosen_name = ""
    if chosen_alternative_id:
        chosen_name = task_frame.loc[task_frame["alternative_id"] == chosen_alternative_id, "alternative_name"].iloc[0]
    return {
        "experiment_name": EXPERIMENT_DIR.name,
        "respondent_id": persona["respondent_id"],
        "persona_id": persona["persona_id"],
        "subsample": subsample,
        "block_id": int(task_frame["block_id"].iloc[0]),
        "task_id": str(task_frame["task_id"].iloc[0]),
        "task_index_within_subsample": int(task_frame["task_index_within_subsample"].iloc[0]),
        "task_in_block": int(task_frame["task_in_block"].iloc[0]),
        "presented_task_position": task_position,
        "choice_label": choice_label,
        "chosen_alternative_id": chosen_alternative_id,
        "chosen_alternative_name": chosen_name,
        "choice_code": ALTERNATIVE_CODE.get(chosen_alternative_id, 0),
        "is_valid_choice": int(bool(chosen_alternative_id)),
        **persona,
    }


def build_long_rows(persona: dict, task_frame: pd.DataFrame, choice_label: str, chosen_alternative_id: str) -> list[dict]:
    choice_code = ALTERNATIVE_CODE.get(chosen_alternative_id, 0)
    rows = []
    for _, row in task_frame.iterrows():
        rows.append(
            {
                "respondent_id": persona["respondent_id"],
                "persona_id": persona["persona_id"],
                "subsample": str(row["subsample"]),
                "block_id": int(row["block_id"]),
                "task_id": str(row["task_id"]),
                "task_index_within_subsample": int(row["task_index_within_subsample"]),
                "task_in_block": int(row["task_in_block"]),
                "chosen_label": choice_label,
                "chosen_alternative_id": chosen_alternative_id,
                "choice_code": choice_code,
                "choice": int(row["alternative_id"] == chosen_alternative_id),
                "is_valid_choice": int(bool(chosen_alternative_id)),
                "alternative_id": str(row["alternative_id"]),
                "alternative_name": str(row["alternative_name"]),
                "display_label": str(row["display_label"]),
                "is_available": int(row["is_available"]),
                "trip_length_km": row["trip_length_km"],
                "travel_time_min": row["travel_time_min"],
                "access_time_min": row["access_time_min"],
                "waiting_time_min": row["waiting_time_min"],
                "egress_time_min": row["egress_time_min"],
                "detour_time_min": row["detour_time_min"],
                "parking_search_time_min": row["parking_search_time_min"],
                "availability_pct": row["availability_pct"],
                "cost_eur": row["cost_eur"],
                "scheme": row["scheme"],
                "engine": row["engine"],
                "range_km": row["range_km"],
                "crowding_pct": row["crowding_pct"],
                "transfer_count": row["transfer_count"],
                "scheme_free_floating": int(row["scheme_free_floating"]),
                "scheme_hybrid": int(row["scheme_hybrid"]),
                "pedelec": int(row["pedelec"]),
                "avail_es": int(row["avail_es"]),
                "avail_bs": int(row["avail_bs"]),
                "avail_walk": int(row["avail_walk"]),
                "avail_car": int(row["avail_car"]),
                "avail_cs": int(row["avail_cs"]),
                "avail_rp": int(row["avail_rp"]),
                "avail_pt": int(row["avail_pt"]),
                "provenance": row["provenance"],
                "female": persona["female"],
                "gender_text": persona["gender_text"],
                "age_band": persona["age_band"],
                "age_years": persona["age_years"],
                "income_band": persona["income_band"],
                "metropolis": persona["metropolis"],
                "city_type_text": persona["city_type_text"],
                "household_cars": persona["household_cars"],
                "accessible_bikes": persona["accessible_bikes"],
                "pt_pass": persona["pt_pass"],
                "maas_subscription": persona["maas_subscription"],
            }
        )
    return rows


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
    assumptions = load_json(EXPERIMENT_DIR / "data" / "public_replication_assumptions.json")
    design = pd.read_csv(EXPERIMENT_DIR / "data" / "pooled_choice_sets_public_materials.csv")
    survey_briefing = (EXPERIMENT_DIR / "data" / "survey_instrument_en.md").read_text(encoding="utf-8")
    backend = OllamaBackend(config["backend"])

    if not args.resume:
        initialize_output_files(output_dir)

    questionnaire_manifest = {
        "experiment_name": config["experiment_name"],
        "replication_standard": config["replication_standard"],
        "subsample_design": "pooled_sd_md",
        "questionnaire_file": "data/pooled_choice_sets_public_materials.csv",
        "survey_briefing_file": "data/survey_instrument_en.md",
        "survey_mode": "multi_turn_panel_prompt_with_grounding_turn",
        "n_tasks_per_respondent": config["tasks_per_respondent"],
        "blocks_per_subsample": config["blocks_per_subsample"],
        "alternatives_by_subsample": {
            "SD": ["e_scooter", "bikesharing", "walking", "private_car"],
            "MD": ["carsharing", "ridepooling", "public_transport", "private_car"],
        },
        "task_ordering": "respondent_specific_random_order_within_assigned_reconstructed_block",
        "design_provenance": {
            "attribute_levels": "public_exact",
            "choice_set_combinations": "inferred_from_public",
        },
        "missing_covariate_assumptions": assumptions["missing_covariates"],
        "subsample_assignment": assumptions["subsample_assignment"],
    }
    (output_dir / "questionnaire_manifest.json").write_text(json.dumps(questionnaire_manifest, indent=2), encoding="utf-8")

    rng = random.Random(int(config["master_seed"]))
    total_personas = args.n_respondents if args.n_respondents is not None else int(config["n_respondents"])
    personas = [build_persona(i, rng, rules) for i in range(1, total_personas + 1)]
    personas = assign_subsamples(personas, int(config["master_seed"]), int(config["blocks_per_subsample"]))
    pd.DataFrame(personas).to_csv(output_dir / "persona_samples.csv", index=False)

    tasks_per_respondent = int(config["tasks_per_respondent"])
    completed = load_completed_respondents(output_dir / "parsed_choices.csv", tasks_per_respondent) if args.resume else set()
    transcript_path = output_dir / "respondent_transcripts.json"
    transcripts = {}
    if args.resume and transcript_path.exists():
        transcripts = json.loads(transcript_path.read_text(encoding="utf-8"))

    personas_to_run = [persona for persona in personas if persona["respondent_id"] not in completed]
    if args.max_respondents is not None:
        personas_to_run = personas_to_run[: args.max_respondents]

    for counter, persona in enumerate(personas_to_run, start=1):
        system_prompt = persona_system_prompt(survey_briefing, persona)
        messages = [{"role": "system", "content": system_prompt}]
        grounding_prompt = build_grounding_prompt()
        grounding_messages = messages + [{"role": "user", "content": grounding_prompt}]
        grounding_response = backend.generate(
            grounding_messages,
            request_state={
                "respondent_id": persona["respondent_id"],
                "persona_id": persona["persona_id"],
                "phase": "grounding",
            },
        )
        grounding_payload = parse_grounding_payload(grounding_response["response_text"])
        append_jsonl(
            output_dir / "raw_interactions.jsonl",
            {
                "experiment_name": config["experiment_name"],
                "phase": "grounding",
                "respondent_id": persona["respondent_id"],
                "persona_id": persona["persona_id"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "system_prompt": system_prompt,
                "user_prompt": grounding_prompt,
                "messages_sent": json.dumps(grounding_messages),
                "raw_response": grounding_response["raw_text"],
                "response_text": grounding_response["response_text"],
                "thinking_text": grounding_response["thinking_text"],
                "grounding_is_valid_json": int(grounding_payload is not None),
                "done_reason": grounding_response["metadata"].get("done_reason", ""),
                "total_duration": grounding_response["metadata"].get("total_duration", 0),
                "prompt_eval_count": grounding_response["metadata"].get("prompt_eval_count", 0),
                "eval_count": grounding_response["metadata"].get("eval_count", 0),
                **persona,
            },
        )
        messages.append({"role": "user", "content": grounding_prompt})
        messages.append({"role": "assistant", "content": grounding_response["response_text"]})

        parsed_choice_rows: list[dict] = []
        long_rows: list[dict] = []
        ordered_tasks = task_sequence_for_persona(design, persona, int(config["master_seed"]))

        for task_position, (task_id, active_task_frame) in enumerate(ordered_tasks, start=1):
            full_task_frame = design.loc[design["task_id"] == task_id].copy()
            user_prompt = build_task_prompt(active_task_frame, task_position, len(ordered_tasks))
            call_messages = messages + [{"role": "user", "content": user_prompt}]
            backend_response = backend.generate(
                call_messages,
                request_state={
                    "respondent_id": persona["respondent_id"],
                    "persona_id": persona["persona_id"],
                    "task_id": task_id,
                    "presented_task_position": task_position,
                    "subsample": persona["subsample"],
                },
            )
            response_text = backend_response["response_text"]
            choice_label = parse_choice_label(response_text)
            parsed_choice = build_parsed_choice_row(persona, active_task_frame, task_position, choice_label)
            parsed_choice_rows.append(parsed_choice)
            long_rows.extend(build_long_rows(persona, full_task_frame, choice_label, parsed_choice["chosen_alternative_id"]))

            append_jsonl(
                output_dir / "raw_interactions.jsonl",
                {
                    "experiment_name": config["experiment_name"],
                    "phase": "choice",
                    "respondent_id": persona["respondent_id"],
                    "persona_id": persona["persona_id"],
                    "subsample": persona["subsample"],
                    "block_id": persona["assigned_block_id"],
                    "task_id": task_id,
                    "task_index_within_subsample": int(active_task_frame["task_index_within_subsample"].iloc[0]),
                    "task_in_block": int(active_task_frame["task_in_block"].iloc[0]),
                    "presented_task_position": task_position,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "messages_sent": json.dumps(call_messages),
                    "raw_response": backend_response["raw_text"],
                    "response_text": response_text,
                    "thinking_text": backend_response["thinking_text"],
                    "parsed_choice_label": choice_label,
                    "chosen_alternative_id": parsed_choice["chosen_alternative_id"],
                    "chosen_alternative_name": parsed_choice["chosen_alternative_name"],
                    "is_valid_choice": parsed_choice["is_valid_choice"],
                    "done_reason": backend_response["metadata"].get("done_reason", ""),
                    "total_duration": backend_response["metadata"].get("total_duration", 0),
                    "prompt_eval_count": backend_response["metadata"].get("prompt_eval_count", 0),
                    "eval_count": backend_response["metadata"].get("eval_count", 0),
                    **persona,
                },
            )

            messages.append({"role": "user", "content": user_prompt})
            messages.append({"role": "assistant", "content": response_text})

        append_csv_rows(output_dir / "parsed_choices.csv", parsed_choice_rows)
        append_csv_rows(output_dir / "pooled_choices_long.csv", long_rows)

        transcripts[persona["persona_id"]] = messages
        transcript_path.write_text(json.dumps(transcripts, indent=2), encoding="utf-8")
        valid_choice_count = sum(int(row["is_valid_choice"]) for row in parsed_choice_rows)
        progress_payload = {
            "completed_respondents": len(completed) + counter,
            "last_respondent_id": persona["respondent_id"],
            "target_respondents": total_personas,
            "valid_choices_last_respondent": valid_choice_count,
        }
        (output_dir / "run_progress.json").write_text(json.dumps(progress_payload, indent=2), encoding="utf-8")

        if counter % 10 == 0 or counter == len(personas_to_run):
            print(
                f"[progress] completed={len(completed) + counter}/{total_personas} "
                f"last_persona={persona['persona_id']} subsample={persona['subsample']} "
                f"block={persona['assigned_block_id']} valid_choices={valid_choice_count}/{tasks_per_respondent}",
                flush=True,
            )


if __name__ == "__main__":
    main()
