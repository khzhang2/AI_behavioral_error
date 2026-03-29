from __future__ import annotations

from typing import Any


PURPOSE_TEXT = {
    1: "a commuting trip",
    2: "a shopping trip",
    3: "a business trip",
    4: "a leisure trip",
    5: "a trip returning from work",
    6: "a trip returning from shopping",
    7: "a trip returning from business",
    8: "a trip returning from leisure",
    9: "another kind of trip",
}

TICKET_TEXT = {
    0: "no special rail ticket",
    1: "a two-way ticket with a half-price card",
    2: "a one-way ticket with a half-price card",
    3: "a two-way full-price ticket",
    4: "a one-way full-price ticket",
    5: "a half-day ticket",
    6: "an annual season ticket",
    7: "an annual junior or senior season ticket",
    8: "a free-travel-after-7pm card",
    9: "a group ticket",
    10: "another ticket type",
}

WHO_TEXT = {
    0: "the payer is unknown",
    1: "you pay for the trip yourself",
    2: "your employer pays for the trip",
    3: "you and someone else split the payment",
}

LUGGAGE_TEXT = {
    0: "no luggage",
    1: "one piece of luggage",
    3: "several pieces of luggage",
}

AGE_TEXT = {
    1: "age 24 or younger",
    2: "age 25 to 39",
    3: "age 40 to 54",
    4: "age 55 to 65",
    5: "older than 65",
    6: "unknown age",
}

INCOME_TEXT = {
    0: "annual income under CHF 50,000",
    1: "annual income under CHF 50,000",
    2: "annual income between CHF 50,000 and CHF 100,000",
    3: "annual income above CHF 100,000",
    4: "unknown annual income",
}

SURVEY_TEXT = {
    0: "the survey stratum originally sampled on a train",
    1: "the survey stratum originally sampled in a car",
}


def _yes_no(flag: int | bool, yes_text: str, no_text: str) -> str:
    return yes_text if int(flag) == 1 else no_text


def _seat_text(value: int) -> str:
    if int(value) == 1:
        return "better seat configuration"
    return "regular seat configuration"


def _lookup(mapping: dict[int, str], value: Any, fallback_prefix: str) -> str:
    try:
        key = int(value)
    except (TypeError, ValueError):
        return fallback_prefix
    return mapping.get(key, f"{fallback_prefix} code {key}")


def build_system_prompt(persona: dict[str, Any]) -> str:
    sex_text = "a male traveler" if int(persona.get("MALE", 0)) == 1 else "a female traveler"
    age_text = _lookup(AGE_TEXT, persona.get("AGE"), "age")
    income_text = _lookup(INCOME_TEXT, persona.get("INCOME"), "income")
    luggage_text = _lookup(LUGGAGE_TEXT, persona.get("LUGGAGE"), "luggage")
    purpose_text = _lookup(PURPOSE_TEXT, persona.get("PURPOSE"), "trip purpose")
    ticket_text = _lookup(TICKET_TEXT, persona.get("TICKET"), "ticket")
    who_text = _lookup(WHO_TEXT, persona.get("WHO"), "payer")
    survey_text = _lookup(SURVEY_TEXT, persona.get("SURVEY"), "survey stratum")
    ga_text = _yes_no(
        persona.get("GA", 0),
        "You have a GA transit pass.",
        "You do not have a GA transit pass.",
    )
    first_class_text = _yes_no(
        persona.get("FIRST", 0),
        "You usually travel first class.",
        "You usually travel second class.",
    )
    car_text = _yes_no(
        persona.get("CAR_AV", 0),
        "A car is available for this trip whenever the card says car is available.",
        "A car is not available for this trip.",
    )

    return f"""You are {sex_text}, {age_text}, with {income_text}. You are making {purpose_text} and you have {luggage_text}. {first_class_text} {ga_text} {car_text}

You are traveling from origin zone code {int(persona.get("ORIGIN", 0))} to destination zone code {int(persona.get("DEST", 0))}. {who_text}. Your rail ticket is {ticket_text}. This case belongs to {survey_text}.

You know the following survey-specific facts.
- Train and Swissmetro are two different rail options.
- GA is a transit pass. If you have GA, it covers both Train and Swissmetro fares for this trip.
- Headway means the minutes between departures. Lower headway means more frequent service.
- Swissmetro seat configuration is a binary comfort attribute: 1 means the better seat configuration and 0 means the regular one.
- If a card says car is unavailable, you must not choose car.

You are answering a mobile stated-preference survey about the same trip context across 9 questions. Stay in this same persona for the whole conversation. Do not invent extra attitudes or outside facts. For every survey reply, return JSON only."""


def build_grounding_prompt(persona: dict[str, Any]) -> str:
    ga_rule = "train_and_swissmetro_fares_covered" if int(persona.get("GA", 0)) == 1 else "no_ga_coverage"
    car_default = bool(int(persona.get("CAR_AV", 0)) == 1)
    return (
        "Before the first question, confirm that you understand who you are. "
        "Return JSON only in exactly this schema:\n"
        '{'
        f'"persona_id":"{persona["respondent_id"]}",'
        f'"ga_rule":"{ga_rule}",'
        f'"car_available_default":{str(car_default).lower()},'
        '"ready":true'
        "}"
    )


def build_choice_prompt(
    task: dict[str, Any],
    task_position: int,
    total_tasks: int,
    previous_answers: list[str] | None = None,
) -> str:
    car_available = int(task.get("CAR_AV", 0)) == 1
    car_note = "available" if car_available else "unavailable"
    history_block = ""
    if previous_answers:
        joined = "; ".join(previous_answers)
        history_block = f"Your earlier answers in this same survey were: {joined}.\nKeep answering as the same person for the same trip.\n\n"
    return f"""Question {task_position} of {total_tasks} for the same trip. Choose exactly one option.

{history_block}Current choice card:

Option A: Train
- Travel time: {int(task["TRAIN_TT"])} minutes
- Travel cost: {int(task["TRAIN_CO"])} CHF
- Headway: {int(task["TRAIN_HE"])} minutes

Option B: Swissmetro
- Travel time: {int(task["SM_TT"])} minutes
- Travel cost: {int(task["SM_CO"])} CHF
- Headway: {int(task["SM_HE"])} minutes
- Seat configuration: {_seat_text(int(task["SM_SEATS"]))}

Option C: Car
- Travel time: {int(task["CAR_TT"])} minutes
- Travel cost: {int(task["CAR_CO"])} CHF
- Availability: {car_note}

Return JSON only like {{"choice_label":"A"}}."""
