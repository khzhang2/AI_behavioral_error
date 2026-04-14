from __future__ import annotations

from typing import Any

from optima_common import INDICATOR_TEXT, TASK_ATTRIBUTE_OPTIONS


LIKERT_SCALE_TEXT = (
    "Use this 1-5 scale: 1 means strongly disagree, 2 means disagree, 3 means neutral, "
    "4 means agree, and 5 means strongly agree."
)


def build_system_prompt(persona: dict[str, Any], prompt_arm: str, prompt_family: str) -> str:
    family_text = (
        "Answer briefly and consistently across the full survey."
        if prompt_family == "concise"
        else "Answer naturally, but stay consistent across the full survey and keep the same persona throughout."
    )
    arm_text = (
        "Most travel tasks show the actual mode names."
        if prompt_arm == "semantic_arm"
        else "Most travel tasks hide the mode names and present the options neutrally."
    )
    return f"""You are {persona['sex_text']}, {persona['age_text']}, with {persona['income_text']}. You are answering as the same traveler for one complete mobile survey session, and you are currently making {persona['trip_purpose_text']}. {persona['car_availability_text']}. Your household has {int(persona['NbHousehold'])} people, {int(persona['NbChild'])} children, {int(persona['NbCar'])} cars, and {int(persona['NbBicy'])} bicycles.

You know the following facts about yourself.
- Your education level is best summarized as: {persona['education_text']}.
- Your parents relied on the car more than the train: {"yes" if int(persona['car_oriented_parents']) == 1 else "no"}.
- As a child, you lived in a suburb: {"yes" if int(persona['childSuburb']) == 1 else "no"}.
- As a child, you lived in a city-center area: {"yes" if int(persona['city_center_as_kid']) == 1 else "no"}.

You are answering a stated-preference transport survey.
- PT means public transport.
- CAR means driving a car.
- SLOW_MODES means walking, cycling, and other non-motorized modes.
- PT travel time excludes the waiting time shown on the next line.
- Waiting time means the time spent waiting before boarding public transport.
- Availability means whether that option can actually be used for this trip.
- Some later tasks deliberately test whether your answers are stable under wording, label, order, dominance, or monotonicity changes.
- {arm_text}
- {family_text}
- First you confirm the persona with a short JSON reply.
- Then you answer seven attitude statements one by one.
- Then you answer sixteen travel-choice questions one by one.
- Every answer must be JSON only.
"""


def build_grounding_prompt(persona: dict[str, Any]) -> str:
    return (
        "Confirm that you understand who you are for this survey. Return JSON only in exactly this schema:\n"
        "{"
        f"\"persona_id\":\"{persona['respondent_id']}\","
        f"\"trip_purpose\":\"{persona['trip_purpose_text']}\","
        f"\"car_available\":{str(bool(int(persona['CAR_AVAILABLE']))).lower()},"
        "\"ready\":true"
        "}"
    )


def build_attitude_prompt(indicator_name: str, question_index: int, total_questions: int, previous_answers: list[str]) -> str:
    history = ""
    if previous_answers:
        history = "Earlier answers in this survey: " + "; ".join(previous_answers) + "\n\n"
    statement = INDICATOR_TEXT[indicator_name]
    return f"""Question {question_index} of {total_questions}. {history}Please rate how much this statement fits you.

Statement:
"{statement}"

{LIKERT_SCALE_TEXT}

Return JSON only like {{"indicator_value":4}}."""


def option_lines(task_row: dict[str, Any]) -> list[str]:
    rows = []
    for display_label in ["A", "B", "C"]:
        alternative = task_row[f"display_{display_label}_alt"]
        header = f"Option {display_label}"
        if int(task_row["semantic_labels"]) == 1:
            header += f": {alternative}"
        rows.append(header)
        if alternative == "PT":
            pt_travel_time = float(task_row.get("TimePT_non_wait", max(float(task_row["TimePT"]) - float(task_row["WaitingTimePT"]), 0.0)))
            rows.append(f"- Travel time: {int(round(pt_travel_time))} minutes")
            rows.append(f"- Waiting time: {int(round(float(task_row['WaitingTimePT'])))} minutes")
            rows.append(f"- Marginal cost: CHF {float(task_row['MarginalCostPT']):.2f}")
        elif alternative == "CAR":
            availability = "available" if int(task_row["CAR_AVAILABLE"]) == 1 else "unavailable"
            rows.append(f"- Travel time: {int(round(float(task_row['TimeCar'])))} minutes")
            rows.append(f"- Car cost: CHF {float(task_row['CostCarCHF']):.2f}")
            rows.append(f"- Availability: {availability}")
        else:
            rows.append(f"- Distance: {float(task_row['distance_km']):.1f} km")
            rows.append("- Availability: available")
        rows.append("")
    return rows


def task_intro(task_row: dict[str, Any], prompt_family: str) -> str:
    if task_row["manipulation_type"] == "paraphrase":
        if prompt_family == "concise":
            return "Please make the same kind of transport decision, but this version states the attributes in different words."
        return "Please make the same transport decision again, but this version describes the very same attributes in slightly different language."
    if prompt_family == "concise":
        return "Choose exactly one travel option for this trip."
    return "Imagine that you must decide now which option you would actually use for this trip, and choose exactly one option."


def task_instruction(task_row: dict[str, Any]) -> str:
    if task_row["task_role"] == "dominance":
        return (
            "This task may contain a clearly dominated option. Decide whether you think one option is clearly worse than another on the displayed burden attributes."
        )
    if task_row["task_role"] == "monotonicity":
        return "One option may have become clearly less attractive than before. Keep your answer consistent with the displayed attributes."
    return "Respond with a structured judgment, not a long explanation."


def build_task_prompt(
    task_row: dict[str, Any],
    question_index: int,
    total_questions: int,
    previous_answers: list[str],
) -> str:
    history = ""
    if previous_answers:
        history = "Earlier answers in this survey: " + "; ".join(previous_answers) + "\n\n"
    lines = option_lines(task_row)
    top_attr_text = ", ".join(TASK_ATTRIBUTE_OPTIONS)
    return f"""Question {question_index} of {total_questions}. {history}{task_intro(task_row, str(task_row['prompt_family']))}

{task_instruction(task_row)}

{chr(10).join(lines)}Return JSON only in exactly this schema:
{{
  "choice_label": "A",
  "confidence": 4,
  "top_attributes": ["travel_time", "cost"],
  "dominated_option_seen": false
}}

The allowed top_attributes are: {top_attr_text}.
- confidence must be an integer from 1 to 5.
- top_attributes must contain exactly two different labels from the allowed list.
- dominated_option_seen must be true or false."""
