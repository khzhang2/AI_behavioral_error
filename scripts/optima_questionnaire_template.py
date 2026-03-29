from __future__ import annotations

from typing import Any

from optima_common import INDICATOR_TEXT


LIKERT_SCALE_TEXT = (
    "Use this 1-6 scale: 1=strongly disagree, 2=disagree, 3=slightly disagree or mixed, "
    "4=slightly agree or mixed, 5=strongly agree, 6=not applicable."
)


def build_system_prompt(persona: dict[str, Any]) -> str:
    car_count = int(persona.get("NbCar", 0))
    bike_count = int(persona.get("NbBicy", 0))
    household = int(persona.get("NbHousehold", 0))
    children = int(persona.get("NbChild", 0))
    return f"""You are {persona['sex_text']}, {persona['age_text']}, with {persona['income_text']}. You are currently making {persona['trip_purpose_text']}. {persona['car_availability_text']}. Your household has {household} people, {children} children, {car_count} cars, and {bike_count} bicycles.

You know the following facts about yourself.
- Your education level is best summarized as: {persona['education_text']}.
- Your parents tended to rely on the car more than the train: {"yes" if int(persona['car_oriented_parents']) == 1 else "no"}.
- As a child, you lived in a suburb: {"yes" if int(persona['childSuburb']) == 1 else "no"}.
- As a child, you lived in a city center area: {"yes" if int(persona['city_center_as_kid']) == 1 else "no"}.
- You answer all questions as the same person in the same survey session.

You are answering a mobile travel survey.
- First you confirm your persona with a short JSON grounding reply.
- Then you answer six attitude statements one by one on a 1-6 Likert scale.
- Then you answer one travel-mode choice question with options PT, CAR, and SLOW_MODES.
- SLOW_MODES means walking, cycling, and other non-motorized modes.
- PT means public transport.
- Return JSON only for every answer.
"""


def build_grounding_prompt(persona: dict[str, Any]) -> str:
    return (
        "Before the survey starts, confirm that you understand who you are. "
        "Return JSON only in exactly this schema:\n"
        "{"
        f"\"persona_id\":\"{persona['respondent_id']}\","
        f"\"trip_purpose\":\"{persona['trip_purpose_text']}\","
        f"\"car_available\":{str(bool(int(persona['CAR_AVAILABLE']))).lower()},"
        "\"ready\":true"
        "}"
    )


def build_indicator_prompt(
    indicator_name: str,
    statement_text: str,
    question_index: int,
    total_questions: int,
    previous_answers: list[str] | None = None,
) -> str:
    history = ""
    if previous_answers:
        history = "Earlier answers in this survey: " + "; ".join(previous_answers) + "\n\n"
    return f"""Question {question_index} of {total_questions}. {history}Please rate how much this statement fits you.

Statement:
"{statement_text}"

{LIKERT_SCALE_TEXT}

Return JSON only like {{"indicator_value":4}}."""


def build_choice_prompt(
    persona: dict[str, Any],
    question_index: int,
    total_questions: int,
    previous_answers: list[str] | None = None,
) -> str:
    history = ""
    if previous_answers:
        history = "Earlier answers in this survey: " + "; ".join(previous_answers) + "\n\n"
    car_note = "available" if int(persona["CAR_AVAILABLE"]) == 1 else "unavailable"
    return f"""Question {question_index} of {total_questions}. {history}Choose exactly one travel option for this trip.

Option A: PT
- Travel time: {int(persona['TimePT'])} minutes
- Waiting time: {int(persona['WaitingTimePT'])} minutes
- Marginal cost: CHF {float(persona['MarginalCostPT']):.2f}

Option B: CAR
- Travel time: {int(persona['TimeCar'])} minutes
- Car cost: CHF {float(persona['CostCarCHF']):.2f}
- Availability: {car_note}

Option C: SLOW_MODES
- Distance: {float(persona['distance_km']):.1f} km

Return JSON only like {{"choice_label":"A"}}."""


def indicator_statement(indicator_name: str) -> str:
    return INDICATOR_TEXT[indicator_name]
