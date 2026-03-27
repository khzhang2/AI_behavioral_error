from __future__ import annotations

from typing import Sequence

from ai_behavioral_error.design import Alternative, ChoiceTask


def _format_value(value: float, unit: str) -> str:
    if value == 0:
        return "-"
    if unit == "EUR":
        return f"{value:g} EUR"
    if unit == "%":
        return f"{value:g}%"
    return f"{value:g} {unit}"


def _format_optional_text(value: str) -> str:
    return value if value else "-"


def build_system_prompt() -> str:
    return (
        "You are answering a transport stated-preference survey. "
        "For each card, choose exactly one alternative using only the attributes shown. "
        "Do not add assumptions, policy opinions, or moral commentary. "
        "If your model emits a separate thinking trace, keep it brief and move quickly to the final answer. "
        "Return strict JSON on one line in the form {\"choice\": \"A\"}."
    )


def build_user_prompt(scenario_text: str, task: ChoiceTask, ordered_alternatives: Sequence[tuple[str, Alternative]]) -> str:
    lines = [
        scenario_text,
        "",
        f"Choice task {task.task_index} of 6",
        f"Trip length: {task.trip_length_km:g} km",
        "",
        "Alternatives:",
    ]

    for display_label, alternative in ordered_alternatives:
        lines.extend(
            [
                f"{display_label}. {alternative.alternative_name}",
                f"  travel_time: {_format_value(alternative.travel_time_min, 'min')}",
                f"  access_time: {_format_value(alternative.access_time_min, 'min')}",
                f"  egress_time: {_format_value(alternative.egress_time_min, 'min')}",
                f"  parking_search_time: {_format_value(alternative.parking_search_time_min, 'min')}",
                f"  availability: {_format_value(alternative.availability_pct, '%')}",
                f"  cost: {_format_value(alternative.cost_eur, 'EUR')}",
                f"  scheme: {_format_optional_text(alternative.scheme)}",
                f"  engine: {_format_optional_text(alternative.engine)}",
                f"  range: {_format_value(alternative.range_km, 'km')}",
                "",
            ]
        )

    lines.extend(
        [
            "If your model emits thinking, keep it short.",
            "Respond with one line of JSON only.",
            "Example: {\"choice\": \"B\"}",
        ]
    )
    return "\n".join(lines)
