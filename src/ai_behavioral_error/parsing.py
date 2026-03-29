from __future__ import annotations

import json
import re


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_choice_label(raw_text: str) -> str:
    text = _strip_code_fence(raw_text)

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("choice", "choice_label", "chosen_label"):
                if key in payload:
                    return str(payload[key]).strip().upper()

    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else ""
