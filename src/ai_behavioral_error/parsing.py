from __future__ import annotations

import json
import re


def parse_choice_label(raw_text: str) -> str:
    text = raw_text.strip()

    if text.startswith("{"):
        payload = json.loads(text)
        return str(payload["choice"]).strip().upper()

    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else ""
