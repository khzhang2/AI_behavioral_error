from __future__ import annotations

import json
from math import exp
from random import Random
from typing import Sequence

from ai_behavioral_error.backends.base import BaseBackend


class MockBackend(BaseBackend):
    def __init__(self, config: dict):
        super().__init__(config)
        self.coefficients = dict(config["utility_coefficients"])
        self.temperature = float(config.get("temperature", 0.2))
        self.seed = int(config.get("seed", 0))

    def _utility(self, alternative_payload: dict) -> float:
        features = alternative_payload["features"]
        utility = 0.0
        for name, value in features.items():
            utility += self.coefficients.get(name, 0.0) * float(value)
        return utility

    def generate(self, messages: Sequence[dict], request_state: dict) -> dict:
        del messages
        rng = Random(f"{self.seed}-{request_state['respondent_id']}-{request_state['task_id']}-{request_state['repeat_id']}")
        alternatives = request_state["alternatives"]
        utilities = [self._utility(alternative) for alternative in alternatives]
        scale = max(self.temperature, 1e-6)
        weights = [exp(utility / scale) for utility in utilities]
        total = sum(weights)
        draw = rng.random() * total

        cumulative = 0.0
        chosen = alternatives[-1]
        for alternative, weight in zip(alternatives, weights):
            cumulative += weight
            if draw <= cumulative:
                chosen = alternative
                break

        response_text = json.dumps(
            {
                "choice": chosen["display_label"],
                "brief_reason": "Mock respondent draw from a simple utility model."
            }
        )
        return {
            "response_text": response_text,
            "thinking_text": "",
            "raw_text": response_text,
            "metadata": {
                "prompt_eval_count": 0,
                "eval_count": 0,
                "total_duration": 0,
            },
        }
