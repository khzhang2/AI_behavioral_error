from __future__ import annotations

from typing import Sequence


class BaseBackend:
    def __init__(self, config: dict):
        self.config = config

    @property
    def model_name(self) -> str:
        return str(self.config["model_name"])

    def generate(self, messages: Sequence[dict], request_state: dict) -> dict:
        raise NotImplementedError
