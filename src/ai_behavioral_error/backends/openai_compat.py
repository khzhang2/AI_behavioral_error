from __future__ import annotations

import json
import os
from typing import Sequence
from urllib.request import Request, urlopen

from ai_behavioral_error.backends.base import BaseBackend


class OpenAICompatBackend(BaseBackend):
    def generate(self, messages: Sequence[dict], request_state: dict) -> dict:
        del request_state
        base_url = self.config["base_url"].rstrip("/")
        body = {
            "model": self.config["model_name"],
            "messages": list(messages),
            "temperature": float(self.config.get("temperature", 0.1)),
            "top_p": float(self.config.get("top_p", 1.0)),
            "seed": int(self.config.get("seed", 0)),
            "max_tokens": int(self.config.get("max_tokens", 64)),
        }

        request = Request(
            url=f"{base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {os.environ[self.config['api_key_env']]}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))

        response_text = payload["choices"][0]["message"]["content"]
        return {
            "response_text": response_text,
            "thinking_text": "",
            "raw_text": response_text,
            "metadata": payload.get("usage", {}),
        }
