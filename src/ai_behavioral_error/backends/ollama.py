from __future__ import annotations

import json
from typing import Sequence
from urllib.request import Request, urlopen

from ai_behavioral_error.backends.base import BaseBackend


class OllamaBackend(BaseBackend):
    def generate(self, messages: Sequence[dict], request_state: dict) -> dict:
        del request_state
        host = self.config.get("host", "http://127.0.0.1:11434").rstrip("/")
        timeout_sec = float(self.config.get("timeout_sec", 120))
        body = {
            "model": self.config["model_name"],
            "messages": list(messages),
            "stream": False,
            "think": self.config.get("think", True),
            "options": {
                "temperature": float(self.config.get("temperature", 0.1)),
                "top_p": float(self.config.get("top_p", 0.95)),
                "top_k": int(self.config.get("top_k", 20)),
                "seed": int(self.config.get("seed", 0)),
                "num_predict": int(self.config.get("num_predict", 128)),
            },
        }

        request = Request(
            url=f"{host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))

        message = payload["message"]
        thinking_text = message.get("thinking", "")
        response_text = message.get("content", "")
        return {
            "response_text": response_text,
            "thinking_text": thinking_text,
            "raw_text": json.dumps(message),
            "metadata": {
                "done_reason": payload.get("done_reason", ""),
                "total_duration": payload.get("total_duration", 0),
                "load_duration": payload.get("load_duration", 0),
                "prompt_eval_count": payload.get("prompt_eval_count", 0),
                "prompt_eval_duration": payload.get("prompt_eval_duration", 0),
                "eval_count": payload.get("eval_count", 0),
                "eval_duration": payload.get("eval_duration", 0),
            },
        }
