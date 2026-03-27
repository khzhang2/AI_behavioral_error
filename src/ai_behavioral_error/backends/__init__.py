from ai_behavioral_error.backends.base import BaseBackend
from ai_behavioral_error.backends.mock import MockBackend
from ai_behavioral_error.backends.ollama import OllamaBackend
from ai_behavioral_error.backends.openai_compat import OpenAICompatBackend


def build_backend(config: dict) -> BaseBackend:
    backend_type = config["type"]
    if backend_type == "mock":
        return MockBackend(config)
    if backend_type == "ollama":
        return OllamaBackend(config)
    if backend_type == "openai_compat":
        return OpenAICompatBackend(config)
    raise ValueError(f"Unknown backend type: {backend_type}")
