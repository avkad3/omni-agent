from __future__ import annotations

from omni_agent.config import Settings


def make_model(settings: Settings):
    if settings.backend == "gemini":
        from omni_agent.models.gemini import GeminiModel

        return GeminiModel(settings.gemini_model, settings.gemini_api_key)
    if settings.backend == "ollama":
        from omni_agent.models.ollama import OllamaModel

        return OllamaModel(settings.ollama_model, settings.ollama_url)
    raise ValueError(f"unknown backend {settings.backend!r}")
