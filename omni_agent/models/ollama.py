"""Local model through Ollama (default: Qwen2.5-VL 3B, ~3.2 GB, fits a 4 GB GPU)."""

from __future__ import annotations

import httpx
from PIL import Image

from omni_agent.models.base import image_to_b64


class ModelError(RuntimeError):
    pass


class OllamaModel:
    """After many image calls the llama.cpp runner can get into a bad state: JSON-constrained replies come
    back empty ("Unexpected empty grammar stack") and free-form ones loop until the repeat limit. An empty
    reply is never treated as an answer: retry once without the JSON grammar, then unload and reload."""

    def __init__(self, model: str, url: str = "http://localhost:11434", timeout: float = 300.0,
                 transport: httpx.BaseTransport | None = None):
        self.name = f"ollama:{model}"
        self.model = model
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=timeout, transport=transport)
        self.recoveries = 0  # how often a bad reply forced a retry or reload

    def generate(self, prompt: str, images: list[Image.Image], system: str = "", json_mode: bool = True) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": [image_to_b64(i) for i in images]})
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # 6k context covers screenshot (~1.2k tokens after Ollama's resize) + element list + history.
            # Leave GPU layer placement to Ollama: forcing every layer onto a 4 GB card spills into
            # shared memory and is ~2.5x slower than its automatic CPU/GPU split.
            "options": {"temperature": 0, "num_ctx": 6144, "repeat_penalty": 1.05},
            "keep_alive": "15m",
        }
        if json_mode:
            body["format"] = "json"
        error = ""
        for attempt in range(3):
            r = self.client.post(f"{self.url}/api/chat", json=body)
            try:
                data = r.json()
            except ValueError:
                data = {"error": r.text[:200]}
            content = (data.get("message") or {}).get("content", "")
            if r.status_code == 200 and content.strip():
                return content
            error = data.get("error") or f"empty reply (HTTP {r.status_code})"
            self.recoveries += 1
            if attempt == 0:
                body.pop("format", None)  # the JSON grammar is the usual casualty; parse free text instead
            else:
                self.reload()
        raise ModelError(f"{self.model}: {error}")

    def reload(self) -> None:
        """Unload the model so the next request starts a fresh runner."""
        try:
            self.client.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0})
        except httpx.HTTPError:
            pass
