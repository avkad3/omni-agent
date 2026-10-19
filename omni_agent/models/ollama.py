"""Local model through Ollama (default: Qwen2.5-VL 3B, ~3.2 GB, fits a 4 GB GPU)."""

from __future__ import annotations

import httpx
from PIL import Image

from omni_agent.models.base import image_to_b64


class OllamaModel:
    def __init__(self, model: str, url: str = "http://localhost:11434", timeout: float = 180.0):
        self.name = f"ollama:{model}"
        self.model = model
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def generate(self, prompt: str, images: list[Image.Image], system: str = "", json_mode: bool = True) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": [image_to_b64(i) for i in images]})
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # 6k context covers screenshot (~300 tokens) + element list + history. Leave GPU layer
            # placement to Ollama: forcing every layer onto a 4 GB card spills into shared memory
            # and is ~2.5x slower than its automatic CPU/GPU split.
            "options": {"temperature": 0, "num_ctx": 6144},
            "keep_alive": "15m",
        }
        if json_mode:
            body["format"] = "json"
        r = self.client.post(f"{self.url}/api/chat", json=body)
        r.raise_for_status()
        return r.json()["message"]["content"]
