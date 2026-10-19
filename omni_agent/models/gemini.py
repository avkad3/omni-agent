"""Optional cloud planner on Google's free Gemini API tier (set GEMINI_API_KEY)."""

from __future__ import annotations

import time

import httpx
from PIL import Image

from omni_agent.models.base import image_to_b64

_API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiModel:
    def __init__(self, model: str, api_key: str, timeout: float = 90.0):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.name = f"gemini:{model}"
        self.model = model
        self.api_key = api_key
        self.client = httpx.Client(timeout=timeout)

    def generate(self, prompt: str, images: list[Image.Image], system: str = "", json_mode: bool = True) -> str:
        parts = [{"inline_data": {"mime_type": "image/jpeg", "data": image_to_b64(i.convert("RGB"), "JPEG")}}
                 for i in images]
        parts.append({"text": prompt})
        body: dict = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": 0}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        for attempt in range(4):
            r = self.client.post(_API.format(model=self.model), params={"key": self.api_key}, json=body)
            if r.status_code == 429 and attempt < 3:  # free-tier rate limit: back off and retry
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        raise RuntimeError("Gemini rate limit")
