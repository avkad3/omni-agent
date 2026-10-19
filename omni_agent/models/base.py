from __future__ import annotations

import base64
import io
from typing import Protocol

from PIL import Image


def image_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


class VisionModel(Protocol):
    name: str

    def generate(self, prompt: str, images: list[Image.Image], system: str = "", json_mode: bool = True) -> str:
        """Return the model's raw text reply."""
        ...
