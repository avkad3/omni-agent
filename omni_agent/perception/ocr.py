"""Text readers. Windows' built-in OCR engine is ~70x faster than RapidOCR on CPU (0.1 s vs 7 s for
a 1080p screen), so it is the default on Windows; RapidOCR is the cross-platform fallback."""

from __future__ import annotations

import asyncio
import sys

import numpy as np
from PIL import Image

from omni_agent.perception.elements import Element


def _text_el(box, text, score=1.0) -> Element:
    return Element(id=-1, kind="text", box=tuple(int(v) for v in box), text=text, score=score)


def split_line(words: list[tuple[str, tuple[float, float, float, float]]], gap_factor: float = 0.6) -> list[Element]:
    """Group a line's words into separate elements wherever the gap is wider than a normal space.
    Menu bars and toolbars come back as one OCR line ("File Edit Selection"); each word must be
    its own target."""
    groups: list[list] = []
    for text, (x1, y1, x2, y2) in words:
        h = max(1.0, y2 - y1)
        if groups and x1 - groups[-1][-1][1][2] <= gap_factor * h:
            groups[-1].append((text, (x1, y1, x2, y2)))
        else:
            groups.append([(text, (x1, y1, x2, y2))])
    out = []
    for g in groups:
        box = (min(b[0] for _, b in g), min(b[1] for _, b in g), max(b[2] for _, b in g), max(b[3] for _, b in g))
        out.append(_text_el(box, " ".join(t for t, _ in g)))
    return out


class WindowsOcr:
    def __init__(self):
        from winrt.windows.media.ocr import OcrEngine

        self.engine = OcrEngine.try_create_from_user_profile_languages()
        if self.engine is None:
            raise RuntimeError("no Windows OCR language pack installed")
        self.max_dim = OcrEngine.max_image_dimension

    async def _recognize(self, image: Image.Image):
        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.storage.streams import DataWriter

        w, h = image.size
        dw = DataWriter()
        dw.write_bytes(image.convert("RGBA").tobytes())
        bmp = SoftwareBitmap.create_copy_from_buffer(dw.detach_buffer(), BitmapPixelFormat.RGBA8, w, h)
        return await self.engine.recognize_async(bmp)

    def __call__(self, image: Image.Image) -> list[Element]:
        scale = 1.0
        if max(image.size) > self.max_dim:
            scale = self.max_dim / max(image.size)
            image = image.resize((int(image.width * scale), int(image.height * scale)))
        result = asyncio.run(self._recognize(image))
        out: list[Element] = []
        for line in result.lines:
            words = []
            for w in line.words:
                r = w.bounding_rect
                words.append((w.text, (r.x / scale, r.y / scale, (r.x + r.width) / scale, (r.y + r.height) / scale)))
            out.extend(split_line(words))
        return out


class RapidOcr:
    def __init__(self, min_score: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()
        self.min_score = min_score

    def __call__(self, image: Image.Image) -> list[Element]:
        result, _ = self.engine(np.asarray(image.convert("RGB")))
        out = []
        for pts, text, score in result or []:
            if float(score) < self.min_score or not str(text).strip():
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            out.append(_text_el((min(xs), min(ys), max(xs), max(ys)), str(text).strip(), float(score)))
        return out


def make_ocr(prefer: str = "auto"):
    if prefer in ("auto", "windows") and sys.platform == "win32":
        try:
            return WindowsOcr()
        except Exception:
            if prefer == "windows":
                raise
    return RapidOcr()
