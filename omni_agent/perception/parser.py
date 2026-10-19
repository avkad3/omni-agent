"""Turns a raw screenshot into a list of UI elements, using only pixels.

Sources:
* OCR (Windows built-in engine, or RapidOCR): every visible text group becomes a clickable element.
* Icon detector (optional, see detector.py): finds unlabeled buttons and icons.
"""

from __future__ import annotations

import time
from typing import Protocol

from PIL import Image

from omni_agent.perception.elements import Element, iou


class IconDetector(Protocol):
    def detect(self, image: Image.Image) -> list[tuple[tuple[int, int, int, int], float]]: ...


class ScreenParser:
    def __init__(self, detector: IconDetector | None = None, ocr: str = "auto"):
        self._ocr = None
        self.ocr_pref = ocr
        self.detector = detector
        self.last_timing: dict[str, float] = {}

    @property
    def ocr(self):
        if self._ocr is None:
            from omni_agent.perception.ocr import make_ocr

            self._ocr = make_ocr(self.ocr_pref)
        return self._ocr

    def parse(self, image: Image.Image) -> list[Element]:
        t0 = time.perf_counter()
        elements = self.ocr(image)
        t1 = time.perf_counter()
        if self.detector is not None:
            for box, score in self.detector.detect(image):
                # An icon box that mostly overlaps a text box is the same control; keep the text one.
                if any(iou(box, e.box) > 0.5 for e in elements):
                    continue
                elements.append(Element(id=-1, kind="icon", box=box, score=score))
        t2 = time.perf_counter()
        # Reading order (top-to-bottom, left-to-right in 20px bands) gives stable, guessable ids.
        elements.sort(key=lambda e: (e.box[1] // 20, e.box[0]))
        for i, el in enumerate(elements):
            el.id = i
        self.last_timing = {"ocr_s": t1 - t0, "detector_s": t2 - t1}
        return elements
