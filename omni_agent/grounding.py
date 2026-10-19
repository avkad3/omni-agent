"""Turn "what to click" into exact pixels.

Cheapest and most reliable first:
1. text   - the target names visible text: click that OCR box. No model call.
2. mark   - the planner picked a numbered element: click its centre.
3. point  - ask the model for raw coordinates, then crop around its guess, enlarge, and ask again
            (zoom refinement). Needed for icons, sprites and canvas content the parser missed.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from PIL import Image

from omni_agent.actions import extract_json
from omni_agent.perception.elements import Element

_FILLER = re.compile(
    r"\b(the|a|an|button|icon|tab|menu|item|option|link|field|box|label|text|named|called|labeled|labelled|"
    r"that says|which says|on|in|at)\b",
    re.I,
)


def normalize(s: str) -> str:
    s = s.lower().strip().strip("\"'`")
    s = re.sub(r"[\"'`“”‘’]", "", s)
    s = _FILLER.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def text_score(target: str, text: str) -> float:
    t, x = normalize(target), normalize(text)
    if not t or not x:
        return 0.0
    if t == x:
        return 1.0
    ratio = difflib.SequenceMatcher(None, t, x).ratio()
    # Target fully contained in a short label ("save" in "save as") is a strong but not perfect match.
    if len(t) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", x):
        ratio = max(ratio, 0.8 if len(x) <= len(t) + 8 else 0.7)
    return ratio


def quoted(target: str) -> str | None:
    m = re.search(r"[\"'“‘]([^\"'”’]{1,60})[\"'”’]", target)
    return m.group(1) if m else None


def best_text_match(target: str, elements: list[Element]) -> tuple[Element | None, float]:
    if not target:
        return None, 0.0
    probe = quoted(target) or target
    best, best_s = None, 0.0
    for el in elements:
        if el.kind != "text":
            continue
        s = text_score(probe, el.text)
        # Prefer the smaller box on ties: a button label beats a paragraph that mentions the word.
        if s > best_s + 1e-6 or (abs(s - best_s) <= 1e-6 and best is not None and el.area < best.area):
            best, best_s = el, s
    return best, best_s


@dataclass
class Grounding:
    x: float  # frame pixels
    y: float
    method: str  # "text" | "mark" | "point" | "point+zoom"
    element: Element | None = None
    detail: str = ""


POINT_PROMPT = (
    "This is a {w}x{h} screenshot. Locate: {target}\n"
    'Reply with JSON only: {{"x": <int>, "y": <int>}} - the pixel at the centre of that element '
    "in this image. If it is not visible reply {{\"x\": -1, \"y\": -1}}."
)


def parse_point(raw: str, w: int, h: int) -> tuple[float, float] | None:
    """Accept {"x","y"}, {"point":[x,y]}, {"bbox_2d":[x1,y1,x2,y2]} or a bare "(x, y)"."""
    try:
        obj = extract_json(raw)
    except Exception:
        obj = {}
    pt = None
    if "x" in obj and "y" in obj:
        pt = (float(obj["x"]), float(obj["y"]))
    elif isinstance(obj.get("point") or obj.get("point_2d"), list):
        p = obj.get("point") or obj.get("point_2d")
        pt = (float(p[0]), float(p[1]))
    elif isinstance(obj.get("bbox_2d") or obj.get("bbox"), list):
        b = obj.get("bbox_2d") or obj.get("bbox")
        pt = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    else:
        m = re.search(r"\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?", raw)
        if m:
            pt = (float(m.group(1)), float(m.group(2)))
    if pt is None or pt[0] < 0 or pt[1] < 0:
        return None
    # Some models answer in 0-1 or 0-1000 normalised space; detect and convert.
    if pt[0] <= 1 and pt[1] <= 1:
        pt = (pt[0] * w, pt[1] * h)
    elif (pt[0] > w or pt[1] > h) and pt[0] <= 1000 and pt[1] <= 1000:
        pt = (pt[0] * w / 1000, pt[1] * h / 1000)
    if pt[0] > w or pt[1] > h:
        return None
    return pt


class Grounder:
    def __init__(self, model, model_size: tuple[int, int], text_threshold: float = 0.82, zoom: bool = True):
        self.model = model
        self.mw, self.mh = model_size
        self.text_threshold = text_threshold
        self.zoom = zoom

    def ground(self, image: Image.Image, elements: list[Element], target: str,
               element_id: int | None = None) -> Grounding | None:
        by_id = {e.id: e for e in elements}
        match, score = best_text_match(target, elements)
        picked = by_id.get(element_id) if element_id is not None else None

        if picked is not None:
            # Small models often cite a neighbouring id. If the target names text that a
            # different element shows much more clearly, trust the text over the id.
            if match is not None and match is not picked and score >= self.text_threshold:
                if picked.kind == "icon" or text_score(target, picked.text) < score - 0.25:
                    return Grounding(*match.center, "text", match, f"id {element_id} overruled, score {score:.2f}")
            return Grounding(*picked.center, "mark", picked)
        if match is not None and score >= self.text_threshold:
            return Grounding(*match.center, "text", match, f"score {score:.2f}")
        if self.model is None or not target:
            return None
        return self.point(image, target)

    def point(self, image: Image.Image, target: str) -> Grounding | None:
        fw, fh = image.size
        view = image.resize((self.mw, self.mh), Image.LANCZOS)
        raw = self.model.generate(POINT_PROMPT.format(w=self.mw, h=self.mh, target=target), [view])
        pt = parse_point(raw, self.mw, self.mh)
        if pt is None:
            return None
        x, y = pt[0] * fw / self.mw, pt[1] * fh / self.mh
        if not self.zoom:
            return Grounding(x, y, "point", detail=raw[:120])
        refined = self._zoom(image, target, x, y)
        if refined is None:
            return Grounding(x, y, "point", detail="zoom failed")
        return Grounding(*refined, "point+zoom")

    def _zoom(self, image: Image.Image, target: str, x: float, y: float, crop: int = 448,
              side: int = 896) -> tuple[float, float] | None:
        """Crop a crop x crop window around the guess, show it enlarged, and ask again."""
        fw, fh = image.size
        x1 = int(min(max(0, x - crop / 2), max(0, fw - crop)))
        y1 = int(min(max(0, y - crop / 2), max(0, fh - crop)))
        cw, ch = min(crop, fw), min(crop, fh)
        view = image.crop((x1, y1, x1 + cw, y1 + ch)).resize((side, side), Image.LANCZOS)
        raw = self.model.generate(POINT_PROMPT.format(w=side, h=side, target=target), [view])
        pt = parse_point(raw, side, side)
        if pt is None:
            return None
        return x1 + pt[0] * cw / side, y1 + pt[1] * ch / side
