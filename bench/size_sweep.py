"""Speed vs accuracy of raw-coordinate pointing at different model-view sizes.

Ground truth comes from OCR on the live screen: pick text elements, ask the model to point at
each one by name (no OCR boxes shown), and count a hit if the point lands inside the box (+6px)."""

from __future__ import annotations

import json
import random
import sys
import time

from omni_agent import screen
from omni_agent.config import Settings
from omni_agent.grounding import POINT_PROMPT, parse_point
from omni_agent.models import make_model
from omni_agent.perception import ScreenParser


def main(sizes, n=10, seed=0):
    frame = screen.capture()
    els = [e for e in ScreenParser().parse(frame.image) if 3 <= len(e.text) <= 24 and e.text.isprintable()]
    texts = {}
    for e in els:  # only unambiguous labels
        texts.setdefault(e.text.lower(), []).append(e)
    pool = [v[0] for v in texts.values() if len(v) == 1]
    random.Random(seed).shuffle(pool)
    targets = pool[:n]
    model = make_model(Settings())
    model.generate("hi", [frame.image.resize((56, 56))], json_mode=False)  # load the model first
    rows = []
    for w, h in sizes:
        view = frame.image.resize((w, h))
        hits, secs = 0, []
        for i, el in enumerate(targets):
            # Nudge a pixel so every request is a fresh image (no prompt-cache hit).
            v = view.copy()
            v.putpixel((i, 0), (i, i, i))
            t = time.perf_counter()
            raw = model.generate(POINT_PROMPT.format(w=w, h=h, target=f'the text "{el.text}"'), [v])
            secs.append(time.perf_counter() - t)
            pt = parse_point(raw, w, h)
            if pt:
                x, y = pt[0] * frame.size[0] / w, pt[1] * frame.size[1] / h
                x1, y1, x2, y2 = el.box
                hits += x1 - 6 <= x <= x2 + 6 and y1 - 6 <= y <= y2 + 6
        row = {"view": f"{w}x{h}", "accuracy": hits / len(targets), "sec_per_call": sum(secs) / len(secs)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    sizes = [tuple(map(int, s.split("x"))) for s in sys.argv[1:]] or [(1288, 728), (980, 560), (784, 448)]
    main(sizes)
