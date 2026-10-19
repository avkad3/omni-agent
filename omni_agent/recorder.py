"""Saves every step of a run (screenshots, boxes, click point, reasoning) and writes an HTML viewer."""

from __future__ import annotations

import html
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw


@dataclass
class StepLog:
    step: int
    action: dict
    result: str
    method: str = ""
    click: tuple[float, float] | None = None  # frame pixels
    change: float | None = None
    timings: dict = field(default_factory=dict)
    image: str = ""
    raw: str = ""
    subtask: str = ""
    source: str = ""  # "skill" (no model call) or "planner"


class Recorder:
    def __init__(self, root: Path, goal: str, meta: dict):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.goal = goal
        self.meta = meta
        self.steps: list[StepLog] = []
        self.started = time.time()
        self.outcome = "running"
        self.summary = ""

    def add(self, log: StepLog, annotated: Image.Image, frame_size: tuple[int, int]) -> None:
        img = annotated.convert("RGB").copy()
        if log.click is not None:
            sx = img.size[0] / frame_size[0]
            sy = img.size[1] / frame_size[1]
            x, y = log.click[0] * sx, log.click[1] * sy
            d = ImageDraw.Draw(img)
            for r, w in ((16, 4), (6, 0)):
                d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=w or 1,
                          fill=(255, 0, 0) if w == 0 else None)
        name = f"step_{log.step:02d}.jpg"
        img.save(self.dir / name, quality=80)
        log.image = name
        self.steps.append(log)
        with open(self.dir / "steps.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(log)) + "\n")

    def finish(self, outcome: str, summary: str) -> Path:
        self.outcome, self.summary = outcome, summary
        result = {"goal": self.goal, "outcome": outcome, "summary": summary, "steps": len(self.steps),
                  "seconds": round(time.time() - self.started, 1), **self.meta}
        (self.dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        page = self.dir / "report.html"
        page.write_text(self._html(result), encoding="utf-8")
        return page

    def _html(self, result: dict) -> str:
        e = html.escape
        cards = []
        for s in self.steps:
            a = s.action
            detail = {k: v for k, v in a.items() if v not in ("", None, [], 5, 1.0, "down") and k != "thought"}
            t = ", ".join(f"{k} {v:.1f}s" for k, v in s.timings.items())
            cards.append(f"""
<section class="step">
  <img src="{e(s.image)}" loading="lazy" alt="step {s.step}">
  <div class="meta">
    <h2>Step {s.step}: {e(a.get('action', ''))}{f' <span class="muted">via {e(s.source)}</span>' if s.source else ''}</h2>
    {f'<p class="muted">Plan step: {e(s.subtask)}</p>' if s.subtask else ''}
    <p class="thought">{e(a.get('thought', ''))}</p>
    <p><b>Action</b> <code>{e(json.dumps(detail))}</code></p>
    <p><b>Result</b> {e(s.result)}{f' &middot; grounded by <b>{e(s.method)}</b>' if s.method else ''}</p>
    <p class="muted">{e(t)}{f' &middot; screen changed {s.change:.0%}' if s.change is not None else ''}</p>
  </div>
</section>""")
        badge = {"done": "ok", "fail": "bad"}.get(result["outcome"], "warn")
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Omni-Agent run</title>
<style>
:root{{--bg:#fff;--fg:#1b1b1f;--muted:#6b6b76;--card:#f5f5f7;--ok:#1a7f37;--bad:#c62828;--warn:#9a6700}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111114;--fg:#ececf1;--muted:#9a9aa5;--card:#1c1c21}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}} .muted{{color:var(--muted);font-size:13px}}
.badge{{display:inline-block;padding:2px 10px;border-radius:99px;color:#fff;font-weight:600;font-size:13px}}
.ok{{background:var(--ok)}} .bad{{background:var(--bad)}} .warn{{background:var(--warn)}}
.step{{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px;background:var(--card);
border-radius:12px;padding:12px;margin:14px 0}} .step img{{width:100%;border-radius:8px}}
.step h2{{font-size:17px;margin:0 0 6px}} .thought{{font-style:italic}} code{{font-size:12px;word-break:break-word}}
@media (max-width:720px){{.step{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>{e(result['goal'])}</h1>
<p><span class="badge {badge}">{e(result['outcome'])}</span> {e(result['summary'])}</p>
{'<p><b>Plan</b> ' + ' &rarr; '.join(e(str(p)) for p in result.get('plan', [])) + '</p>' if result.get('plan') else ''}
<p class="muted">{result['steps']} steps &middot; {result['seconds']}s &middot; model {e(str(result.get('model', '')))}</p>
{''.join(cards)}
</body></html>"""
