"""Grounding accuracy on ScreenSpot (desktop split: Windows + macOS screenshots).

Each sample is an instruction ("close", "open the File menu") and the true bounding box. A method scores
a hit when its click point lands inside that box. All methods are evaluated on the same samples and
share model calls, so one sample costs at most two calls (a point and a zoom):

  text        OCR text match only, no model (misses whatever has no matching text)
  point       one raw coordinate guess from the model
  point+snap  the same guess, snapped to the nearest OCR/icon box
  point+zoom  the guess refined on a 2x crop (second call)
  pipeline    what the agent does: text match if confident, else point+snap, else point+zoom

    python bench/screenspot.py --per-type 40           # results -> bench/results/screenspot.jsonl
    python bench/screenspot.py --report                 # summary table from saved results
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from omni_agent.config import Settings
from omni_agent.grounding import POINT_PROMPT, Grounder, best_text_match, parse_point, snap
from omni_agent.models import make_model
from omni_agent.perception import ScreenParser
from omni_agent.perception.detector import load_detector

DATA = Path("data/screenspot")
OUT = Path("bench/results/screenspot.jsonl")
METHODS = ["text", "point", "point+snap", "point+zoom", "pipeline"]


def load_samples(per_type: int, seed: int):
    """Pick samples from the light columns first, then decode only the chosen screenshots one at a time
    (the full table is ~600 MB; on an 8 GB machine loading it all starves the model of RAM)."""
    meta = pq.read_table(DATA, columns=["file_name", "data_source", "data_type"]).to_pylist()
    rng = random.Random(seed)
    chosen = set()
    for kind in ("text", "icon"):
        group = [i for i, r in enumerate(meta) if r["data_type"] == kind and r["data_source"] in ("windows", "macos")]
        rng.shuffle(group)
        chosen.update(group[:per_type])
    offset = 0
    for f in sorted(DATA.glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=16):
            for j, row in enumerate(batch.to_pylist()):
                if offset + j in chosen:
                    yield row
            offset += batch.num_rows


def inside(pt, box) -> bool:
    return pt is not None and box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def evaluate(sample: dict, parser: ScreenParser, grounder: Grounder, model) -> dict:
    img = Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB")
    W, H = img.size
    b = sample["bbox"]
    box = (b[0] * W, b[1] * H, b[2] * W, b[3] * H)
    instr = sample["instruction"]
    t = time.perf_counter()
    elements = parser.parse(img, icons=True)
    perceive_s = time.perf_counter() - t
    pts: dict[str, tuple[float, float] | None] = {}
    calls = 0

    match, score = best_text_match(instr, elements)
    pts["text"] = match.center if match is not None and score >= grounder.text_threshold else None

    t = time.perf_counter()
    view = img.resize((grounder.mw, grounder.mh), Image.LANCZOS)
    raw = model.generate(POINT_PROMPT.format(w=grounder.mw, h=grounder.mh, target=instr), [view])
    calls += 1
    p = parse_point(raw, grounder.mw, grounder.mh)
    pts["point"] = (p[0] * W / grounder.mw, p[1] * H / grounder.mh) if p else None
    snapped = snap(*pts["point"], elements, radius=grounder.snap_radius * W / 1920) if pts["point"] else None
    pts["point+snap"] = snapped.center if snapped is not None else pts["point"]
    zoomed = grounder._zoom(img, instr, *pts["point"]) if pts["point"] else None
    calls += pts["point"] is not None
    pts["point+zoom"] = zoomed or pts["point"]
    model_s = time.perf_counter() - t

    if pts["text"] is not None:
        pts["pipeline"], route = pts["text"], "text"
    elif snapped is not None:
        pts["pipeline"], route = snapped.center, "point+snap"
    else:
        pts["pipeline"], route = pts["point+zoom"], "point+zoom"

    return {"file": sample["file_name"], "source": sample["data_source"], "type": sample["data_type"],
            "instruction": instr, "size": [W, H], "route": route, "text_score": round(score, 3),
            "hits": {m: inside(pts[m], box) for m in METHODS}, "calls": calls,
            "perceive_s": round(perceive_s, 2), "model_s": round(model_s, 1), "n_elements": len(elements)}


def report(path: Path = OUT) -> str:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    lines = [f"ScreenSpot desktop, n={len(rows)} ({sum(r['type'] == 'text' for r in rows)} text, "
             f"{sum(r['type'] == 'icon' for r in rows)} icon)", "",
             "| Method | Text targets | Icon targets | All | Model calls |", "|---|---|---|---|---|"]
    for m in METHODS:
        acc = defaultdict(list)
        for r in rows:
            acc[r["type"]].append(r["hits"][m])
            acc["all"].append(r["hits"][m])
        pct = {k: f"{100 * sum(v) / len(v):.0f}%" if v else "-" for k, v in acc.items()}
        cost = {"text": "0", "point": "1", "point+snap": "1", "point+zoom": "2",
                "pipeline": f"{sum(r['route'] == 'point+zoom' for r in rows) / len(rows) + 1:.1f} avg"}[m]
        lines.append(f"| {m} | {pct.get('text', '-')} | {pct.get('icon', '-')} | {pct['all']} | {cost} |")
    routes = defaultdict(list)
    for r in rows:
        routes[r["route"]].append(r["hits"]["pipeline"])
    lines += ["", "Pipeline routes: " + ", ".join(f"{k} {len(v)} ({100 * sum(v) / len(v):.0f}% hit)"
                                               for k, v in routes.items())]
    lines.append(f"Avg model time per sample {sum(r['model_s'] for r in rows) / len(rows):.1f}s, "
                 f"perception {sum(r['perceive_s'] for r in rows) / len(rows):.1f}s")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["file"] + json.loads(line)["instruction"] for line in OUT.open(encoding="utf-8")} \
        if OUT.exists() else set()
    s = Settings()
    model = make_model(s)
    parser = ScreenParser(detector=load_detector())
    grounder = Grounder(model, (s.model_width, s.model_height))
    total = 2 * a.per_type
    print(f"{total} samples selected ({len(done)} already done)", flush=True)
    i = 0
    for sample in load_samples(a.per_type, a.seed):
        if sample["file_name"] + sample["instruction"] in done:
            continue
        i += 1
        r = evaluate(sample, parser, grounder, model)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        print(f"{i}/{total - len(done)} {r['type']:4} {r['route']:10} pipeline={r['hits']['pipeline']!s:5} "
              f"{r['model_s']:5.1f}s {r['instruction'][:50]!r}", flush=True)
    print(report())


if __name__ == "__main__":
    main()
