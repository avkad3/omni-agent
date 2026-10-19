"""Live task suite: runs each task in tasks/suite.json on the real desktop and checks the outcome itself
(screen text, window title or file contents) instead of trusting the agent's own "done".

Safety:
* each task runs with its own allow-list, so the agent only acts inside the named app's windows;
* files are written only under C:\\Users\\<you>\\omni-agent-sandbox;
* cleanup posts WM_CLOSE only to top-level windows that did not exist before the task started, so the
  user's own windows (editors, browser sessions) are never touched;
* web tasks use an InPrivate Edge window opened by the harness.

    python bench/run_suite.py                 # all tasks
    python bench/run_suite.py calc-add web-example
    python bench/run_suite.py --report
"""

from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

from omni_agent import screen
from omni_agent.agent import Agent
from omni_agent.config import Settings
from omni_agent.perception import ScreenParser
from omni_agent.safety import foreground_title

TASKS = Path("tasks/suite.json")
OUT = Path("bench/results/suite.jsonl")
SANDBOX = Path.home() / "omni-agent-sandbox"

user32 = ctypes.windll.user32
WM_CLOSE = 0x0010


def top_windows() -> dict[int, str]:
    found: dict[int, str] = {}

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                found[hwnd] = buf.value
        return True

    user32.EnumWindows(cb, 0)
    return found


def close_new_windows(before: dict[int, str]) -> list[str]:
    closed = []
    for hwnd, title in top_windows().items():
        if hwnd not in before and title not in ("Program Manager",):
            user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            closed.append(title)
    return closed


def setup(kind: str | None) -> None:
    if kind == "edge-inprivate":
        subprocess.Popen(["cmd", "/c", "start", "", "msedge", "--inprivate", "--new-window", "about:blank"])
        time.sleep(4)


def check(spec: dict, parser: ScreenParser) -> tuple[bool, str]:
    if "file" in spec:
        path, text = spec["file"]
        p = Path(path)
        ok = p.exists() and text.lower() in p.read_text(encoding="utf-8", errors="replace").lower()
        return ok, f"file {path} {'contains' if ok else 'missing or lacks'} {text!r}"
    if "title" in spec:
        title = foreground_title()
        return any(t.lower() in title.lower() for t in spec["title"]), f"title {title!r}"
    texts = [e.text for e in parser.parse(screen.capture().image, icons=False)]
    if "screen_text_exact" in spec:
        ok = any(t.strip() == want for t in texts for want in spec["screen_text_exact"])
        return ok, f"exact {spec['screen_text_exact']} on screen: {ok}"
    blob = " | ".join(texts).lower()
    found = [w for w in spec["screen_text"] if w.lower() in blob]
    return bool(found), f"found {found}" if found else f"none of {spec['screen_text']} on screen"


def run(ids: list[str]) -> None:
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))
    if ids:
        tasks = [t for t in tasks if t["id"] in ids]
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    SANDBOX.mkdir()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parser = ScreenParser()
    for t in tasks:
        before = top_windows()
        setup(t.get("setup"))
        s = Settings(max_steps=12, allowed_apps=t["allow"], confirm_risky=True, runs_dir=Path("runs/suite"))
        agent = Agent(s, ask=lambda q: False)  # unattended: risky actions are refused, never assumed
        start = time.time()
        rec = agent.run(t["goal"])
        time.sleep(1.0)
        ok, why = check(t["check"], parser)
        result = {"id": t["id"], "goal": t["goal"], "success": ok, "check": why, "agent_outcome": rec.outcome,
                  "agent_summary": rec.summary, "steps": len(rec.steps), "seconds": round(time.time() - start, 1),
                  "planner_calls": sum(1 for st in rec.steps if st.source == "planner"),
                  "report": str(rec.dir / "report.html")}
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(f"== {t['id']}: {'PASS' if ok else 'FAIL'} ({why}) agent said {rec.outcome}, {result['seconds']}s",
              flush=True)
        closed = close_new_windows(before)
        if closed:
            print(f"   closed new windows: {closed}", flush=True)
        time.sleep(2)


def report() -> None:
    rows = {}
    for line in OUT.open(encoding="utf-8"):
        r = json.loads(line)
        rows[r["id"]] = r  # keep the latest attempt per task
    rows = list(rows.values())
    passed = sum(r["success"] for r in rows)
    false_done = sum(r["agent_outcome"] == "done" and not r["success"] for r in rows)
    print(f"{passed}/{len(rows)} tasks passed; agent claimed done but check failed: {false_done}")
    print("| Task | Result | Steps | Time | Check |\n|---|---|---|---|---|")
    for r in rows:
        verdict = "pass" if r["success"] else "fail"
        print(f"| {r['id']} | {verdict} | {r['steps']} | {r['seconds']:.0f}s | {r['check']} |")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        run([a for a in sys.argv[1:] if not a.startswith("-")])
