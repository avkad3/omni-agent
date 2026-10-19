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
    python bench/run_suite.py --replay        # also rerun each passed task from its stored trajectory
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
REPLAY_DIR = Path("trajectories/suite")

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
        if hwnd not in before and title != "Program Manager":
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


def attempt(t: dict, parser: ScreenParser) -> dict:
    before = top_windows()
    setup(t.get("setup"))
    s = Settings(max_steps=12, allowed_apps=t["allow"], confirm_risky=True, runs_dir=Path("runs/suite"),
                 trajectories_dir=REPLAY_DIR)
    agent = Agent(s, ask=lambda q: False)  # unattended: risky actions are refused, never assumed
    start = time.time()
    rec = agent.run(t["goal"])
    time.sleep(1.0)
    ok, why = check(t["check"], parser)
    closed = close_new_windows(before)
    if closed:
        print(f"   closed new windows: {closed}", flush=True)
    time.sleep(2)
    return {"success": ok, "check": why, "agent_outcome": rec.outcome, "agent_summary": rec.summary,
            "steps": len(rec.steps), "seconds": round(time.time() - start, 1),
            "planner_calls": sum(1 for st in rec.steps if st.source == "planner"),
            "replayed": sum(1 for st in rec.steps if st.source == "replay"),
            "report": str(rec.dir / "report.html")}


def run(ids: list[str], with_replay: bool = False) -> None:
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))
    if ids:
        tasks = [t for t in tasks if t["id"] in ids]
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    SANDBOX.mkdir()
    if REPLAY_DIR.exists():
        shutil.rmtree(REPLAY_DIR)  # first attempts always plan from scratch
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parser = ScreenParser()
    for t in tasks:
        result = {"id": t["id"], "goal": t["goal"], **attempt(t, parser)}
        print(f"== {t['id']}: {'PASS' if result['success'] else 'FAIL'} ({result['check']}) "
              f"agent said {result['agent_outcome']}, {result['seconds']}s", flush=True)
        if with_replay and result["success"]:
            if "file" in t["check"]:  # remove the output so the replay has to produce it again
                Path(t["check"]["file"][0]).unlink(missing_ok=True)
            r2 = attempt(t, parser)
            result["replay"] = {k: r2[k] for k in ("success", "check", "seconds", "replayed", "planner_calls")}
            print(f"   replay: {'PASS' if r2['success'] else 'FAIL'} in {r2['seconds']}s "
                  f"({r2['replayed']} replayed actions)", flush=True)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")


def report() -> None:
    rows: dict[str, dict] = {}
    for line in OUT.open(encoding="utf-8"):
        r = json.loads(line)
        rows[r["id"]] = r  # keep the latest attempt per task
    vals = list(rows.values())
    passed = sum(r["success"] for r in vals)
    false_done = sum(r["agent_outcome"] == "done" and not r["success"] for r in vals)
    print(f"{passed}/{len(vals)} tasks passed; agent claimed done but the check failed: {false_done}")
    replays = [r for r in vals if "replay" in r]
    if replays:
        ok = [r for r in replays if r["replay"]["success"]]
        if ok:
            speed = sum(r["seconds"] for r in ok) / sum(r["replay"]["seconds"] for r in ok)
            print(f"replay: {len(ok)}/{len(replays)} passed again, {speed:.1f}x faster in total time")
    print("| Task | Result | Steps | Time | Replay | Check |\n|---|---|---|---|---|---|")
    for r in vals:
        verdict = "pass" if r["success"] else "fail"
        rp = r.get("replay")
        rtxt = f"{'pass' if rp['success'] else 'fail'} {rp['seconds']:.0f}s" if rp else "-"
        print(f"| {r['id']} | {verdict} | {r['steps']} | {r['seconds']:.0f}s | {rtxt} | {r['check']} |")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        run([a for a in sys.argv[1:] if not a.startswith("-")], with_replay="--replay" in sys.argv)
