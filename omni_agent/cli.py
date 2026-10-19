"""Command line: `omni-agent run "open notepad and type hello"`."""

from __future__ import annotations

import argparse
import sys
import webbrowser

from omni_agent.config import Settings


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252 and choke on OCR'd symbols
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="omni-agent", description="Pixels-only computer-use agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="carry out a goal on this computer")
    r.add_argument("goal")
    r.add_argument("--backend", choices=["ollama", "gemini"])
    r.add_argument("--model", help="Ollama or Gemini model name")
    r.add_argument("--max-steps", type=int)
    r.add_argument("--vision", choices=["auto", "always", "never"],
                   help="when the planner sees the screenshot (auto = only when OCR text is not enough)")
    r.add_argument("--monitor", type=int, help="mss monitor number (1 = primary)")
    r.add_argument("--dry-run", action="store_true", help="plan and locate targets, but never touch input")
    r.add_argument("--no-confirm", action="store_true", help="do not ask before risky actions")
    r.add_argument("--allow", action="append", default=[], help="only act while this window title is active")
    r.add_argument("--delay", type=float, default=3.0, help="seconds before the first step")
    r.add_argument("--open", action="store_true", help="open the HTML report when finished")
    r.add_argument("--no-replay", action="store_true", help="plan from scratch even if this goal succeeded before")

    f = sub.add_parser("forget", help="delete the stored trajectory for a goal")
    f.add_argument("goal")

    s = sub.add_parser("see", help="show what the agent perceives on the current screen")
    s.add_argument("--monitor", type=int, default=1)
    s.add_argument("--out", default="see.png")

    a = p.parse_args(argv)
    if a.cmd == "see":
        return _see(a.monitor, a.out)
    if a.cmd == "forget":
        import shutil

        from omni_agent.replay import TrajectoryStore

        path = TrajectoryStore(Settings().trajectories_dir).path(a.goal)
        if path.exists():
            shutil.rmtree(path)
            print(f"deleted {path}")
        else:
            print("no stored trajectory for that goal")
        return 0

    settings = Settings()
    if a.backend:
        settings.backend = a.backend
    if a.model:
        if settings.backend == "gemini":
            settings.gemini_model = a.model
        else:
            settings.ollama_model = a.model
    if a.max_steps:
        settings.max_steps = a.max_steps
    if a.vision:
        settings.vision = a.vision
    if a.monitor:
        settings.monitor = a.monitor
    settings.dry_run = a.dry_run
    settings.confirm_risky = not a.no_confirm
    settings.allowed_apps = a.allow
    settings.replay = not a.no_replay

    import time

    from omni_agent.agent import Agent

    agent = Agent(settings)
    if a.delay > 0:
        print(f"Starting in {a.delay:.0f}s - switch to the screen you want it to use.")
        time.sleep(a.delay)
    rec = agent.run(a.goal)
    if a.open:
        webbrowser.open((rec.dir / "report.html").resolve().as_uri())
    return 0 if rec.outcome == "done" else 1


def _see(monitor: int, out: str) -> int:
    from omni_agent import screen
    from omni_agent.perception import ScreenParser, annotate, describe

    frame = screen.capture(monitor)
    parser = ScreenParser()
    elements = parser.parse(frame.image)
    annotate(frame.image, elements).save(out)
    print(describe(elements, frame.size))
    print(f"\n{len(elements)} elements, timing {parser.last_timing}. Annotated screenshot: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
