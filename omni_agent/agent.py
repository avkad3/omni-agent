"""The perceive -> plan -> ground -> act -> verify loop."""

from __future__ import annotations

import time
from collections.abc import Callable

from rich.console import Console

from omni_agent import screen
from omni_agent.actions import POINTING, Action, Executor, is_risky
from omni_agent.config import Settings
from omni_agent.grounding import Grounder
from omni_agent.models import make_model
from omni_agent.perception import ScreenParser, annotate, describe
from omni_agent.planner import Planner, StepRecord
from omni_agent.recorder import Recorder, StepLog
from omni_agent.safety import KillSwitch, allowed, confirm, foreground_title

console = Console()


class Agent:
    def __init__(self, settings: Settings, model=None, parser: ScreenParser | None = None,
                 executor: Executor | None = None, capture: Callable[[], screen.Frame] | None = None,
                 ask: Callable[[str], bool] = confirm):
        self.s = settings
        self.model = model or make_model(settings)
        self.parser = parser or ScreenParser()
        self.executor = executor or Executor(settings.type_interval, dry_run=settings.dry_run)
        self.capture = capture or (lambda: screen.capture(settings.monitor))
        self.ask = ask
        self.planner = Planner(self.model, settings.max_steps)
        self.grounder = Grounder(self.model, (settings.model_width, settings.model_height))
        self.kill = KillSwitch()

    def run(self, goal: str) -> Recorder:
        s = self.s
        rec = Recorder(s.runs_dir, goal, {"model": getattr(self.model, "name", "?"), "dry_run": s.dry_run})
        history: list[StepRecord] = []
        self.kill.start()
        outcome, summary = "fail", "step limit reached"
        console.print(f"[bold]Goal:[/] {goal}   [dim](Ctrl+Alt+Q or mouse to a screen corner to abort)[/]")
        try:
            for step in range(1, s.max_steps + 1):
                if self.kill.triggered:
                    outcome, summary = "aborted", "kill switch pressed"
                    break
                timings: dict[str, float] = {}

                t = time.perf_counter()
                frame = self.capture()
                elements = self.parser.parse(frame.image)
                timings["perceive"] = time.perf_counter() - t

                view = frame.model_view(s.model_width, s.model_height)
                scale = s.model_width / frame.size[0]
                annotated = annotate(view, elements, scale=scale)

                t = time.perf_counter()
                action = self.planner.next_action(goal, annotated, describe(elements, frame.size), history, step)
                timings["plan"] = time.perf_counter() - t
                console.print(f"[cyan]{step}.[/] {action.action} "
                              f"{action.target or action.text or '+'.join(action.keys)} [dim]- {action.thought}[/]")

                if action.action in {"done", "fail"}:
                    rec.add(StepLog(step, action.model_dump(), action.summary, raw=self.planner.last_raw,
                                    timings=timings), annotated, frame.size)
                    outcome, summary = action.action, action.summary
                    break

                # Launching goes through the Start menu, so only guard actions that touch the active window.
                guarded = action.action in POINTING | {"type", "hotkey", "scroll"}
                if guarded and s.allowed_apps and not allowed(foreground_title(), s.allowed_apps):
                    outcome, summary = "aborted", f"active window '{foreground_title()}' is not in the allow-list"
                    break

                result, method, click = self._act(action, frame, elements, timings)
                change = None
                if action.action != "wait" and not s.dry_run:
                    time.sleep(s.action_pause)
                    change = screen.change_ratio(frame, self.capture())
                    if change < 0.002 and action.action not in {"type"}:
                        result += " (no visible change)"
                history.append(StepRecord(action, result))
                rec.add(StepLog(step, action.model_dump(), result, method, click, change, timings,
                                raw=self.planner.last_raw), annotated, frame.size)
                console.print(f"   [green]->[/] {result}")
                if result.startswith("aborted"):
                    outcome, summary = "aborted", result
                    break
        except Exception as e:  # pyautogui.FailSafeException lands here too
            outcome, summary = ("aborted", "fail-safe corner") if "FailSafe" in type(e).__name__ else ("error", repr(e))
        finally:
            self.kill.stop()
        page = rec.finish(outcome, summary)
        console.print(f"[bold]{outcome}[/]: {summary}\nReport: {page}")
        return rec

    def _act(self, a: Action, frame: screen.Frame, elements, timings) -> tuple[str, str, tuple[float, float] | None]:
        ex = self.executor
        if a.action in POINTING:
            t = time.perf_counter()
            g = self.grounder.ground(frame.image, elements, a.target, a.element_id)
            timings["ground"] = time.perf_counter() - t
            if g is None:
                return f"could not locate '{a.target}' on screen", "", None
            label = g.element.text if g.element is not None else a.target
            if self.s.confirm_risky and is_risky(a, label) and not self.ask(f"About to {a.action} '{label}'."):
                return "aborted: user declined risky action", g.method, (g.x, g.y)
            ex.point(a.action, *frame.to_screen(g.x, g.y))
            return f"{a.action.replace('_', ' ')}ed '{label}'", g.method, (g.x, g.y)
        if a.action == "type":
            if self.s.confirm_risky and is_risky(a) and not self.ask(f"About to type '{a.text}'."):
                return "aborted: user declined risky action", "", None
            ex.type_text(a.text)
            return f"typed {len(a.text)} chars", "", None
        if a.action == "hotkey":
            if self.s.confirm_risky and is_risky(a) and not self.ask(f"About to press {'+'.join(a.keys)}."):
                return "aborted: user declined risky action", "", None
            ex.hotkey(a.keys)
            return f"pressed {'+'.join(a.keys)}", "", None
        if a.action == "scroll":
            ex.scroll(a.direction, a.amount, frame.to_screen(frame.size[0] / 2, frame.size[1] / 2))
            return f"scrolled {a.direction}", "", None
        if a.action == "launch":
            ex.launch(a.text)
            time.sleep(1.5)
            return f"launched '{a.text}' from Start", "", None
        if a.action == "wait":
            time.sleep(min(a.seconds, 10))
            return f"waited {a.seconds}s", "", None
        return f"unsupported action {a.action}", "", None
