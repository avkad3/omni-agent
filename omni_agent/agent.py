"""The perceive -> plan -> ground -> act -> verify loop.

Hierarchical: the goal is split once into short steps. Steps that map straight onto an action
(open / type / press / click <named text>) run without consulting the model; the model is asked
only for steps that need judgment, and sees the screenshot only when the OCR text is not enough.
"""

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
from omni_agent.perception.detector import load_detector
from omni_agent.planner import Planner, StepRecord
from omni_agent.recorder import Recorder, StepLog
from omni_agent.safety import KillSwitch, allowed, confirm, foreground_title
from omni_agent.skills import direct_action, existing_file_in_title, make_safe_plan

console = Console()


class Agent:
    def __init__(self, settings: Settings, model=None, parser: ScreenParser | None = None,
                 executor: Executor | None = None, capture: Callable[[], screen.Frame] | None = None,
                 ask: Callable[[str], bool] = confirm):
        self.s = settings
        self.model = model or make_model(settings)
        self.parser = parser or ScreenParser(detector=load_detector() if settings.icons else None)
        self.executor = executor or Executor(settings.type_interval, dry_run=settings.dry_run)
        self.capture = capture or (lambda: screen.capture(settings.monitor))
        self.ask = ask
        self.title = foreground_title  # safety checks only; planning never uses window metadata
        self.planner = Planner(self.model, settings.max_steps)
        self.grounder = Grounder(self.model, (settings.model_width, settings.model_height))
        self.kill = KillSwitch()

    def run(self, goal: str) -> Recorder:
        s = self.s
        console.print(f"[bold]Goal:[/] {goal}   [dim](Ctrl+Alt+Q or mouse to a screen corner to abort)[/]")
        t = time.perf_counter()
        plan = make_safe_plan(self.planner.decompose(goal)) if s.decompose else [goal]
        console.print(f"[bold]Plan[/] ({time.perf_counter() - t:.1f}s): " + " -> ".join(plan))
        rec = Recorder(s.runs_dir, goal, {"model": getattr(self.model, "name", "?"), "dry_run": s.dry_run,
                                          "plan": plan})
        history: list[StepRecord] = []
        idx = 0
        direct_failed: set[int] = set()  # plan steps whose direct execution could not ground
        verified = False
        self.kill.start()
        outcome, summary = "fail", "step limit reached"
        try:
            for step in range(1, s.max_steps + 1):
                if self.kill.triggered:
                    outcome, summary = "aborted", "kill switch pressed"
                    break
                timings: dict[str, float] = {}

                t = time.perf_counter()
                frame = self.capture()
                elements = self.parser.parse(frame.image, icons=False)
                timings["perceive"] = time.perf_counter() - t
                current = plan[idx] if idx < len(plan) else ""
                action = None if idx in direct_failed or idx >= len(plan) else direct_action(current)
                look = idx >= len(plan) or (action is None and (idx in direct_failed
                                                                 or self._needs_image(elements, history)))
                if look:  # the model will see the screen: number the icons too
                    t = time.perf_counter()
                    elements = self.parser.parse(frame.image, icons=True)
                    timings["detect_icons"] = time.perf_counter() - t
                view = frame.model_view(s.model_width, s.model_height)
                annotated = annotate(view, elements, scale=s.model_width / frame.size[0])
                elements_text = describe(elements, frame.size)

                if idx >= len(plan):  # every step ran: confirm visually before declaring success
                    if not s.verify_end or verified:
                        outcome, summary = "done", "all plan steps carried out"
                        break
                    t = time.perf_counter()
                    ok, reason = self.planner.check_done(goal, annotated, elements_text, history)
                    timings["verify"] = time.perf_counter() - t
                    verified = True
                    rec.add(StepLog(step, {"action": "check", "thought": reason}, f"goal complete: {ok}",
                                    timings=timings, subtask="final check"), annotated, frame.size)
                    console.print(f"[cyan]{step}.[/] final check -> {ok} [dim]- {reason}[/]")
                    if ok:
                        outcome, summary = "done", reason
                        break
                    plan.append(f"finish the goal: {goal}")  # hand the rest to free planning
                    continue

                source = "skill"
                if action is None:
                    source = "planner"
                    t = time.perf_counter()
                    action = self.planner.next_action(goal, annotated if look else None, elements_text, history,
                                                      step, plan, current)
                    timings["plan_with_image" if look else "plan_text_only"] = time.perf_counter() - t
                console.print(f"[cyan]{step}.[/] [dim]\\[{current}][/] {action.action} "
                              f"{action.target or action.text or '+'.join(action.keys)} [dim]({source})[/]")

                if action.action == "done":
                    rec.add(StepLog(step, action.model_dump(), "step complete", timings=timings, subtask=current,
                                    raw=self.planner.last_raw), annotated, frame.size)
                    idx += 1
                    continue
                if action.action == "fail":
                    rec.add(StepLog(step, action.model_dump(), action.summary, timings=timings, subtask=current,
                                    raw=self.planner.last_raw), annotated, frame.size)
                    outcome, summary = "fail", action.summary
                    break

                # Launching goes through the Start menu, so only guard actions that touch the active window.
                guarded = action.action in POINTING | {"type", "hotkey", "scroll"}
                if guarded and s.allowed_apps and not allowed(foreground_title(), s.allowed_apps):
                    outcome, summary = "aborted", f"active window '{foreground_title()}' is not in the allow-list"
                    break

                if action.action == "type" and not s.dry_run:
                    existing = existing_file_in_title(self.title(), goal)
                    if existing and not self.ask(f"About to type into the existing file '{existing}', "
                                                 "which the goal does not mention."):
                        outcome, summary = "aborted", f"refused to type into existing file '{existing}'"
                        break

                result, method, click = self._act(action, frame, elements, timings)
                change = None
                if action.action != "wait" and not s.dry_run:
                    time.sleep(s.action_pause)
                    change = screen.change_ratio(frame, self.capture())
                    if change < 0.002 and action.action != "type":
                        result += " (no visible change)"
                history.append(StepRecord(action, result))
                rec.add(StepLog(step, action.model_dump(), result, method, click, change, timings,
                                raw=self.planner.last_raw if source == "planner" else "", subtask=current,
                                source=source), annotated, frame.size)
                console.print(f"   [green]->[/] {result}")
                if result.startswith("aborted"):
                    outcome, summary = "aborted", result
                    break

                if result.startswith("could not locate"):
                    direct_failed.add(idx)  # retry this step with the model looking at the screen
                elif source == "skill" or direct_action(current) is not None:
                    idx += 1  # one-action steps are complete once the action ran
        except Exception as e:  # pyautogui.FailSafeException lands here too
            outcome, summary = ("aborted", "fail-safe corner") if "FailSafe" in type(e).__name__ else ("error", repr(e))
        finally:
            self.kill.stop()
        page = rec.finish(outcome, summary)
        console.print(f"[bold]{outcome}[/]: {summary}\nReport: {page}")
        return rec

    def _needs_image(self, elements, history: list[StepRecord]) -> bool:
        """Adaptive perception: send the screenshot only when the OCR element list is not enough."""
        mode = self.s.vision
        if mode in ("always", "never"):
            return mode == "always"
        if len(elements) < self.s.min_elements_for_text:
            return True  # canvas, game or loading screen: text alone says little
        if history:
            last = history[-1]
            if "no visible change" in last.result or "could not locate" in last.result:
                return True
            if last.action.action == "wait" and "look" in last.action.thought.lower():
                return True  # the planner asked to see the screen
        return False

    def _act(self, a: Action, frame: screen.Frame, elements, timings) -> tuple[str, str, tuple[float, float] | None]:
        ex = self.executor
        if a.action in POINTING:
            t = time.perf_counter()
            g = self.grounder.ground(frame.image, elements, a.target, a.element_id, use_model=False)
            if g is None:  # no text match: add detected icons, then let the model point and snap
                if not any(e.kind == "icon" for e in elements):
                    elements = self.parser.parse(frame.image, icons=True)
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
