"""The perceive -> plan -> ground -> act -> verify loop.

Hierarchical: the goal is split once into short steps. Steps that map straight onto an action
(open / type / press / click <named text>) run without consulting the model; the model is asked
only for steps that need judgment, and sees the screenshot only when the OCR text is not enough.

A goal that already succeeded is replayed from its stored trajectory without any model call, and
handed back to this loop at the first step whose target cannot be re-found.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from PIL import Image
from rich.console import Console

from omni_agent import screen
from omni_agent.actions import POINTING, Action, Executor, is_risky
from omni_agent.config import Settings
from omni_agent.grounding import Grounder, text_score
from omni_agent.models import make_model
from omni_agent.perception import ScreenParser, annotate, describe
from omni_agent.perception.detector import load_detector
from omni_agent.planner import Planner, StepRecord
from omni_agent.recorder import Recorder, StepLog
from omni_agent.replay import Trajectory, TrajectoryStore, TrajStep, crop_patch, find_patch
from omni_agent.safety import KillSwitch, allowed, confirm, foreground_title
from omni_agent.skills import direct_action, existing_file_in_title, make_safe_plan
from omni_agent.windows import focus_app

console = Console()


class Abort(Exception):
    """Stop the run with outcome 'aborted' and this message."""


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
        self.focus = focus_app  # brings a launched app forward; Windows blocks focus stealing otherwise
        self.planner = Planner(self.model, settings.max_steps)
        self.grounder = Grounder(self.model, (settings.model_width, settings.model_height))
        self.store = TrajectoryStore(settings.trajectories_dir) if settings.replay else None
        self.kill = KillSwitch()

    # ------------------------------------------------------------------ main loop

    def run(self, goal: str) -> Recorder:
        s = self.s
        console.print(f"[bold]Goal:[/] {goal}   [dim](Ctrl+Alt+Q or mouse to a screen corner to abort)[/]")
        traj = self.store.load(goal) if self.store is not None and not s.dry_run else None
        t = time.perf_counter()
        if traj is not None:
            plan = list(traj.plan)
            console.print(f"[bold]Replaying[/] a stored run ({len(traj.steps)} actions): " + " -> ".join(plan))
        else:
            plan = make_safe_plan(self.planner.decompose(goal)) if s.decompose else [goal]
            console.print(f"[bold]Plan[/] ({time.perf_counter() - t:.1f}s): " + " -> ".join(plan))
        rec = Recorder(s.runs_dir, goal, {"model": getattr(self.model, "name", "?"), "dry_run": s.dry_run,
                                          "plan": plan, "replay": traj is not None})
        history: list[StepRecord] = []
        executed: list[tuple[TrajStep, Image.Image | None]] = []  # becomes the stored trajectory
        idx = 0
        direct_failed: set[int] = set()  # plan steps whose direct execution could not ground
        checks = 0
        self.kill.start()
        outcome, summary = "fail", "step limit reached"
        try:
            if traj is not None:
                idx = self._replay(traj, goal, rec, history, executed)
                if idx < len(plan):
                    console.print(f"[yellow]Replay lost its anchor at plan step {idx + 1}; handing over to the "
                                  "agent[/]")
            first = len(rec.steps) + 1
            for step in range(first, first + s.max_steps):
                if self.kill.triggered:
                    raise Abort("kill switch pressed")
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
                elements_text = describe(elements, frame.size, limit=s.max_prompt_elements)

                if idx >= len(plan):  # every step ran: confirm visually before declaring success
                    if not s.verify_end:
                        outcome, summary = "done", "all plan steps carried out (not verified)"
                        break
                    if checks >= 2:
                        outcome, summary = "fail", "goal still not visibly complete after a second attempt"
                        break
                    t = time.perf_counter()
                    ok, reason = self.planner.check_done(goal, annotated, describe(
                        [e for e in elements if e.kind == "text"], frame.size, limit=60), history)
                    timings["verify"] = time.perf_counter() - t
                    checks += 1
                    rec.add(StepLog(step, {"action": "check", "thought": reason}, f"goal complete: {ok}",
                                    timings=timings, subtask="final check"), annotated, frame.size)
                    console.print(f"[cyan]{step}.[/] final check -> {ok} [dim]- {reason}[/]")
                    if ok:
                        outcome, summary = "done", reason
                        break
                    if checks >= 2:
                        outcome, summary = "fail", f"goal still not visibly complete after a second attempt: {reason}"
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

                self._guard(action, goal)
                result, method, click, anchor = self._act(action, frame, elements, timings)
                result = self._settle(action, frame, result)
                history.append(StepRecord(action, result))
                rec.add(StepLog(step, action.model_dump(), result, method, click, self._last_change, timings,
                                raw=self.planner.last_raw if source == "planner" else "", subtask=current,
                                source=source), annotated, frame.size)
                console.print(f"   [green]->[/] {result}")
                if result.startswith("aborted"):
                    raise Abort(result)

                if result.startswith("could not locate"):
                    direct_failed.add(idx)  # retry this step with the model looking at the screen
                    continue
                executed.append(self._traj_step(idx, action, frame, click, anchor))
                if source == "skill" or direct_action(current) is not None:
                    idx += 1  # one-action steps are complete once the action ran
        except Abort as e:
            outcome, summary = "aborted", str(e)
        except Exception as e:  # pyautogui.FailSafeException lands here too
            outcome, summary = ("aborted", "fail-safe corner") if "FailSafe" in type(e).__name__ else ("error", repr(e))
        finally:
            self.kill.stop()
        if outcome == "done" and self.store is not None and not s.dry_run and executed:
            saved = self.store.save(goal, plan[: len(plan) - plan.count(f"finish the goal: {goal}")], executed)
            console.print(f"[dim]Stored trajectory for replay: {saved}[/]")
        page = rec.finish(outcome, summary)
        console.print(f"[bold]{outcome}[/]: {summary}\nReport: {page}")
        return rec

    # ------------------------------------------------------------------ replay

    def _replay(self, traj: Trajectory, goal: str, rec: Recorder, history: list[StepRecord],
                executed: list[tuple[TrajStep, Image.Image | None]]) -> int:
        """Run stored actions. Returns the plan index to continue from (len(plan) if everything replayed)."""
        for ts in traj.steps:
            if self.kill.triggered:
                raise Abort("kill switch pressed")
            timings: dict[str, float] = {}
            t = time.perf_counter()
            frame = self.capture()
            a = Action.model_validate(ts.action)
            click = None
            if a.action in POINTING:
                found = self._relocate(goal, ts, frame)
                timings["relocate"] = time.perf_counter() - t
                if found is None:
                    return ts.plan_idx
                x, y, how = found
                self._guard(a, goal)
                label = ts.anchor_text or a.target
                if self.s.confirm_risky and is_risky(a, label) and not self.ask(f"About to {a.action} '{label}'."):
                    raise Abort("user declined risky action")
                self.executor.point(a.action, *frame.to_screen(x, y))
                result, click = f"replayed {a.action.replace('_', ' ')} '{label}' ({how})", (x, y)
            else:
                self._guard(a, goal)
                result, _, _, _ = self._act(a, frame, [], timings)
                result = "replayed: " + result
            result = self._settle(a, frame, result)
            history.append(StepRecord(a, result))
            step = len(rec.steps) + 1
            view = frame.model_view(self.s.model_width, self.s.model_height)
            rec.add(StepLog(step, a.model_dump(), result, "replay", click, self._last_change, timings,
                            subtask=traj.plan[ts.plan_idx] if ts.plan_idx < len(traj.plan) else "",
                            source="replay"), view, frame.size)
            console.print(f"[cyan]{step}.[/] {result}")
            if result.startswith("aborted"):
                raise Abort(result)
            executed.append(self._traj_step(ts.plan_idx, a, frame, click, ts.anchor_text))
        return len(traj.plan)

    def _relocate(self, goal: str, ts: TrajStep, frame: screen.Frame) -> tuple[float, float, str] | None:
        """Find a stored click target on the current screen: by its text, else by its image patch."""
        W, H = frame.size
        if ts.anchor_text:
            elements = self.parser.parse(frame.image, icons=False)
            hits = [e for e in elements if e.kind == "text" and text_score(ts.anchor_text, e.text) >= 0.9]
            if hits:
                if ts.rel_xy is not None:  # the same word can appear in several windows: take the closest
                    ox, oy = ts.rel_xy[0] * W, ts.rel_xy[1] * H
                    hits.sort(key=lambda e: (e.center[0] - ox) ** 2 + (e.center[1] - oy) ** 2)
                return (*hits[0].center, "text")
        patch = self.store.patch(goal, ts) if self.store is not None else None
        if patch is not None and ts.rel_xy is not None:
            hit = find_patch(frame.image, patch, (ts.rel_xy[0] * W, ts.rel_xy[1] * H))
            if hit is not None and hit[2] >= self.s.replay_threshold:
                return hit[0], hit[1], f"image match {hit[2]:.2f}"
        return None

    @staticmethod
    def _traj_step(plan_idx: int, a: Action, frame: screen.Frame, click, anchor: str
                   ) -> tuple[TrajStep, Image.Image | None]:
        if click is None:
            return TrajStep(plan_idx, a.model_dump()), None
        W, H = frame.size
        return (TrajStep(plan_idx, a.model_dump(), anchor_text=anchor, rel_xy=(click[0] / W, click[1] / H)),
                crop_patch(frame.image, *click))

    # ------------------------------------------------------------------ helpers

    _last_change: float | None = None

    def _guard(self, action: Action, goal: str) -> None:
        """Safety checks that apply to every action, planned or replayed."""
        s = self.s
        # Launching goes through the Start menu, so only guard actions that touch the active window.
        if action.action in POINTING | {"type", "hotkey", "scroll"} and s.allowed_apps:
            title = self.title()
            if not allowed(title, s.allowed_apps):
                raise Abort(f"active window '{title}' is not in the allow-list")
        if action.action == "type" and not s.dry_run:
            existing = existing_file_in_title(self.title(), goal)
            if existing and not self.ask(f"About to type into the existing file '{existing}', "
                                         "which the goal does not mention."):
                raise Abort(f"refused to type into existing file '{existing}'")

    def _settle(self, action: Action, before: screen.Frame, result: str) -> str:
        """Wait for the UI, then note whether the action visibly changed the screen."""
        self._last_change = None
        if action.action == "wait" or self.s.dry_run or result.startswith(("could not", "aborted")):
            return result
        time.sleep(self.s.action_pause)
        self._last_change = screen.change_ratio(before, self.capture())
        if self._last_change < 0.002 and action.action != "type":
            result += " (no visible change)"
        return result

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

    def _act(self, a: Action, frame: screen.Frame, elements, timings
             ) -> tuple[str, str, tuple[float, float] | None, str]:
        """Returns (result, grounding method, click point, anchor text for replay)."""
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
                return f"could not locate '{a.target}' on screen", "", None, ""
            label = g.element.text if g.element is not None and g.element.text else a.target
            anchor = g.element.text if g.element is not None and g.element.kind == "text" else ""
            if self.s.confirm_risky and is_risky(a, label) and not self.ask(f"About to {a.action} '{label}'."):
                return "aborted: user declined risky action", g.method, (g.x, g.y), ""
            ex.point(a.action, *frame.to_screen(g.x, g.y))
            return f"{a.action.replace('_', ' ')}ed '{label}'", g.method, (g.x, g.y), anchor
        if a.action == "type":
            if self.s.confirm_risky and is_risky(a) and not self.ask(f"About to type '{a.text}'."):
                return "aborted: user declined risky action", "", None, ""
            ex.type_text(a.text)
            return f"typed {len(a.text)} chars", "", None, ""
        if a.action == "hotkey":
            if self.s.confirm_risky and is_risky(a) and not self.ask(f"About to press {'+'.join(a.keys)}."):
                return "aborted: user declined risky action", "", None, ""
            ex.hotkey(a.keys)
            return f"pressed {'+'.join(a.keys)}", "", None, ""
        if a.action == "scroll":
            ex.scroll(a.direction, a.amount, frame.to_screen(frame.size[0] / 2, frame.size[1] / 2))
            return f"scrolled {a.direction}", "", None, ""
        if a.action == "launch":
            ex.launch(a.text)
            time.sleep(1.5)
            front = self.s.dry_run or self.focus(a.text)
            return f"launched '{a.text}' from Start" + ("" if front else " (window not in front)"), "", None, ""
        if a.action == "wait":
            time.sleep(min(a.seconds, 10))
            return f"waited {a.seconds}s", "", None, ""
        return f"unsupported action {a.action}", "", None, ""
