"""Decides what to do: splits the goal into steps, picks the next action, and checks the result."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from omni_agent.actions import Action, extract_json, parse_action

SYSTEM = """You control a Windows computer using only the screen. Each turn you get the list of numbered
text/icon elements on screen (and sometimes the screenshot with those numbers drawn on it), the goal,
the plan, the current step, and what you already did. Choose ONE next action for the current step.
Reply with a single JSON object and nothing else.

Actions:
- {"action":"click"|"double_click"|"right_click", "element_id":<id or null>, "target":"<what it is>"}
  Use element_id when a numbered element is the target. Always fill "target" with its visible text
  or a short description (e.g. "Save", "the close X at top right").
- {"action":"type", "text":"<text>"}  types into the focused field. Click the field first if needed.
- {"action":"hotkey", "keys":["ctrl","s"]}  keyboard shortcut or a single key like ["enter"].
- {"action":"scroll", "direction":"up"|"down", "amount":5}
- {"action":"launch", "text":"<app name>"}  opens an app from the Start menu.
- {"action":"wait", "seconds":2}
- {"action":"done", "summary":"<what was achieved>"}  the CURRENT STEP is complete.
- {"action":"fail", "summary":"<why>"}  if the goal is impossible.

Always include "thought": one short sentence on what you see and why this action.
Rules:
1. If the goal names an app and that app's window is not clearly on screen, launch it first.
   Never type into whatever window happens to be open.
2. Prefer keyboard shortcuts when they are reliable. Do not repeat an action that had no effect."""

NO_IMAGE = ("You are not given the screenshot this turn; decide from the element list. If you cannot "
            "tell what is on screen, use {\"action\":\"wait\",\"seconds\":1,\"thought\":\"need to look\"}.\n")

USER = """Goal: {goal}
Plan: {plan}
Current step: {current}
Step {step} of at most {max_steps}.
{vision_note}Screen elements (id, kind, text, position as fraction of width,height):
{elements}

History (oldest first):
{history}

What is the next action for the current step? JSON only."""

DECOMPOSE = """Split this computer task into short steps a person would do. Use ONLY these step forms:
open <app name>
type "<exact text>"
press <keys>              (e.g. press ctrl+s, press enter, press alt+f4)
click <visible text or short description of the thing>
double-click <...> / right-click <...>
scroll up / scroll down
wait <n> seconds

Examples
Task: Save a note saying hi in Notepad as test.txt
{{"steps": ["open Notepad", "type \\"hi\\"", "press ctrl+s", "wait 1 seconds", "type \\"test.txt\\"", "press enter"]}}
Task: Calculate 12 times 7 in Calculator
{{"steps": ["open Calculator", "type \\"12*7\\"", "press enter"]}}
Task: Turn on dark mode in Settings
{{"steps": ["open Settings", "click Personalization", "click Colors", "click Choose your mode", "click Dark"]}}

Task: {goal}
Reply with JSON only: {{"steps": [...]}}"""

CHECK = """Goal: {goal}
Actions taken:
{history}
Screen elements:
{elements}
Look at the screenshot. Is the goal visibly complete? Reply JSON only:
{{"done": true|false, "reason": "<one sentence>"}}"""


@dataclass
class StepRecord:
    action: Action
    result: str  # what happened, e.g. "clicked 'Save' via text", "no visible change"

    def line(self, i: int) -> str:
        a = self.action
        what = {
            "type": f'type "{a.text}"', "hotkey": f"hotkey {'+'.join(a.keys)}", "launch": f'launch "{a.text}"',
            "scroll": f"scroll {a.direction} {a.amount}", "wait": f"wait {a.seconds}s",
        }.get(a.action, f"{a.action} {a.target or ''} (id {a.element_id})")
        return f"{i}. {what} -> {self.result}"


def history_text(history: list[StepRecord], last: int = 8) -> str:
    start = max(0, len(history) - last)
    return "\n".join(r.line(start + i + 1) for i, r in enumerate(history[start:])) or "(nothing yet)"


class Planner:
    def __init__(self, model, max_steps: int):
        self.model = model
        self.max_steps = max_steps
        self.last_raw = ""

    def decompose(self, goal: str) -> list[str]:
        """One cheap text-only call. Falls back to treating the whole goal as a single step."""
        try:
            raw = self.model.generate(DECOMPOSE.format(goal=goal), [])
            steps = extract_json(raw).get("steps", [])
            steps = [str(s).strip() for s in steps if str(s).strip()]
            return steps[:20] or [goal]
        except Exception:
            return [goal]

    def next_action(self, goal: str, annotated: Image.Image | None, elements_text: str,
                    history: list[StepRecord], step: int, plan: list[str] | None = None,
                    current: str = "") -> Action:
        """Pass annotated=None for a fast text-only turn (no image encoding)."""
        prompt = USER.format(goal=goal, plan=" -> ".join(plan or [goal]), current=current or goal, step=step,
                             max_steps=self.max_steps, elements=elements_text or "(none)",
                             history=history_text(history), vision_note="" if annotated is not None else NO_IMAGE)
        images = [annotated] if annotated is not None else []
        err = ""
        for _ in range(2):
            self.last_raw = self.model.generate(prompt + err, images, system=SYSTEM)
            try:
                return parse_action(self.last_raw)
            except ValueError as e:
                err = f"\n\nYour previous reply was invalid ({e}). Reply with one JSON object using the actions listed."
        return Action(action="fail", summary=f"planner gave no valid action: {self.last_raw[:150]}")

    def check_done(self, goal: str, annotated: Image.Image, elements_text: str,
                   history: list[StepRecord]) -> tuple[bool, str]:
        raw = self.model.generate(CHECK.format(goal=goal, history=history_text(history, 12),
                                               elements=elements_text), [annotated])
        try:
            obj = extract_json(raw)
            done = obj.get("done")
            done = done if isinstance(done, bool) else str(done).lower() in {"true", "yes", "1"}
            return done, str(obj.get("reason", ""))
        except ValueError:
            return False, f"unclear verdict: {raw[:100]}"
