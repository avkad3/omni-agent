"""Decides the next action from the goal, the numbered screenshot and recent history."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from omni_agent.actions import Action, parse_action

SYSTEM = """You control a Windows computer using only the screen. Each turn you get a screenshot with
numbered boxes around text and icons, the list of those elements, the goal, and what you already did.
Choose ONE next action. Reply with a single JSON object and nothing else.

Actions:
- {"action":"click"|"double_click"|"right_click", "element_id":<id or null>, "target":"<what it is>"}
  Use element_id when a numbered box covers the target. Always fill "target" with its visible text
  or a short description (e.g. "Save", "the close X at top right").
- {"action":"type", "text":"<text>"}  types into the focused field. Click the field first if needed.
- {"action":"hotkey", "keys":["ctrl","s"]}  keyboard shortcut or a single key like ["enter"].
- {"action":"scroll", "direction":"up"|"down", "amount":5}
- {"action":"launch", "text":"<app name>"}  opens an app from the Start menu.
- {"action":"wait", "seconds":2}
- {"action":"done", "summary":"<what was achieved>"}  only when the goal is visibly complete.
- {"action":"fail", "summary":"<why>"}  if the goal is impossible.

Always include "thought": one short sentence on what you see and why this action.
Prefer keyboard shortcuts when they are reliable. Do not repeat an action that had no effect."""

USER = """Goal: {goal}
Step {step} of at most {max_steps}.
Screen elements (id, kind, text, position as fraction of width,height):
{elements}

History (oldest first):
{history}

What is the next action? JSON only."""


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


class Planner:
    def __init__(self, model, max_steps: int):
        self.model = model
        self.max_steps = max_steps
        self.last_raw = ""

    def next_action(self, goal: str, annotated: Image.Image, elements_text: str, history: list[StepRecord],
                    step: int) -> Action:
        hist = "\n".join(r.line(i + 1) for i, r in enumerate(history[-8:])) or "(nothing yet)"
        prompt = USER.format(goal=goal, step=step, max_steps=self.max_steps, elements=elements_text or "(none)",
                             history=hist)
        err = ""
        for _ in range(2):
            self.last_raw = self.model.generate(prompt + err, [annotated], system=SYSTEM)
            try:
                return parse_action(self.last_raw)
            except ValueError as e:
                err = f"\n\nYour previous reply was invalid ({e}). Reply with one JSON object using the actions listed."
        return Action(action="fail", summary=f"planner gave no valid action: {self.last_raw[:150]}")
