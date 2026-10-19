"""The agent's action space, parsing of model replies into actions, and execution."""

from __future__ import annotations

import json
import re
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

ActionType = Literal[
    "click", "double_click", "right_click", "type", "hotkey", "scroll", "launch", "wait", "done", "fail"
]
POINTING = {"click", "double_click", "right_click"}


class Action(BaseModel):
    thought: str = ""
    action: ActionType
    element_id: int | None = None  # id from the numbered screen elements
    target: str = ""  # plain-language description of what to point at, used when no id fits
    text: str = ""  # for type / launch
    keys: list[str] = Field(default_factory=list)  # for hotkey, e.g. ["ctrl", "s"]
    direction: Literal["up", "down"] = "down"
    amount: int = 5
    seconds: float = 1.0
    summary: str = ""  # for done / fail

    @field_validator("keys", mode="before")
    @classmethod
    def _split_keys(cls, v):
        if isinstance(v, str):
            return [k.strip().lower() for k in re.split(r"[+,\s]+", v) if k.strip()]
        return [str(k).strip().lower() for k in v or []]

    @field_validator("element_id", mode="before")
    @classmethod
    def _id(cls, v):
        if v in (None, "", "null", "none", -1, "-1"):
            return None
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            return int(m.group()) if m else None
        return v


_ALIASES = {
    "left_click": "click", "tap": "click", "press": "hotkey", "key": "hotkey", "keypress": "hotkey",
    "input": "type", "type_text": "type", "write": "type", "open": "launch", "open_app": "launch",
    "finish": "done", "finished": "done", "complete": "done", "stop": "done", "doubleclick": "double_click",
    "rightclick": "right_click", "sleep": "wait",
}


def extract_json(raw: str) -> dict:
    """Pull the first JSON object out of a model reply (handles code fences and chatter)."""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fence:
        raw = fence.group(1)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        depth = 0
        for i in range(start, len(raw)) if start >= 0 else []:
            depth += {"{": 1, "}": -1}.get(raw[i], 0)
            if depth == 0:
                obj = json.loads(raw[start : i + 1])
                break
        else:
            raise ValueError(f"no JSON object in reply: {raw[:200]!r}") from None
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj


def parse_action(raw: str) -> Action:
    obj = extract_json(raw)
    if "action" not in obj:
        for k in ("type", "name", "action_type", "next_action"):
            if k in obj and isinstance(obj[k], str):
                obj["action"] = obj.pop(k)
                break
    a = str(obj.get("action", "")).strip().lower().replace(" ", "_")
    obj["action"] = _ALIASES.get(a, a)
    if obj["action"] == "hotkey" and not obj.get("keys") and obj.get("key"):
        obj["keys"] = obj["key"]
    try:
        return Action.model_validate(obj)
    except ValidationError as e:
        raise ValueError(f"invalid action {obj}: {e.errors()[0]['msg']}") from e


RISKY_WORDS = re.compile(
    r"\b(delete|remove|erase|format|uninstall|send|submit|pay|purchase|buy|checkout|transfer|"
    r"shutdown|shut down|restart|sign out|log out)\b",
    re.I,
)


def is_risky(action: Action, target_label: str = "") -> bool:
    if action.action == "hotkey" and set(action.keys) & {"delete", "del"} and "shift" in action.keys:
        return True  # shift+delete skips the recycle bin
    blob = " ".join([action.target, action.text, target_label, action.thought if action.action in POINTING else ""])
    return action.action in POINTING | {"hotkey", "type"} and bool(RISKY_WORDS.search(blob))


class Executor:
    """Performs actions with PyAutoGUI (or pydirectinput for games that read raw input)."""

    def __init__(self, type_interval: float = 0.02, direct_input: bool = False, dry_run: bool = False):
        self.dry_run = dry_run
        self.type_interval = type_interval
        if dry_run:
            self.gui = None
            return
        import pyautogui

        pyautogui.FAILSAFE = True  # slam the mouse into a screen corner to abort
        pyautogui.PAUSE = 0.05
        self.gui = pyautogui
        self.direct = None
        if direct_input:
            import pydirectinput

            pydirectinput.FAILSAFE = True
            self.direct = pydirectinput

    def point(self, kind: str, x: int, y: int) -> None:
        if self.dry_run:
            return
        dev = self.direct or self.gui
        dev.moveTo(x, y, duration=0.15) if dev is self.gui else dev.moveTo(x, y)
        {"click": dev.click, "double_click": dev.doubleClick, "right_click": dev.rightClick}[kind](x, y)

    def type_text(self, text: str) -> None:
        if self.dry_run:
            return
        if text.isascii():
            self.gui.write(text, interval=self.type_interval)
        else:  # pyautogui cannot type non-ASCII; paste it instead
            self._paste(text)

    def _paste(self, text: str) -> None:
        import subprocess

        subprocess.run(["clip"], input=text.encode("utf-16le"), check=True)
        self.gui.hotkey("ctrl", "v")

    def hotkey(self, keys: list[str]) -> None:
        if self.dry_run or not keys:
            return
        keys = [{"control": "ctrl", "windows": "win", "cmd": "win", "return": "enter", "escape": "esc"}.get(k, k)
                for k in keys]
        self.gui.hotkey(*keys)

    def scroll(self, direction: str, amount: int, at: tuple[int, int] | None = None) -> None:
        if self.dry_run:
            return
        clicks = (1 if direction == "up" else -1) * max(1, amount) * 120
        if at:
            self.gui.moveTo(*at, duration=0.1)
        self.gui.scroll(clicks)

    def launch(self, app: str) -> None:
        """Open an app the way a person would: Start menu, type the name, Enter."""
        if self.dry_run:
            return
        self.gui.press("win")
        time.sleep(0.8)
        self.type_text(app)
        time.sleep(1.2)
        self.gui.press("enter")
