"""Sub-tasks that map straight onto actions, so the vision model is only consulted when judgment is needed.

The planner first splits a goal into short imperative steps ("open Notepad", 'type "hi"', "press ctrl+s",
"click Save"). open / type / press / wait / scroll steps are executed directly here; click-style steps
become a click on a named target that grounding resolves (OCR text match first)."""

from __future__ import annotations

import re
from collections.abc import Callable

from omni_agent.actions import Action

_KEY_NAMES = {
    "control": "ctrl", "ctl": "ctrl", "windows": "win", "window": "win", "cmd": "win", "return": "enter",
    "escape": "esc", "del": "delete", "pgdn": "pagedown", "pgup": "pageup", "spacebar": "space",
}
_QUOTED = re.compile(r"[\"“”'‘’`](.+)[\"“”'‘’`]", re.S)


def _keys(spec: str) -> list[str]:
    parts = [p for p in re.split(r"\s*\+\s*|\s*,\s*|\s+and\s+|\s+", spec.strip().lower()) if p]
    return [_KEY_NAMES.get(p, p) for p in parts]


def _text_arg(s: str) -> str:
    """Prefer the quoted part ('type into the box "abc"' -> abc), else the whole remainder."""
    m = _QUOTED.search(s)
    return m.group(1) if m else s.strip().rstrip(".")


def _click_kind(verb: str) -> str:
    v = verb.lower()
    return "double_click" if v.startswith("double") else "right_click" if v.startswith("right") else "click"


_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], Action]]] = [
    (re.compile(r"^(?:open|launch|start|run)\s+(?:the\s+)?(?P<app>[\w .+-]{2,40}?)(?:\s+app(?:lication)?)?\.?$", re.I),
     lambda m: Action(action="launch", text=m["app"].strip(), thought="open the app")),
    (re.compile(r"^(?:type|write|input)\s+(?:the\s+text\s+)?(?P<text>.+)$", re.I | re.S),
     lambda m: Action(action="type", text=_text_arg(m["text"]), thought="type the text")),
    (re.compile(r"^(?:press|hit)\s+(?:the\s+)?(?P<keys>[\w+ ,]{1,30}?)(?:\s+keys?|\s+shortcut)?\.?$", re.I),
     lambda m: Action(action="hotkey", keys=_keys(m["keys"]), thought="keyboard shortcut")),
    (re.compile(r"^wait(?:\s+(?:for\s+)?(?P<n>\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?))?", re.I),
     lambda m: Action(action="wait", seconds=float(m["n"] or 1.5), thought="let the UI settle")),
    (re.compile(r"^scroll\s+(?P<dir>up|down)", re.I),
     lambda m: Action(action="scroll", direction=m["dir"].lower(), thought="scroll")),
    (re.compile(r"^(?P<verb>double[- ]?click|right[- ]?click|click|tap|select|choose)\s+(?:on\s+)?(?P<t>.+?)\.?$",
                re.I),
     lambda m: Action(action=_click_kind(m["verb"]), target=_text_arg(m["t"]), thought="click the named target")),
]


EDITORS = {"notepad", "notepad++", "wordpad", "word", "microsoft word", "writer", "libreoffice writer"}


def make_safe_plan(plan: list[str]) -> list[str]:
    """Editors such as Windows 11 Notepad reopen the user's previous files. If the plan opens one and
    then types, start a fresh document first so the text never lands in an existing file."""
    out: list[str] = []
    for i, step in enumerate(plan):
        out.append(step)
        a = direct_action(step)
        if a is None or a.action != "launch" or a.text.lower() not in EDITORS:
            continue
        nxt = direct_action(plan[i + 1]) if i + 1 < len(plan) else None
        if nxt is not None and nxt.action == "type":
            out += ["press ctrl+n", "wait 1 seconds"]
    return out


_FILE_IN_TITLE = re.compile(r"([\w ()\[\]{}.,'+-]+\.[A-Za-z0-9]{1,5})\s+[-–—]\s")


def existing_file_in_title(title: str, goal: str) -> str | None:
    """Window titles like 'recovery-codes.txt - Notepad' mean an existing file is open. Return its name
    unless the goal mentions it (then the user asked to edit that file)."""
    m = _FILE_IN_TITLE.search(title)
    if not m:
        return None
    name = m.group(1).strip().lstrip("*● ").strip()
    return None if name.lower() in goal.lower() else name


def direct_action(subtask: str) -> Action | None:
    """Return an Action for sub-tasks that need no model, else None."""
    s = subtask.strip()
    for pat, build in _PATTERNS:
        m = pat.match(s)
        if m:
            a = build(m)
            if (a.action == "hotkey" and not a.keys) or (a.action == "type" and not a.text):
                return None
            return a
    return None
