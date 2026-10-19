"""Ways to stop the agent and keep it away from things it should not touch."""

from __future__ import annotations

import ctypes
import sys
import threading


class KillSwitch:
    """Global hotkey (default Ctrl+Alt+Q) that stops the run before the next action.
    PyAutoGUI's corner fail-safe (mouse into any screen corner) also stays active."""

    def __init__(self, combo: str = "<ctrl>+<alt>+q"):
        self.combo = combo
        self.event = threading.Event()
        self._listener = None

    def start(self) -> None:
        try:
            from pynput import keyboard

            self._listener = keyboard.GlobalHotKeys({self.combo: self.event.set})
            self._listener.daemon = True
            self._listener.start()
        except Exception:  # no keyboard hook available (e.g. CI); the corner fail-safe still works
            self._listener = None

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()

    @property
    def triggered(self) -> bool:
        return self.event.is_set()


def foreground_title() -> str:
    """Title of the active window. Used only by the allow-list guard, never for planning."""
    if sys.platform != "win32":
        return ""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def allowed(title: str, allow: list[str]) -> bool:
    if not allow:
        return True
    t = title.lower()
    return any(a.lower() in t for a in allow)


def confirm(question: str) -> bool:
    try:
        return input(f"\n[safety] {question} Proceed? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False
