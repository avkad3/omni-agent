"""Top-level window helpers (Windows only). Used to execute actions and for safety checks, never to decide
what to do: the planner still sees only pixels.

Launching an app that is already running often leaves its window behind the current one, because
Windows blocks programs from stealing focus. `focus_app` brings it forward the way the Alt key would.
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes


def top_windows() -> dict[int, str]:
    """Visible top-level windows with a title: {hwnd: title}."""
    if sys.platform != "win32":
        return {}
    user32 = ctypes.windll.user32
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


def find_window(app: str, windows: dict[int, str] | None = None) -> int | None:
    """The window whose title names the app. EnumWindows lists in z-order, so the first hit is the
    most recently active one."""
    a = app.lower().strip()
    for hwnd, title in (windows if windows is not None else top_windows()).items():
        t = title.lower()
        if t == a or t.endswith(f"- {a}") or t.startswith(f"{a} ") or t == a + "s" or f" {a}" in f" {t}":
            return hwnd
    return None


def focus_app(app: str, wait: float = 3.0) -> bool:
    """Bring the app's window to the foreground. Returns True if it is in front afterwards."""
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    deadline = time.time() + wait
    while time.time() < deadline:
        hwnd = find_window(app)
        if hwnd:
            if user32.GetForegroundWindow() == hwnd:
                return True
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            # A synthetic Alt press satisfies Windows' "last input" rule for SetForegroundWindow.
            user32.keybd_event(0x12, 0, 0, 0)
            user32.keybd_event(0x12, 0, 2, 0)
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.3)
            if user32.GetForegroundWindow() == hwnd:
                return True
        time.sleep(0.3)
    return False
