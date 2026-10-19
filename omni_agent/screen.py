"""Screen capture and coordinate mapping.

Three coordinate spaces exist and mixing them up is the most common cause of missed clicks:

* screen  - physical pixels on the virtual desktop (what the mouse uses once the process is DPI aware)
* frame   - pixels inside the captured monitor image (screen minus the monitor's top-left offset)
* model   - pixels inside the resized copy the vision model sees
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image

_dpi_set = False


def make_dpi_aware() -> None:
    """Use physical pixels everywhere. Without this, Windows display scaling (125%, 150%)
    makes screenshots and mouse coordinates disagree."""
    global _dpi_set
    if _dpi_set or sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    _dpi_set = True


@dataclass
class Frame:
    image: Image.Image  # full-resolution RGB capture of one monitor
    left: int  # monitor offset on the virtual desktop
    top: int

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        return int(round(self.left + x)), int(round(self.top + y))

    def model_view(self, width: int, height: int) -> Image.Image:
        return self.image.resize((width, height), Image.LANCZOS)

    def model_to_frame(self, x: float, y: float, width: int, height: int) -> tuple[float, float]:
        fw, fh = self.size
        return x * fw / width, y * fh / height

    def frame_to_model(self, x: float, y: float, width: int, height: int) -> tuple[float, float]:
        fw, fh = self.size
        return x * width / fw, y * height / fh

    def gray_small(self, size: int = 256) -> np.ndarray:
        return np.asarray(self.image.convert("L").resize((size, size * self.size[1] // self.size[0])), np.float32)


def list_monitors() -> list[dict]:
    import mss

    with mss.mss() as sct:
        return [dict(m) for m in sct.monitors]


def capture(monitor: int = 1) -> Frame:
    import mss

    make_dpi_aware()
    with mss.mss() as sct:
        mon = sct.monitors[monitor]
        shot = sct.grab(mon)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        return Frame(image=img, left=mon["left"], top=mon["top"])


def change_ratio(before: Frame, after: Frame, threshold: float = 12.0) -> float:
    """Fraction of (downscaled) pixels that changed noticeably between two frames."""
    a, b = before.gray_small(), after.gray_small()
    if a.shape != b.shape:
        return 1.0
    return float((np.abs(a - b) > threshold).mean())
