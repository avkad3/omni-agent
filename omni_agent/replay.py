"""Trajectory replay: re-run a goal that already succeeded without calling the model.

After a verified run, every executed action is stored with an anchor for re-finding its target:
the clicked text (matched again by OCR) or a small image patch around the click point (matched by
normalised cross-correlation near its old position). Replay walks the stored actions; the moment an
anchor cannot be found, it hands control back to the normal agent at that plan step, so earlier steps
are never repeated.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

PATCH = 48  # half-size of the stored patch around a click, in screen pixels


def goal_key(goal: str) -> str:
    norm = re.sub(r"\s+", " ", goal.strip().lower())
    slug = re.sub(r"[^a-z0-9]+", "-", norm)[:48].strip("-")
    return f"{slug}-{hashlib.sha1(norm.encode()).hexdigest()[:8]}"


@dataclass
class TrajStep:
    plan_idx: int
    action: dict
    anchor_text: str = ""  # OCR text that was clicked, if any
    patch: str = ""  # PNG file name of the image patch around the click
    rel_xy: tuple[float, float] | None = None  # click position as a fraction of the screen


@dataclass
class Trajectory:
    goal: str
    plan: list[str]
    steps: list[TrajStep] = field(default_factory=list)


def crop_patch(image: Image.Image, x: float, y: float, half: int = PATCH) -> Image.Image:
    W, H = image.size
    x1, y1 = int(max(0, min(W - 2 * half, x - half))), int(max(0, min(H - 2 * half, y - half)))
    return image.crop((x1, y1, x1 + 2 * half, y1 + 2 * half))


def _gray(img: Image.Image, scale: float) -> np.ndarray:
    w, h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    return np.asarray(img.convert("L").resize((w, h), Image.BILINEAR), np.float32)


def find_patch(image: Image.Image, patch: Image.Image, near: tuple[float, float], search: int = 260,
               scale: float = 0.5) -> tuple[float, float, float] | None:
    """Best match of `patch` within +-search px of `near` (screen pixels). Returns (x, y, score), the
    centre of the match, with score = normalised cross-correlation in [-1, 1]."""
    W, H = image.size
    pw, ph = patch.size
    x1, y1 = int(max(0, near[0] - search - pw / 2)), int(max(0, near[1] - search - ph / 2))
    x2, y2 = int(min(W, near[0] + search + pw / 2)), int(min(H, near[1] + search + ph / 2))
    if x2 - x1 < pw or y2 - y1 < ph:
        return None
    region = _gray(image.crop((x1, y1, x2, y2)), scale)
    tmpl = _gray(patch, scale)
    th, tw = tmpl.shape
    if region.shape[0] < th or region.shape[1] < tw:
        return None
    t = tmpl - tmpl.mean()
    tn = np.sqrt((t * t).sum())
    if tn < 1e-3:  # flat patch (empty background): not a usable anchor
        return None
    win = sliding_window_view(region, (th, tw))
    wm = win.mean(axis=(2, 3), keepdims=True)
    wc = win - wm
    num = np.einsum("ijkl,kl->ij", wc, t)
    den = np.sqrt((wc * wc).sum(axis=(2, 3))) * tn + 1e-6
    ncc = num / den
    iy, ix = np.unravel_index(int(ncc.argmax()), ncc.shape)
    cx = x1 + (ix + tw / 2) / scale
    cy = y1 + (iy + th / 2) / scale
    return float(cx), float(cy), float(ncc[iy, ix])


class TrajectoryStore:
    def __init__(self, root: str | Path = "trajectories"):
        self.root = Path(root)

    def path(self, goal: str) -> Path:
        return self.root / goal_key(goal)

    def load(self, goal: str) -> Trajectory | None:
        f = self.path(goal) / "trajectory.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text(encoding="utf-8"))
        return Trajectory(d["goal"], d["plan"], [TrajStep(**s) for s in d["steps"]])

    def patch(self, goal: str, step: TrajStep) -> Image.Image | None:
        p = self.path(goal) / step.patch if step.patch else None
        return Image.open(p).convert("RGB") if p and p.exists() else None

    def save(self, goal: str, plan: list[str], steps: list[tuple[TrajStep, Image.Image | None]]) -> Path:
        d = self.path(goal)
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("patch_*.png"):
            old.unlink()
        out = []
        for i, (st, img) in enumerate(steps):
            if img is not None:
                st.patch = f"patch_{i:02d}.png"
                img.save(d / st.patch)
            out.append(asdict(st))
        (d / "trajectory.json").write_text(json.dumps({"goal": goal, "plan": plan, "steps": out}, indent=2),
                                           encoding="utf-8")
        return d
