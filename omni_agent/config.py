"""Runtime settings. Everything can be overridden with OMNI_* environment variables or CLI flags."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"OMNI_{name}", default)


@dataclass
class Settings:
    # Which planner backend to use: "ollama" (local, default) or "gemini" (free tier, needs GEMINI_API_KEY).
    backend: str = field(default_factory=lambda: _env("BACKEND", "ollama"))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5vl:3b"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))

    # Screen capture: 1 = primary monitor (mss numbering).
    monitor: int = field(default_factory=lambda: int(_env("MONITOR", "1")))
    # The model sees a resized copy. Multiples of 28 keep Qwen-VL from rescaling again,
    # so coordinates it returns map back exactly.
    model_width: int = 1288
    model_height: int = 728

    # When the planner gets the screenshot: "auto" (only when the element list is not enough),
    # "always", or "never". On a 4 GB GPU an image turn costs ~20 s and a text-only turn ~5-8 s.
    vision: str = field(default_factory=lambda: _env("VISION", "auto"))
    min_elements_for_text: int = 8  # fewer OCR elements than this (games, canvases) -> show the image

    icons: bool = True  # use the OmniParser icon detector when its model file is present
    replay: bool = True  # re-run goals that already succeeded from their stored trajectory
    trajectories_dir: Path = field(default_factory=lambda: Path(_env("TRAJECTORIES_DIR", "trajectories")))
    replay_threshold: float = 0.8  # minimum image-match score to trust a stored click anchor
    decompose: bool = True  # split the goal into steps first; simple steps then run without the model
    verify_end: bool = True  # one screenshot check that the goal is really complete before reporting success

    max_steps: int = field(default_factory=lambda: int(_env("MAX_STEPS", "15")))
    action_pause: float = 0.6  # seconds to let the UI settle after an action
    type_interval: float = 0.02

    # Safety
    dry_run: bool = False  # plan and ground, but never move the mouse or press keys
    confirm_risky: bool = True  # ask in the terminal before delete/send/pay-like actions
    allowed_apps: list[str] = field(default_factory=list)  # empty = no restriction

    runs_dir: Path = field(default_factory=lambda: Path(_env("RUNS_DIR", "runs")))
