from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


@dataclass
class Settings:
    jobs_dir: Path = field(default_factory=lambda: Path(_env("JOBS_DIR", "jobs")))

    # LLM (planning + creative QA)
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-opus-5-5"))

    # Video provider
    video_model: str = field(default_factory=lambda: _env("VIDEO_MODEL", "gemini-omni-1.2-flash"))
    video_resolution: str = field(default_factory=lambda: _env("VIDEO_RESOLUTION", "1080p"))
    render_mode: str = "auto"  # auto | single | per_scene

    # Reliability
    max_attempts: int = field(default_factory=lambda: int(_env("MAX_ATTEMPTS", "3")))
    backoff_base_s: float = 4.0
    poll_interval_s: float = 5.0
    poll_timeout_s: float = field(default_factory=lambda: float(_env("POLL_TIMEOUT_S", "900")))

    # QA gate. The thresholds are applied in code, never by the model.
    qa_regenerations: int = field(default_factory=lambda: int(_env("QA_REGENERATIONS", "1")))
    qa_min_dimension: int = 6
    qa_min_mean: float = 7.0

    # Output
    width: int = 1080
    height: int = 1920
    fps: int = 30
    hook_super_seconds: float = 2.6
    end_card_seconds: float = 2.8
    font_path: str | None = field(default_factory=lambda: os.environ.get("FONT_PATH"))
