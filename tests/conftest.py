import json
import shutil
import subprocess
from pathlib import Path

import pytest

from creative_operator.config import Settings
from creative_operator.models import Brief

ROOT = Path(__file__).resolve().parents[1]

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not installed"
)


@pytest.fixture
def brief() -> Brief:
    return Brief.model_validate_json((ROOT / "briefs" / "nomadguard.json").read_text("utf-8"))


@pytest.fixture
def fixture_json():
    return lambda name: json.loads((ROOT / "fixtures" / f"{name}.json").read_text("utf-8"))


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.jobs_dir = tmp_path / "jobs"
    s.poll_interval_s = 0
    s.backoff_base_s = 0  # retries become immediately eligible
    return s


@pytest.fixture
def make_video(tmp_path):
    def _make(name: str, seconds: float, size: str = "1080x1920") -> Path:
        out = tmp_path / name
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             f"testsrc2=size={size}:rate=30:duration={seconds}", "-c:v", "libx264",
             "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)],
            check=True,
        )  # fmt: skip
        return out

    return _make
