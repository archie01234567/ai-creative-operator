"""Mock provider: ffmpeg test clips with scripted failures.

Used by the tests and by `--provider mock`, so the whole lifecycle (polling, retries,
assembly, technical QA) can be exercised without spending money.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from .base import (
    Capabilities,
    ErrorKind,
    GenerationRequest,
    ProviderError,
    ProviderState,
    ProviderStatus,
    VideoProvider,
)


class MockProvider(VideoProvider):
    name = "mock"

    def __init__(
        self,
        failures: list[ErrorKind | None] | None = None,
        max_clip_seconds: int = 15,
        polls_until_done: int = 2,
    ):
        """`failures[i]` is the outcome of the i-th generate() call (None = success)."""
        self.capabilities = Capabilities(
            min_clip_seconds=1, max_clip_seconds=max_clip_seconds, supports_reference_images=True
        )
        self._failures = list(failures or [])
        self._polls_until_done = polls_until_done
        self._jobs: dict[str, dict] = {}
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> str:
        outcome = self._failures.pop(0) if self._failures else None
        self.requests.append(request)
        job_id = f"mock-{uuid.uuid4().hex[:8]}"
        self._jobs[job_id] = {"request": request, "polls": 0, "outcome": outcome}
        return job_id

    def get_status(self, provider_job_id: str) -> ProviderStatus:
        job = self._jobs.get(provider_job_id)
        if job is None:
            return ProviderStatus(
                ProviderState.FAILED,
                ProviderError(ErrorKind.LOST, "mock provider has no record of this job"),
            )
        job["polls"] += 1
        if job["polls"] < self._polls_until_done:
            return ProviderStatus(ProviderState.RUNNING)
        if job["outcome"] is not None:
            return ProviderStatus(
                ProviderState.FAILED,
                ProviderError(job["outcome"], f"scripted {job['outcome']} failure"),
            )
        if not job["request"].output_path.exists():
            _render_test_clip(job["request"], len(self.requests))
        return ProviderStatus(ProviderState.SUCCEEDED)

    def get_result(self, provider_job_id: str) -> Path:
        return self._jobs[provider_job_id]["request"].output_path


def _render_test_clip(request: GenerationRequest, seed: int) -> None:
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = (720, 1280) if request.aspect_ratio == "9:16" else (1280, 720)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=24:duration={request.duration}",
        "-f", "lavfi", "-i", f"sine=frequency={220 + 110 * (seed % 4)}:duration={request.duration}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(request.output_path),
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True)
