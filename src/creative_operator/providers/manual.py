"""Manual provider: clips made by hand in a platform UI (Kling, Higgsfield, Runway, ...).

`generate()` hands back the clip the operator supplied for that slot. With nothing supplied
it fails with the exact prompts to paste, so the job can be resumed once the clips exist:

    creative-operator run --provider manual           # plans, writes prompts, stops
    creative-operator resume <job> --clips a.mp4      # assembles + QA on the manual clips
"""

from __future__ import annotations

import shutil
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


class ManualProvider(VideoProvider):
    name = "manual"

    def __init__(self, clips: list[Path] | None = None, max_clip_seconds: int = 15):
        self.capabilities = Capabilities(min_clip_seconds=1, max_clip_seconds=max_clip_seconds)
        self._clips = [Path(c) for c in (clips or [])]
        self._served = 0

    def generate(self, request: GenerationRequest) -> str:
        prompt_file = request.output_path.with_suffix(".prompt.txt")
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(request.prompt, encoding="utf-8")

        if self._served >= len(self._clips):
            raise ProviderError(
                ErrorKind.FATAL,
                f"No manual clip supplied for {request.output_path.name}.",
                hint=f"Generate the prompt in {prompt_file} ({request.duration}s, "
                f"{request.aspect_ratio}) and resume with --clips <file.mp4>.",
            )
        source = self._clips[self._served]
        self._served += 1
        if not source.exists():
            raise ProviderError(ErrorKind.FATAL, f"Manual clip not found: {source}")
        shutil.copyfile(source, request.output_path)
        return str(request.output_path)

    def get_status(self, provider_job_id: str) -> ProviderStatus:
        if Path(provider_job_id).exists():
            return ProviderStatus(ProviderState.SUCCEEDED)
        return ProviderStatus(
            ProviderState.FAILED, ProviderError(ErrorKind.LOST, "manual clip is missing")
        )

    def get_result(self, provider_job_id: str) -> Path:
        return Path(provider_job_id)
