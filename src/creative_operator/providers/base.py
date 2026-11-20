"""The only video-generation surface the pipeline knows about.

Adding a provider means implementing three methods and declaring capabilities. Nothing in
the pipeline, prompt builder or QA imports a concrete provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class ErrorKind(StrEnum):
    TRANSIENT = "transient"  # rate limit, 5xx, timeout: retry the same request
    LOST = "lost"  # provider no longer knows the job (e.g. process restart): resubmit
    BLOCKED = "blocked"  # safety filter: retrying the same prompt will not help
    FATAL = "fatal"  # auth, billing, invalid request: needs a human


class ProviderError(Exception):
    def __init__(self, kind: ErrorKind, message: str, hint: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.hint = hint

    @property
    def retryable(self) -> bool:
        return self.kind in (ErrorKind.TRANSIENT, ErrorKind.LOST)


class ProviderState(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class ProviderStatus:
    state: ProviderState
    error: ProviderError | None = None


@dataclass(frozen=True)
class Capabilities:
    min_clip_seconds: int
    max_clip_seconds: int
    supports_reference_images: bool = False


@dataclass
class GenerationRequest:
    prompt: str
    duration: int
    output_path: Path  # where get_result() must leave the mp4
    aspect_ratio: str = "9:16"
    resolution: str = "1080p"
    reference_images: list[Path] = field(default_factory=list)


class VideoProvider(ABC):
    name: str
    capabilities: Capabilities

    @abstractmethod
    def generate(self, request: GenerationRequest) -> str:
        """Start a generation and return a provider job id immediately."""

    @abstractmethod
    def get_status(self, provider_job_id: str) -> ProviderStatus:
        """Non-blocking status check. Never raises for a failed generation."""

    @abstractmethod
    def get_result(self, provider_job_id: str) -> Path:
        """Path of the finished mp4. Only valid after status is SUCCEEDED."""
