from __future__ import annotations

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

__all__ = [
    "Capabilities",
    "ErrorKind",
    "GenerationRequest",
    "ProviderError",
    "ProviderState",
    "ProviderStatus",
    "VideoProvider",
    "build_provider",
]

PROVIDERS = ("omni", "mock", "manual")


def build_provider(name: str, *, model: str, clips: list[Path] | None = None) -> VideoProvider:
    """The only place that knows which concrete providers exist."""
    if name == "omni":
        from .omni import OmniProvider

        return OmniProvider(model=model)
    if name == "mock":
        from .mock import MockProvider

        return MockProvider()
    if name == "manual":
        from .manual import ManualProvider

        return ManualProvider(clips=clips)
    raise ValueError(f"Unknown provider '{name}'. Choose from {PROVIDERS}.")
