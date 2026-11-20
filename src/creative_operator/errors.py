from __future__ import annotations


class StageError(Exception):
    """An expected, explainable failure. The pipeline turns it into `job.error`."""

    kind = "error"

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint


class LLMError(StageError):
    kind = "llm"


class PlanningError(StageError):
    kind = "planning"


class MediaError(StageError):
    kind = "media"


class GenerationError(StageError):
    kind = "generation"
