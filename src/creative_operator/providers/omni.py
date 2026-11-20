"""Gemini Omni Flash adapter (Gemini API, Interactions endpoint, v1beta).

Generations are created with `background=True`, so `generate()` returns as soon as Google
has accepted the job and `get_status()` is a cheap GET on the interaction. The provider job
id carries the interaction id and the output path, which makes it self-contained: a job
whose process died mid-generation is picked up again on `resume` without paying twice.
"""

from __future__ import annotations

import base64
import time
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

# Interactions API error codes that mean "this prompt was refused", not "try again".
BLOCK_CODES = {
    "safety", "recitation", "language", "prohibited_content", "spii", "blocklist",
    "image_safety", "image_prohibited_content", "image_recitation", "image_other",
    "content_blocked",
}  # fmt: skip
FILE_READY_TIMEOUT_S = 300
API_VERSION = "v1beta"
REQUEST_TIMEOUT_S = 60
RUNNING_STATUSES = {"in_progress", "queued", "requires_action"}
ID_SEPARATOR = "|"


class OmniProvider(VideoProvider):
    name = "omni"
    capabilities = Capabilities(
        min_clip_seconds=3, max_clip_seconds=15, supports_reference_images=True
    )

    def __init__(self, model: str = "gemini-omni-1.2-flash"):
        from google import genai

        self.model = model
        try:
            # Reads GEMINI_API_KEY / GOOGLE_API_KEY. The SDK's Interactions client defaults
            # to v1, where the Omni models 404; they are served on v1beta.
            self.client = genai.Client(http_options={"api_version": API_VERSION})
        except Exception as e:  # the SDK raises a bare ValueError when no key is set
            raise ProviderError(
                ErrorKind.FATAL,
                f"Could not create the Gemini client: {e}",
                hint="Set GEMINI_API_KEY in .env (or use --provider mock / manual).",
            ) from e

    # ------------------------------------------------------------------ contract

    def generate(self, request: GenerationRequest) -> str:
        parts: list[dict] = [
            {
                "type": "image",
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                "mime_type": "image/jpeg",
            }
            for path in request.reference_images
        ]
        parts.append({"type": "text", "text": request.prompt})
        try:
            interaction = self.client.interactions.create(
                model=self.model,
                input=parts if request.reference_images else request.prompt,
                response_format={
                    "type": "video",
                    "aspect_ratio": request.aspect_ratio,
                    "duration": f"{request.duration}s",
                    "resolution": request.resolution,
                    "delivery": "uri",  # inline base64 is only reliable under ~4 MB
                },
                background=True,
                timeout=REQUEST_TIMEOUT_S,
            )
        except Exception as e:
            raise self._from_exception(e) from e
        return f"{interaction.id}{ID_SEPARATOR}{request.output_path}"

    def get_status(self, provider_job_id: str) -> ProviderStatus:
        interaction_id, _ = _split(provider_job_id)
        try:
            interaction = self.client.interactions.get(interaction_id, timeout=REQUEST_TIMEOUT_S)
        except Exception as e:
            error = self._from_exception(e)
            if error.kind == ErrorKind.TRANSIENT:
                return ProviderStatus(ProviderState.RUNNING)  # a failed poll is not a failed job
            if "not_found" in error.message or "[404]" in error.message:
                error = ProviderError(ErrorKind.LOST, "the provider no longer knows this job")
            return ProviderStatus(ProviderState.FAILED, error)

        if interaction.status in RUNNING_STATUSES:
            return ProviderStatus(ProviderState.RUNNING)
        if interaction.status == "completed" and interaction.output_video is not None:
            return ProviderStatus(ProviderState.SUCCEEDED)
        first = (interaction.errors or [None])[0]
        code = getattr(first, "code", None) or str(interaction.status)
        message = getattr(first, "message", None) or "no video in the response"
        return ProviderStatus(
            ProviderState.FAILED,
            self._classify(None, code, f"interaction {interaction.status}: {message}"),
        )

    def get_result(self, provider_job_id: str) -> Path:
        interaction_id, output_path = _split(provider_job_id)
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
        try:
            video = self.client.interactions.get(
                interaction_id, timeout=REQUEST_TIMEOUT_S
            ).output_video
        except Exception as e:
            raise self._from_exception(e) from e
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if video is not None and video.uri:
            self._download(video.uri, output_path)
        elif video is not None and video.data:
            output_path.write_bytes(base64.b64decode(video.data))
        else:
            raise ProviderError(ErrorKind.TRANSIENT, "response had neither a uri nor video data")
        return output_path

    # ------------------------------------------------------------------ internals

    def _download(self, uri: str, destination: Path) -> None:
        name = "files/" + uri.rstrip("/").split("/")[-1]
        deadline = time.monotonic() + FILE_READY_TIMEOUT_S
        while True:
            try:
                state = self.client.files.get(name=name).state.name
            except Exception as e:
                raise self._from_exception(e) from e
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise ProviderError(ErrorKind.TRANSIENT, "provider failed while processing the file")
            if time.monotonic() > deadline:
                raise ProviderError(ErrorKind.TRANSIENT, "generated file never became ACTIVE")
            time.sleep(5)
        try:
            data = self.client.files.download(file=uri, destination=str(destination))
        except Exception as e:
            raise self._from_exception(e) from e
        if data and not destination.exists():
            destination.write_bytes(data)

    def _from_exception(self, error: Exception) -> ProviderError:
        """Map either SDK error family to a ProviderError.

        `files.*` raises google.genai.errors.APIError (int `.code`); `interactions.*` raises
        a separate hierarchy with `.status_code` and `.body`. Neither subclasses the other.
        """
        if isinstance(error, ProviderError):
            return error
        http = getattr(error, "status_code", None)
        if http is None and isinstance(getattr(error, "code", None), int):
            http = error.code
        message = getattr(error, "message", None) or str(error)
        if http is None:  # no HTTP response at all: connection reset, timeout, DNS
            return ProviderError(ErrorKind.TRANSIENT, f"{type(error).__name__}: {message}")
        return self._classify(http, _api_code(error), message)

    @staticmethod
    def _classify(http: int | None, code: str | None, message: str) -> ProviderError:
        code = (code or "").lower()
        text = f"[{code or http}] {message}"
        if code in BLOCK_CODES:
            return ProviderError(
                ErrorKind.BLOCKED, text, hint="The prompt or output hit a safety filter; reword it."
            )
        if http in (401, 403) or code in ("authentication", "permission_denied"):
            return ProviderError(ErrorKind.FATAL, text, hint="Check GEMINI_API_KEY and its access.")
        if http == 402 or code in (
            "payment_required", "failed_precondition", "quota_exceeded", "resource_exhausted",
        ):  # fmt: skip
            return ProviderError(
                ErrorKind.FATAL,
                text,
                hint="The Gemini key has no video quota: enable billing on its project in "
                "Google AI Studio (or raise the quota), then `resume`.",
            )
        if http in (408, 409, 429) or (http or 0) >= 500 or code in (
            "rate_limit_exceeded", "too_many_requests", "aborted", "api_error",
            "service_unavailable", "deadline_exceeded", "failed",
        ):  # fmt: skip
            return ProviderError(ErrorKind.TRANSIENT, text)
        if http == 404 or code == "not_found":
            return ProviderError(
                ErrorKind.FATAL, text, hint="Check VIDEO_MODEL and that the key has access to it."
            )
        return ProviderError(ErrorKind.FATAL, text, hint="The request itself was rejected.")


def _api_code(error) -> str | None:
    """The Interactions API puts a snake_case string code at error.code in the JSON body."""
    details = getattr(error, "body", None) or getattr(error, "details", None)
    if isinstance(details, dict):
        code = details.get("error", {}).get("code") if "error" in details else details.get("code")
        if isinstance(code, str):
            return code
    return error.status if isinstance(getattr(error, "status", None), str) else None


def _split(provider_job_id: str) -> tuple[str, Path]:
    interaction_id, _, output = provider_job_id.partition(ID_SEPARATOR)
    return interaction_id, Path(output)
