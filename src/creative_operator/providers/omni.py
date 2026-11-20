"""Gemini Omni Flash adapter (Gemini API, Interactions endpoint).

The Interactions call for video is request/response: it returns when the clip exists. To
honour the asynchronous `VideoProvider` contract each request runs on a worker thread and
`get_status()` reads the future. The consequence is stated plainly rather than hidden: an
in-flight generation does not survive a process restart, so after a restart `get_status()`
answers LOST and the pipeline resubmits. A provider with server-side job ids (Kling, Runway)
would return the remote id from `generate()` and resume polling instead.
"""

from __future__ import annotations

import base64
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
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


class OmniProvider(VideoProvider):
    name = "omni"
    capabilities = Capabilities(
        min_clip_seconds=3, max_clip_seconds=15, supports_reference_images=True
    )

    def __init__(self, model: str = "gemini-omni-1.2-flash", max_parallel: int = 2):
        from google import genai
        from google.genai import errors

        self._errors = errors
        self.model = model
        try:
            self.client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        except Exception as e:  # the SDK raises a bare ValueError when no key is set
            raise ProviderError(
                ErrorKind.FATAL,
                f"Could not create the Gemini client: {e}",
                hint="Set GEMINI_API_KEY in .env (or use --provider mock / manual).",
            ) from e
        self._pool = ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="omni")
        self._futures: dict[str, Future[Path]] = {}

    # ------------------------------------------------------------------ contract

    def generate(self, request: GenerationRequest) -> str:
        job_id = f"omni-{uuid.uuid4().hex[:10]}"
        self._futures[job_id] = self._pool.submit(self._run, request)
        return job_id

    def get_status(self, provider_job_id: str) -> ProviderStatus:
        future = self._futures.get(provider_job_id)
        if future is None:
            return ProviderStatus(
                ProviderState.FAILED,
                ProviderError(
                    ErrorKind.LOST,
                    "Generation was started by a previous process and cannot be re-attached.",
                ),
            )
        if not future.done():
            return ProviderStatus(ProviderState.RUNNING)
        error = future.exception()
        if error is None:
            return ProviderStatus(ProviderState.SUCCEEDED)
        if not isinstance(error, ProviderError):
            error = ProviderError(ErrorKind.TRANSIENT, f"{type(error).__name__}: {error}")
        return ProviderStatus(ProviderState.FAILED, error)

    def get_result(self, provider_job_id: str) -> Path:
        return self._futures[provider_job_id].result()

    # ------------------------------------------------------------------ internals

    def _run(self, request: GenerationRequest) -> Path:
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
            )
        except self._errors.APIError as e:
            raise self._classify(e.code, _api_code(e), e.message or str(e)) from e

        if interaction.status != "completed" or interaction.output_video is None:
            first = (interaction.errors or [None])[0]
            code = getattr(first, "code", None) or str(interaction.status)
            message = getattr(first, "message", None) or "no video in the response"
            raise self._classify(None, code, f"interaction {interaction.status}: {message}")

        video = interaction.output_video
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        if video.uri:
            self._download(video.uri, request.output_path)
        elif video.data:
            request.output_path.write_bytes(base64.b64decode(video.data))
        else:
            raise ProviderError(ErrorKind.TRANSIENT, "response had neither a uri nor video data")
        return request.output_path

    def _download(self, uri: str, destination: Path) -> None:
        name = "files/" + uri.rstrip("/").split("/")[-1]
        deadline = time.monotonic() + FILE_READY_TIMEOUT_S
        while True:
            try:
                state = self.client.files.get(name=name).state.name
            except self._errors.APIError as e:
                raise self._classify(e.code, _api_code(e), e.message or str(e)) from e
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise ProviderError(ErrorKind.TRANSIENT, "provider failed while processing the file")
            if time.monotonic() > deadline:
                raise ProviderError(ErrorKind.TRANSIENT, "generated file never became ACTIVE")
            time.sleep(5)
        try:
            data = self.client.files.download(file=uri, destination=str(destination))
        except self._errors.APIError as e:
            raise self._classify(e.code, _api_code(e), e.message or str(e)) from e
        if data and not destination.exists():
            destination.write_bytes(data)

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
        if http == 402 or code in ("payment_required", "failed_precondition", "quota_exceeded"):
            return ProviderError(
                ErrorKind.FATAL, text, hint="Enable billing / raise quota for the Gemini API key."
            )
        if http in (408, 409, 429) or (http or 0) >= 500 or code in (
            "rate_limit_exceeded", "too_many_requests", "aborted", "api_error",
            "service_unavailable", "deadline_exceeded", "failed",
        ):  # fmt: skip
            return ProviderError(ErrorKind.TRANSIENT, text)
        return ProviderError(ErrorKind.FATAL, text, hint="The request itself was rejected.")


def _api_code(error) -> str | None:
    """The Interactions API puts a snake_case string code at error.code in the JSON body."""
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        code = details.get("error", {}).get("code") if "error" in details else details.get("code")
        if isinstance(code, str):
            return code
    return error.status if isinstance(getattr(error, "status", None), str) else None
