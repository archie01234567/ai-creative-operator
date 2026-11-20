"""Replicate adapter: any official text-to-video model on replicate.com.

Replicate is natively asynchronous (create a prediction, poll it by id), so it maps onto
`VideoProvider` one to one. Input names differ per model, so the request is built from the
model's own published schema instead of being hard-coded: only fields the model declares
are sent, and duration / resolution / aspect ratio are snapped to values it accepts. That
is also where `capabilities` comes from, so switching REPLICATE_MODEL needs no code change.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
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

API = "https://api.replicate.com/v1"
DEFAULT_MODEL = "kwaivgi/kling-v4-flash"
REQUEST_TIMEOUT_S = 60
DOWNLOAD_TIMEOUT_S = 300
ID_SEPARATOR = "|"
RUNNING_STATUSES = {"starting", "processing"}
BLOCK_MARKERS = ("nsfw", "safety", "sensitive", "moderation", "content policy", "flagged")
REFERENCE_FIELDS = ("reference_images",)


class ReplicateProvider(VideoProvider):
    name = "replicate"

    def __init__(self, model: str | None = None, token: str | None = None):
        self.model = model or os.environ.get("REPLICATE_MODEL") or DEFAULT_MODEL
        self._token = token or os.environ.get("REPLICATE_API_TOKEN")
        if not self._token:
            raise ProviderError(
                ErrorKind.FATAL,
                "No Replicate API token found.",
                hint="Set REPLICATE_API_TOKEN in .env (replicate.com/account/api-tokens).",
            )
        self.schema = self._input_schema()
        low, high = _duration_range(self.schema.get("duration", {}))
        self.capabilities = Capabilities(
            min_clip_seconds=low,
            max_clip_seconds=high,
            supports_reference_images=any(f in self.schema for f in REFERENCE_FIELDS),
        )

    # ------------------------------------------------------------------ contract

    def generate(self, request: GenerationRequest) -> str:
        prediction = self._call(
            "POST", f"{API}/models/{self.model}/predictions", {"input": self.build_input(request)}
        )
        return f"{prediction['id']}{ID_SEPARATOR}{request.output_path}"

    def get_status(self, provider_job_id: str) -> ProviderStatus:
        prediction_id, _ = _split(provider_job_id)
        try:
            prediction = self._call("GET", f"{API}/predictions/{prediction_id}")
        except ProviderError as e:
            if e.kind == ErrorKind.TRANSIENT:
                return ProviderStatus(ProviderState.RUNNING)  # a failed poll is not a failed job
            return ProviderStatus(ProviderState.FAILED, e)

        status = prediction.get("status")
        if status in RUNNING_STATUSES:
            return ProviderStatus(ProviderState.RUNNING)
        if status == "succeeded" and _output_url(prediction):
            return ProviderStatus(ProviderState.SUCCEEDED)
        message = str(prediction.get("error") or f"prediction {status} with no output")
        if any(marker in message.lower() for marker in BLOCK_MARKERS):
            kind, hint = ErrorKind.BLOCKED, "The prompt or output hit a safety filter; reword it."
        elif status == "canceled":
            kind, hint = ErrorKind.FATAL, "The prediction was cancelled on Replicate."
        else:
            kind, hint = ErrorKind.TRANSIENT, None  # model-side failure: worth another try
        return ProviderStatus(ProviderState.FAILED, ProviderError(kind, message, hint))

    def get_result(self, provider_job_id: str) -> Path:
        prediction_id, output_path = _split(provider_job_id)
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
        url = _output_url(self._call("GET", f"{API}/predictions/{prediction_id}"))
        if not url:
            raise ProviderError(
                ErrorKind.LOST, "the output is gone (Replicate deletes files after one hour)"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
                partial.write_bytes(response.read())
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ProviderError(ErrorKind.TRANSIENT, f"download failed: {e}") from e
        partial.replace(output_path)
        return output_path

    # ------------------------------------------------------------------ internals

    def build_input(self, request: GenerationRequest) -> dict:
        """Translate a provider-neutral request into this model's own input names."""
        schema = self.schema
        payload: dict = {"prompt": request.prompt}
        if "duration" in schema:
            payload["duration"] = _fit_number(schema["duration"], request.duration)
        if "aspect_ratio" in schema:
            payload["aspect_ratio"] = _fit_choice(schema["aspect_ratio"], request.aspect_ratio)
        if "resolution" in schema:
            payload["resolution"] = _fit_choice(schema["resolution"], request.resolution)
        field = next((f for f in REFERENCE_FIELDS if f in schema), None)
        if field and request.reference_images:
            payload[field] = [_data_uri(path) for path in request.reference_images]
        return {key: value for key, value in payload.items() if value is not None}

    def _input_schema(self) -> dict:
        model = self._call("GET", f"{API}/models/{self.model}")
        try:
            schemas = model["latest_version"]["openapi_schema"]["components"]["schemas"]
        except (KeyError, TypeError) as e:
            raise ProviderError(
                ErrorKind.FATAL, f"Replicate returned no input schema for {self.model}."
            ) from e
        properties = schemas["Input"].get("properties", {})
        # Enums are published as separate schemas referenced through allOf / $ref.
        return {name: _resolve(spec, schemas) for name, spec in properties.items()}

    def _call(self, method: str, url: str, body: dict | None = None) -> dict:
        request = urllib.request.Request(
            url,
            method=method,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                # Replicate's edge rejects urllib's default agent with 403 / error 1010.
                "User-Agent": "creative-operator/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            raise _classify(e.code, _detail(e), self.model) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ProviderError(ErrorKind.TRANSIENT, f"could not reach Replicate: {e}") from e


def _classify(http: int, detail: str, model: str) -> ProviderError:
    text = f"[{http}] {detail}"
    if http in (401, 403):
        return ProviderError(ErrorKind.FATAL, text, hint="Check REPLICATE_API_TOKEN.")
    if http == 402:
        return ProviderError(
            ErrorKind.FATAL, text, hint="Replicate credit is used up; add credit, then `resume`."
        )
    if http == 404:
        return ProviderError(ErrorKind.FATAL, text, hint=f"Check REPLICATE_MODEL ('{model}').")
    if http in (408, 409, 429) or http >= 500:
        return ProviderError(ErrorKind.TRANSIENT, text)
    return ProviderError(ErrorKind.FATAL, text, hint="The model rejected the input.")


def _detail(error: urllib.error.HTTPError) -> str:
    raw = error.read().decode("utf-8", "replace")
    try:
        body = json.loads(raw)
    except ValueError:
        return raw[:300] or error.reason
    return str(body.get("detail") or body.get("title") or body)[:500]


def _resolve(spec: dict, schemas: dict) -> dict:
    merged = dict(spec)
    for ref in [spec, *spec.get("allOf", []), *spec.get("anyOf", [])]:
        target = ref.get("$ref", "").rsplit("/", 1)[-1]
        if target in schemas:
            merged = {**schemas[target], **merged}
    return merged


def _duration_range(spec: dict) -> tuple[int, int]:
    if spec.get("enum"):
        values = sorted(int(v) for v in spec["enum"])
        return values[0], values[-1]
    if "maximum" in spec:
        return int(spec.get("minimum", 1)), int(spec["maximum"])
    return 5, 5  # no declared duration control: assume a short fixed-length clip


def _fit_number(spec: dict, wanted: int) -> int:
    """The shortest duration the model accepts that still covers what was asked for."""
    if spec.get("enum"):
        values = sorted(int(v) for v in spec["enum"])
        return next((v for v in values if v >= wanted), values[-1])
    return max(int(spec.get("minimum", wanted)), min(int(spec.get("maximum", wanted)), wanted))


def _fit_choice(spec: dict, wanted: str) -> str | None:
    options = spec.get("enum")
    if not options or wanted in options:
        return wanted
    return spec.get("default")  # e.g. a 720p-only model asked for 1080p; assembly rescales


def _output_url(prediction: dict) -> str | None:
    output = prediction.get("output")
    if isinstance(output, list):
        output = output[0] if output else None
    return output if isinstance(output, str) and output.startswith("http") else None


def _data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _split(provider_job_id: str) -> tuple[str, Path]:
    prediction_id, _, output = provider_job_id.partition(ID_SEPARATOR)
    return prediction_id, Path(output)
