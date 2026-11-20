"""The one seam between the pipeline and a language model.

Every stage asks for a validated Pydantic object, never free text, so there is no JSON
scraping anywhere else in the codebase.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Protocol, Sequence, TypeVar

from pydantic import BaseModel

from .errors import LLMError

T = TypeVar("T", bound=BaseModel)
LabelledImage = tuple[str, Path]  # (caption shown to the model, jpeg path)


class LLM(Protocol):
    supports_vision: bool

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        images: Sequence[LabelledImage] = (),
        effort: str = "high",
    ) -> T: ...


class ClaudeLLM:
    supports_vision = True

    def __init__(self, model: str):
        import anthropic

        self._anthropic = anthropic
        self.model = model
        # The SDK already retries 408/409/429/5xx and connection errors with backoff.
        self.client = anthropic.Anthropic(max_retries=4, timeout=300.0)

    def structured(self, *, system, user, schema, images=(), effort="high"):
        content: list[dict] = []
        for label, path in images:
            data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
                }
            )
        content.append({"type": "text", "text": user})

        a = self._anthropic
        try:
            response = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_format=schema,
                output_config={"effort": effort},
                # If a safety classifier declines, re-run on a fallback model in the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except TypeError as e:
            # The SDK resolves credentials lazily and raises a bare TypeError when none exist.
            if "authentication" not in str(e).lower():
                raise
            raise LLMError(
                "No Anthropic credentials found.",
                hint="Set ANTHROPIC_API_KEY in .env (or run with --offline).",
            ) from e
        except a.AuthenticationError as e:
            raise LLMError(
                f"Anthropic rejected the credentials: {e.message}",
                hint="Set ANTHROPIC_API_KEY in .env (or run with --offline).",
            ) from e
        except a.RateLimitError as e:
            raise LLMError(f"Anthropic rate limit persisted after retries: {e.message}") from e
        except a.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMError(f"Could not reach the Anthropic API: {e}") from e

        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request.", hint="Review the brief wording.")
        if response.stop_reason == "max_tokens":
            raise LLMError("The model ran out of output tokens before finishing the JSON.")
        if response.parsed_output is None:
            raise LLMError(f"No parseable {schema.__name__} in the model response.")
        return response.parsed_output


class FixtureLLM:
    """Offline stand-in: returns canned planner output so the pipeline runs with no API key.

    It has no eyes, so creative QA reports itself as skipped instead of inventing scores.
    """

    supports_vision = False

    def __init__(self, fixtures_dir: Path):
        self.dir = Path(fixtures_dir)

    def structured(self, *, system, user, schema, images=(), effort="high"):
        path = self.dir / f"{schema.__name__.lower()}.json"
        if not path.exists():
            raise LLMError(f"No offline fixture for {schema.__name__} at {path}")
        return schema.model_validate(json.loads(path.read_text(encoding="utf-8")))
