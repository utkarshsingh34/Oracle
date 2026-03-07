"""
LLM client abstraction for Oracle.

Supports Ollama and Claude backends via instructor for structured Pydantic output.
All LLM calls go through this module — no direct OpenAI/Anthropic usage elsewhere.

Usage:
    from Oracle.llm.client import LLMClient
    from Oracle.llm.schemas import FunctionSummary

    client = LLMClient()  # defaults from config
    result = client.complete("Summarize this function...", FunctionSummary)
"""

from __future__ import annotations

import time
from typing import TypeVar

import instructor
from pydantic import BaseModel

from Oracle.config import (
    LLM_BACKEND,
    OLLAMA_MODEL,
    CLAUDE_MODEL,
    LOG_LLM_PROMPTS,
)
from Oracle.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_MAX_RETRIES = 5
_BACKOFF_BASE = 0.5 # seconds; doubles each retry: 0.5, 1.0, 2.0


class LLMClient:
    """Unified LLM client with structured Pydantic output via instructor."""

    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
    ) -> None:
        self.backend = backend or LLM_BACKEND

        if self.backend == "ollama":
            self.model = model or OLLAMA_MODEL
            self._client = self._build_ollama_client()
        elif self.backend == "claude":
            self.model = model or CLAUDE_MODEL
            self._client = self._build_claude_client()
        else:
            raise ValueError(f"Unsupported LLM backend: {self.backend!r}")

    # ------------------------------------------------------------------
    # Backend construction
    # ------------------------------------------------------------------

    def _build_ollama_client(self) -> instructor.Instructor:
        from openai import OpenAI

        raw = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        return instructor.from_openai(raw, mode=instructor.Mode.JSON) # Ollama's OpenAI API is JSON-only so we use that to validate against the pydantic schema.

    def _build_claude_client(self) -> instructor.Instructor:
        import anthropic

        raw = anthropic.Anthropic()
        return instructor.from_anthropic(raw)

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    def complete(self, prompt: str, output_schema: type[T]) -> T:
        """
        Send *prompt* to the LLM and parse the response into *output_schema*.

        Retries up to 3 times on validation failure with exponential backoff.
        Raises RuntimeError after exhausting retries.
        """
        schema_name = output_schema.__name__
        prompt_token_estimate = len(prompt) // 4  # rough approximation

        logger.info(
            "llm_call_started",
            backend=self.backend,
            model=self.model,
            output_schema=schema_name,
            prompt_token_estimate=prompt_token_estimate,
        )

        if LOG_LLM_PROMPTS:
            logger.debug("llm_prompt_text", prompt=prompt)

        last_error: str | None = None
        total_start = time.perf_counter()

        for attempt in range(1, _MAX_RETRIES + 1):
            attempt_start = time.perf_counter()
            try:
                result = self._call_backend(prompt, output_schema)
                latency_ms = (time.perf_counter() - attempt_start) * 1000.0

                logger.info(
                    "llm_call_completed",
                    backend=self.backend,
                    model=self.model,
                    output_schema=schema_name,
                    latency_ms=round(latency_ms, 2),
                    retry_count=attempt - 1,
                    success=True,
                )
                return result

            except Exception as exc:
                latency_ms = (time.perf_counter() - attempt_start) * 1000.0
                last_error = str(exc)

                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "llm_call_retry",
                        backend=self.backend,
                        model=self.model,
                        output_schema=schema_name,
                        attempt=attempt,
                        validation_error=last_error,
                        latency_ms=round(latency_ms, 2),
                        exc_info=True,
                    )
                    time.sleep(_BACKOFF_BASE * (2 ** (attempt - 1)))
                else:
                    total_latency_ms = (time.perf_counter() - total_start) * 1000.0
                    logger.error(
                        "llm_call_failed",
                        backend=self.backend,
                        model=self.model,
                        output_schema=schema_name,
                        total_attempts=_MAX_RETRIES,
                        last_validation_error=last_error,
                        total_latency_ms=round(total_latency_ms, 2),
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"LLM call failed after {_MAX_RETRIES} attempts: {last_error}"
                    ) from exc

        # Unreachable, but satisfies type checkers
        raise RuntimeError("LLM call failed unexpectedly")  # pragma: no cover

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _call_backend(self, prompt: str, output_schema: type[T]) -> T:
        """Dispatch to the correct instructor client."""
        if self.backend == "ollama":
            return self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_model=output_schema,
            )
        elif self.backend == "claude":
            return self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                response_model=output_schema,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend!r}")  # pragma: no cover
