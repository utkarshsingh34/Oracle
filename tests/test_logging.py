"""
Tests that validate structured log events emitted by Oracle modules.

Uses structlog.testing.capture_logs() to capture log events as plain dicts
and assert on their contents.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import structlog
import structlog.testing

from Oracle.logging_config import get_logger, bind_trace_id, unbind_trace_id
from Oracle.llm.client import LLMClient
from Oracle.llm.schemas import FunctionSummary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_structlog_context():
    """
    Clear structlog context vars between tests so trace_ids don't leak.
    capture_logs() handles its own structlog reconfiguration internally.
    """
    structlog.contextvars.unbind_contextvars("trace_id")
    structlog.reset_defaults()
    yield
    structlog.contextvars.unbind_contextvars("trace_id")
    structlog.reset_defaults()


@pytest.fixture()
def ollama_client() -> LLMClient:
    """Build an LLMClient pointing at ollama without connecting."""
    with patch("Oracle.llm.client.LLMClient._build_ollama_client") as mock_build:
        mock_build.return_value = MagicMock()
        client = LLMClient(backend="ollama", model="test-model")
    return client


# ---------------------------------------------------------------------------
# Tests — LLM client logging
# ---------------------------------------------------------------------------

def test_llm_call_logs_on_success(ollama_client: LLMClient) -> None:
    """Verify LLM client emits structured log on successful call."""
    expected = FunctionSummary(summary="Test summary of a function.")
    ollama_client._client.chat.completions.create = MagicMock(return_value=expected)

    with structlog.testing.capture_logs() as logs:
        result = ollama_client.complete("test prompt", FunctionSummary)

    assert result == expected

    completed = [e for e in logs if e.get("event") == "llm_call_completed"]
    assert len(completed) == 1
    assert completed[0]["backend"] == "ollama"
    assert completed[0]["success"] is True
    assert "latency_ms" in completed[0]
    assert completed[0]["retry_count"] == 0


def test_llm_call_logs_retry_on_validation_failure(ollama_client: LLMClient) -> None:
    """Verify LLM client logs WARNING on retry and INFO on eventual success."""
    expected = FunctionSummary(summary="Fixed summary.")
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Validation failed: missing required field 'summary'")
        return expected

    ollama_client._client.chat.completions.create = MagicMock(side_effect=side_effect)

    with patch("Oracle.llm.client.time.sleep"):  # skip actual backoff
        with structlog.testing.capture_logs() as logs:
            result = ollama_client.complete("test prompt", FunctionSummary)

    assert result == expected

    retry_logs = [e for e in logs if e.get("event") == "llm_call_retry"]
    assert len(retry_logs) >= 1
    assert "validation_error" in retry_logs[0]

    completed = [e for e in logs if e.get("event") == "llm_call_completed"]
    assert len(completed) == 1


def test_llm_call_logs_error_after_max_retries(ollama_client: LLMClient) -> None:
    """Verify LLM client logs ERROR after exhausting all retries."""
    ollama_client._client.chat.completions.create = MagicMock(
        side_effect=ValueError("always fails")
    )

    with patch("Oracle.llm.client.time.sleep"):  # skip actual backoff
        with structlog.testing.capture_logs() as logs:
            with pytest.raises(RuntimeError, match="LLM call failed after 5 attempts"):
                ollama_client.complete("test prompt", FunctionSummary)

    failed = [e for e in logs if e.get("event") == "llm_call_failed"]
    assert len(failed) == 1
    assert failed[0]["total_attempts"] == 5
    assert "last_validation_error" in failed[0]


def test_trace_id_binding() -> None:
    """Verify bind_trace_id attaches trace_id to all subsequent log events."""
    # capture_logs() doesn't include merge_contextvars by default,
    # so we configure structlog with it explicitly for this test.
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.testing.ReturnLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    tid = bind_trace_id()
    assert len(tid) == 12

    logger = get_logger("trace_test")
    # ReturnLoggerFactory makes .info() return the rendered string
    output = logger.info("test_event", key="value")

    unbind_trace_id()

    # The rendered output contains the trace_id
    assert tid in output


def test_get_logger_returns_bound_logger() -> None:
    """Verify get_logger binds the module name to every log event."""
    with structlog.testing.capture_logs() as logs:
        logger = get_logger("test_module")
        logger.info("hello", data=42)

    hello_events = [e for e in logs if e.get("event") == "hello"]
    assert len(hello_events) == 1
    assert hello_events[0]["module"] == "test_module"
