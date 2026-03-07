"""
Oracle structured logging configuration.

Usage in every module:
    from Oracle.logging_config import get_logger
    logger = get_logger(__name__)

Usage at pipeline entrypoints:
    from Oracle.logging_config import get_logger, bind_trace_id, unbind_trace_id
    trace_id = bind_trace_id()
    logger.info("pipeline_started", pipeline="ingestion")
    # ... all downstream logs carry trace_id automatically
    unbind_trace_id()
"""

from __future__ import annotations

import logging
import sys

import structlog
from uuid import uuid4

from Oracle.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logging() -> None:
    """Call once at application startup (main.py / server.py)."""

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if LOG_FORMAT == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(
            file=open(LOG_FILE, "a") if LOG_FILE else sys.stderr  # noqa: SIM115
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger bound to the given module name."""
    return structlog.get_logger(module=name)


def bind_trace_id(trace_id: str | None = None) -> str:
    """
    Generate and bind a trace_id to structlog context vars.
    Returns the trace_id for passing to API responses / SSE events.
    """
    if trace_id is None:
        trace_id = uuid4().hex[:12]
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    return trace_id


def unbind_trace_id() -> None:
    """Clear trace_id from structlog context vars after pipeline completes."""
    structlog.contextvars.unbind_contextvars("trace_id")
