import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

import structlog
from structlog import contextvars as structlog_contextvars

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_configured = False


def bind_request_id(request_id: str | None = None) -> str:
    rid = request_id or uuid4().hex
    _request_id.set(rid)
    structlog_contextvars.bind_contextvars(request_id=rid)
    return rid


def get_request_id() -> str | None:
    return _request_id.get()


def clear_request_id() -> None:
    _request_id.set(None)
    structlog_contextvars.unbind_contextvars("request_id")


@contextmanager
def request_id_context(request_id: str | None = None) -> Iterator[str]:
    rid = bind_request_id(request_id)
    try:
        yield rid
    finally:
        clear_request_id()


def configure_logging(log_level: str | None = None) -> None:
    global _configured
    from pd_assistant.core.config import get_settings

    level_name = (log_level or get_settings().log_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog_contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
