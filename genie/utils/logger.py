"""Structured logging wrapper (SPEC §14.1 "Observability").

A thin layer over ``structlog`` providing three guarantees the rest of genie
relies on:

* **Per-session correlation.** :func:`bind_session` stamps a ``session_id`` onto
  a logger so every downstream event can be traced to one session.
* **Secret hygiene.** :func:`_redact_secrets` is a processor in the chain that
  never lets values for sensitive keys reach a sink; it replaces them with
  ``"***"``. The key set (:data:`SECRET_KEYS`) is a module-level constant so it
  is extensible without touching the processor.
* **Pluggable rendering.** The final processor is the single pluggable seam:
  a human-friendly console renderer by default, or a JSON renderer when
  ``json_output=True``. Swapping renderers requires no change to call sites.

:func:`configure_logging` is idempotent — the structlog ``cache_logger_on_first_use``
flag plus a re-runnable :func:`structlog.configure` call mean it can be invoked
repeatedly without double-wrapping the processor chain.
"""

from __future__ import annotations

import logging

import structlog

SECRET_KEYS: frozenset[str] = frozenset(
    {"api_key", "apikey", "authorization", "token", "password", "secret"}
)
"""Event-dict keys whose values are redacted before rendering.

Module-level so callers can extend redaction (e.g. ``SECRET_KEYS | {"cookie"}``)
without modifying the processor itself.
"""

_REDACTED = "***"


def _redact_secrets(
    logger: object, method_name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """structlog processor that masks values for keys in :data:`SECRET_KEYS`.

    Matching is case-insensitive. Non-secret keys pass through untouched, so the
    event dict is mutated in place only where a sensitive key is found.
    """
    for key in event_dict:
        if key.lower() in SECRET_KEYS:
            event_dict[key] = _REDACTED
    return event_dict


def configure_logging(*, json_output: bool = False, level: str = "INFO") -> None:
    """Configure the global structlog processor chain.

    Idempotent: safe to call more than once (e.g. once at startup and again in a
    test fixture) without stacking processors. ``level`` gates events via
    :func:`structlog.make_filtering_bound_logger`; events below it are dropped
    before rendering. The final renderer is the pluggable seam:
    :class:`structlog.dev.ConsoleRenderer` for humans by default, or
    :class:`structlog.processors.JSONRenderer` when ``json_output=True``.
    """
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _redact_secrets,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **initial_context: object) -> structlog.stdlib.BoundLogger:
    """Return a logger bound with ``name`` and any ``initial_context``.

    ``name`` is bound under the ``logger`` key; remaining keyword arguments are
    bound verbatim as initial event context.
    """
    return structlog.get_logger(name, **initial_context)


def bind_session(
    logger: structlog.stdlib.BoundLogger, session_id: str
) -> structlog.stdlib.BoundLogger:
    """Return a new logger carrying ``session_id`` as a correlation id.

    A thin wrapper over ``logger.bind(session_id=...)``; the original logger is
    left unbound so callers can hold a session-scoped child independently.
    """
    return logger.bind(session_id=session_id)
