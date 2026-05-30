"""Structured logging wrapper (SPEC §14.1 "Observability").

A thin layer over ``structlog`` providing three guarantees the rest of genie
relies on:

* **Per-session correlation.** :func:`bind_session` stamps a ``session_id`` onto
  a logger so every downstream event can be traced to one session.
* **Secret hygiene.** :func:`_redact_secrets` is a processor in the chain that
  masks values for sensitive keys — including keys **nested** inside dicts and
  lists, and affixed variants like ``x-api-key`` / ``auth_token`` — replacing
  them with ``"***"``. The key vocabulary (:data:`SECRET_KEYS`) is a
  module-level constant so it is extensible without touching the processor.
* **Pluggable rendering.** The final processor is the single pluggable seam:
  a human-friendly console renderer by default, or a JSON renderer when
  ``json_output=True``. Swapping renderers requires no change to call sites.

:func:`configure_logging` is idempotent: :func:`structlog.configure` *replaces*
the processor chain wholesale, so calling it repeatedly never stacks processors.

**Known limitation.** Redaction is *key*-based. A secret embedded in a free-text
event message (``log.info("token=sk-…")``) or carried as the *value* of a
benign key is not detected — pass secrets as values of well-named keys.
"""

from __future__ import annotations

import logging
import re
from itertools import pairwise

import structlog
from structlog.typing import FilteringBoundLogger

SECRET_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "token",
        "access_token",
        "refresh_token",
        "session_token",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "private_key",
        "credential",
        "credentials",
        "cookie",
        "bearer",
    }
)
"""Key vocabulary whose values are redacted before rendering.

Module-level so callers can extend redaction (e.g. ``SECRET_KEYS | {"otp"}``)
without modifying the processor. Matching is component-based (see
:func:`_is_secret_key`), so ``input_tokens`` is *not* treated as a secret even
though it contains the substring ``token``.
"""

_REDACTED = "***"
_SEPARATORS = re.compile(r"[-_.\s/]+")


def _is_secret_key(key: str) -> bool:
    """Return True if ``key`` names a sensitive value.

    Matching is case-insensitive and works on word *components*, not raw
    substrings, so it catches ``api_key``, ``x-api-key``, ``auth_token`` and
    ``Authorization`` while leaving ``input_tokens`` / ``output_tokens`` (cost
    telemetry) untouched. A key matches when:

    * its lowercased form is in :data:`SECRET_KEYS`, or
    * any separator-delimited component is in :data:`SECRET_KEYS`, or
    * any adjacent component pair, joined, is in :data:`SECRET_KEYS`
      (so ``x-api-key`` → ``api`` + ``key`` → ``apikey``).
    """
    low = key.lower()
    if low in SECRET_KEYS:
        return True
    parts = [p for p in _SEPARATORS.split(low) if p]
    if any(p in SECRET_KEYS for p in parts):
        return True
    for a, b in pairwise(parts):
        if f"{a}{b}" in SECRET_KEYS or f"{a}_{b}" in SECRET_KEYS:
            return True
    return False


def _redact_value(value: object) -> object:
    """Recursively redact secrets in nested dicts/lists; pass scalars through."""
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _is_secret_key(str(k)) else _redact_value(v)) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_value(v) for v in value)
    return value


def _redact_secrets(
    logger: object, method_name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """structlog processor that masks sensitive keys anywhere in the event dict.

    Top-level sensitive keys are replaced with ``"***"``; every other value is
    walked recursively so secrets nested inside dicts and lists are caught too.
    """
    for key in list(event_dict.keys()):
        if _is_secret_key(str(key)):
            event_dict[key] = _REDACTED
        else:
            event_dict[key] = _redact_value(event_dict[key])
    return event_dict


def configure_logging(*, json_output: bool = False, level: str = "INFO") -> None:
    """Configure the global structlog processor chain.

    Idempotent: :func:`structlog.configure` replaces the chain, so repeated calls
    (startup, then a test fixture) never stack processors. ``level`` gates events
    via :func:`structlog.make_filtering_bound_logger`; events below it are
    dropped before rendering. The final renderer is the pluggable seam:
    :class:`structlog.dev.ConsoleRenderer` for humans by default, or
    :class:`structlog.processors.JSONRenderer` when ``json_output=True``.

    Raises :class:`ValueError` if ``level`` is not a known logging level.
    """
    level_map = logging.getLevelNamesMapping()
    level_value = level_map.get(level.upper())
    if level_value is None:
        valid = ", ".join(sorted(level_map))
        raise ValueError(f"invalid log level {level!r}; choose one of: {valid}")

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
        wrapper_class=structlog.make_filtering_bound_logger(level_value),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str, **initial_context: object) -> FilteringBoundLogger:
    """Return a logger bound with ``name`` and any ``initial_context``.

    ``name`` is bound under the ``logger`` key; remaining keyword arguments are
    bound verbatim as initial event context.
    """
    return structlog.get_logger(name, **initial_context)


def bind_session(logger: FilteringBoundLogger, session_id: str) -> FilteringBoundLogger:
    """Return a new logger carrying ``session_id`` as a correlation id.

    A thin wrapper over ``logger.bind(session_id=...)``; the original logger is
    left unbound so callers can hold a session-scoped child independently.
    """
    return logger.bind(session_id=session_id)
