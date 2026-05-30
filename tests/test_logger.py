"""Tests for the structured logger (SPEC §14.1)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import structlog
from structlog.testing import capture_logs

from genie.utils.logger import (
    SECRET_KEYS,
    _redact_secrets,
    bind_session,
    configure_logging,
    get_logger,
)


def test_get_logger_emits_bound_name_and_context() -> None:
    """get_logger binds the name and initial context onto emitted events."""
    logger = get_logger("genie.test", component="loop")
    with capture_logs() as logs:
        logger.info("started", turn=1)
    assert len(logs) == 1
    entry = logs[0]
    assert entry["event"] == "started"
    assert entry["component"] == "loop"
    assert entry["turn"] == 1


def test_bind_session_adds_session_id() -> None:
    """bind_session stamps session_id onto every subsequent log entry."""
    logger = bind_session(get_logger("genie.test"), session_id="sess-123")
    with capture_logs() as logs:
        logger.info("first")
        logger.info("second")
    assert [entry["session_id"] for entry in logs] == ["sess-123", "sess-123"]


def test_bind_session_does_not_mutate_original() -> None:
    """bind_session returns a child; the original logger stays unbound."""
    base = get_logger("genie.test")
    bind_session(base, session_id="sess-xyz")
    with capture_logs() as logs:
        base.info("event")
    assert "session_id" not in logs[0]


@pytest.mark.parametrize("key", sorted(SECRET_KEYS))
def test_redact_secrets_masks_each_sensitive_key(key: str) -> None:
    """Each key in SECRET_KEYS has its value replaced with '***'."""
    event_dict = {"event": "call", key: "super-secret-value", "user": "alice"}
    result = _redact_secrets(None, "info", event_dict)
    assert result[key] == "***"
    assert result["user"] == "alice"
    assert result["event"] == "call"


def test_redact_secrets_is_case_insensitive() -> None:
    """Redaction matches keys regardless of case."""
    result = _redact_secrets(None, "info", {"API_KEY": "abc", "Authorization": "Bearer x"})
    assert result["API_KEY"] == "***"
    assert result["Authorization"] == "***"


def test_redact_secrets_leaves_non_secrets_intact() -> None:
    """Non-secret keys are untouched by the redaction processor."""
    event_dict = {"event": "ok", "path": "/tmp/x", "count": 3}
    result = _redact_secrets(None, "info", event_dict)
    assert result == event_dict


def test_configure_logging_default_is_idempotent() -> None:
    """Default console configuration runs and can be called twice safely."""
    configure_logging()
    configure_logging()
    get_logger("genie.test").info("hello")


def test_configure_logging_json_is_idempotent() -> None:
    """JSON configuration runs and can be called twice safely."""
    configure_logging(json_output=True)
    configure_logging(json_output=True)
    get_logger("genie.test").info("hello")


def test_level_filtering_suppresses_debug_when_info(capsys: pytest.CaptureFixture[str]) -> None:
    """A DEBUG event is dropped, but an INFO event passes, when level is INFO."""
    configure_logging(level="INFO")
    logger = get_logger("genie.test")
    logger.debug("debug-suppressed")
    logger.info("info-shown")
    out = capsys.readouterr().out
    assert "debug-suppressed" not in out
    assert "info-shown" in out


def test_level_filtering_allows_debug_when_debug(capsys: pytest.CaptureFixture[str]) -> None:
    """A DEBUG event is emitted when the configured level is DEBUG."""
    configure_logging(level="DEBUG")
    logger = get_logger("genie.test")
    logger.debug("debug-shown")
    out = capsys.readouterr().out
    assert "debug-shown" in out


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Restore structlog defaults after tests that reconfigure it."""
    yield
    structlog.reset_defaults()
