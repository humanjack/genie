"""Tests for :class:`~genie.session.session.Session`.

These cover SPEC §9.1 creation (``meta.json`` + empty transcript), the
in-memory/on-disk append invariant, the §9.3 resume/replay primitive that the
e2e tests rely on, and §9.2 tree-session ``parent_id`` recording.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genie.providers.base import ChatMessage
from genie.session.session import Session, SessionError


def _conversation() -> list[ChatMessage]:
    """A three-message conversation spanning all ChatMessage shapes."""
    return [
        ChatMessage(role="user", content="add a test"),
        ChatMessage(
            role="assistant",
            content="running the suite",
            tool_calls=[{"id": "c1", "name": "bash", "arguments": {"cmd": "pytest"}}],
        ),
        ChatMessage(role="tool", content="1 passed", tool_call_id="c1"),
    ]


def test_create_writes_meta_json(tmp_path: Path) -> None:
    Session.create(
        tmp_path,
        id="s1",
        model="anthropic:claude-sonnet-4-6",
        working_dir="/repo",
        parent_id=None,
    )

    meta = json.loads((tmp_path / "s1" / "meta.json").read_text(encoding="utf-8"))
    assert meta == {
        "id": "s1",
        "parent_id": None,
        "model": "anthropic:claude-sonnet-4-6",
        "working_dir": "/repo",
    }


def test_create_makes_session_dir_and_empty_transcript(tmp_path: Path) -> None:
    session = Session.create(tmp_path, id="s1", model="m", working_dir="/repo")

    session_dir = tmp_path / "s1"
    transcript_path = session_dir / "transcript.jsonl"
    assert session_dir.is_dir()
    assert transcript_path.exists()
    assert transcript_path.read_text(encoding="utf-8") == ""
    assert session.materialize_messages() == []


def test_create_started_at_recorded_when_given(tmp_path: Path) -> None:
    Session.create(
        tmp_path, id="s1", model="m", working_dir="/repo", started_at="2026-05-29T07:00:00Z"
    )

    meta = json.loads((tmp_path / "s1" / "meta.json").read_text(encoding="utf-8"))
    assert meta["started_at"] == "2026-05-29T07:00:00Z"


def test_create_started_at_omitted_by_default(tmp_path: Path) -> None:
    Session.create(tmp_path, id="s1", model="m", working_dir="/repo")

    meta = json.loads((tmp_path / "s1" / "meta.json").read_text(encoding="utf-8"))
    assert "started_at" not in meta


def test_working_dir_defaults_to_session_dir(tmp_path: Path) -> None:
    session = Session.create(tmp_path, id="s1", model="m")

    expected = str(tmp_path / "s1")
    assert session.working_dir == expected
    meta = json.loads((tmp_path / "s1" / "meta.json").read_text(encoding="utf-8"))
    assert meta["working_dir"] == expected


def test_append_updates_memory_and_disk(tmp_path: Path) -> None:
    session = Session.create(tmp_path, id="s1", model="m", working_dir="/repo")
    message = ChatMessage(role="user", content="hello")

    session.append(message)

    assert session.materialize_messages() == [message]
    assert session.transcript.read() == [message]


def test_append_persists_metadata(tmp_path: Path) -> None:
    session = Session.create(tmp_path, id="s1", model="m", working_dir="/repo")
    session.append(
        ChatMessage(role="user", content="hello"),
        ts="2026-05-29T00:00:00Z",
        usage={"input_tokens": 5},
    )

    record = session.transcript.read_records()[0]
    assert record["ts"] == "2026-05-29T00:00:00Z"
    assert record["usage"] == {"input_tokens": 5}


def test_materialize_returns_copy_not_alias(tmp_path: Path) -> None:
    session = Session.create(tmp_path, id="s1", model="m", working_dir="/repo")
    session.append(ChatMessage(role="user", content="hello"))

    snapshot = session.materialize_messages()
    snapshot.append(ChatMessage(role="user", content="mutation"))
    assert len(session.materialize_messages()) == 1


def test_resume_rebuilds_messages_and_meta(tmp_path: Path) -> None:
    original = Session.create(
        tmp_path,
        id="s1",
        model="anthropic:claude-sonnet-4-6",
        working_dir="/repo",
        started_at="2026-05-29T07:00:00Z",
    )
    conversation = _conversation()
    for message in conversation:
        original.append(message, ts="2026-05-29T07:00:01Z")

    resumed = Session.resume(tmp_path, "s1")

    assert resumed.materialize_messages() == conversation
    assert resumed.id == "s1"
    assert resumed.model == "anthropic:claude-sonnet-4-6"
    assert resumed.working_dir == "/repo"
    assert resumed.parent_id is None


def test_resume_can_continue_appending(tmp_path: Path) -> None:
    original = Session.create(tmp_path, id="s1", model="m", working_dir="/repo")
    original.append(ChatMessage(role="user", content="first"))

    resumed = Session.resume(tmp_path, "s1")
    resumed.append(ChatMessage(role="assistant", content="second"))

    again = Session.resume(tmp_path, "s1")
    assert [m.content for m in again.materialize_messages()] == ["first", "second"]


def test_resume_missing_session_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Session.resume(tmp_path, "ghost")


def test_child_session_records_parent_id(tmp_path: Path) -> None:
    Session.create(tmp_path, id="parent", model="m", working_dir="/repo")
    child = Session.create(tmp_path, id="child", model="m", working_dir="/repo", parent_id="parent")

    assert child.parent_id == "parent"
    meta = json.loads((tmp_path / "child" / "meta.json").read_text(encoding="utf-8"))
    assert meta["parent_id"] == "parent"
    assert Session.resume(tmp_path, "child").parent_id == "parent"


def test_resume_restores_started_at(tmp_path: Path) -> None:
    """started_at survives create -> resume so re-saved meta keeps creation time."""
    Session.create(
        tmp_path, id="s1", model="m", working_dir="/repo", started_at="2026-01-01T00:00:00Z"
    )
    resumed = Session.resume(tmp_path, "s1")
    assert resumed.started_at == "2026-01-01T00:00:00Z"


def test_resume_corrupt_meta_raises_session_error(tmp_path: Path) -> None:
    Session.create(tmp_path, id="s1", model="m", working_dir="/repo")
    (tmp_path / "s1" / "meta.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(SessionError, match="corrupt"):
        Session.resume(tmp_path, "s1")


def test_resume_meta_missing_key_raises_session_error(tmp_path: Path) -> None:
    Session.create(tmp_path, id="s1", model="m", working_dir="/repo")
    (tmp_path / "s1" / "meta.json").write_text('{"id": "s1"}', encoding="utf-8")
    with pytest.raises(SessionError, match="missing required key"):
        Session.resume(tmp_path, "s1")
