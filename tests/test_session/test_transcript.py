"""Tests for :class:`~genie.session.transcript.Transcript`.

These exercise the SPEC §9.1 on-disk contract: round-tripping
:class:`ChatMessage` records, persisting (but not requiring) ``ts``/``usage``
metadata, tolerating a missing file and blank lines, and crash-durable appends
visible across independent reader/writer instances.
"""

from __future__ import annotations

from pathlib import Path

from genie.providers.base import ChatMessage
from genie.session.transcript import Transcript


def _sample_messages() -> list[ChatMessage]:
    """A user turn, an assistant tool-call turn, and a tool-result turn."""
    return [
        ChatMessage(role="user", content="fix the bug in parse_args"),
        ChatMessage(
            role="assistant",
            content="I'll read the file first.",
            tool_calls=[{"id": "call_1", "name": "read_file", "arguments": {"path": "a.py"}}],
        ),
        ChatMessage(
            role="tool",
            content="def parse_args(): ...",
            tool_call_id="call_1",
        ),
    ]


def test_append_then_read_round_trips(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path / "transcript.jsonl")
    messages = _sample_messages()
    for message in messages:
        transcript.append(message)

    assert transcript.read() == messages


def test_round_trip_preserves_tool_calls_and_tool_call_id(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path / "transcript.jsonl")
    messages = _sample_messages()
    for message in messages:
        transcript.append(message)

    restored = transcript.read()
    assert restored[1].tool_calls == messages[1].tool_calls
    assert restored[2].tool_call_id == "call_1"


def test_list_content_blocks_round_trip(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path / "transcript.jsonl")
    message = ChatMessage(
        role="assistant",
        content=[{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}],
    )
    transcript.append(message)

    assert transcript.read() == [message]


def test_ts_and_usage_persisted_in_records(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path / "transcript.jsonl")
    usage = {"input_tokens": 12, "output_tokens": 3}
    transcript.append(
        ChatMessage(role="user", content="hello"),
        ts="2026-05-29T00:00:00Z",
        usage=usage,
    )

    records = transcript.read_records()
    assert records[0]["ts"] == "2026-05-29T00:00:00Z"
    assert records[0]["usage"] == usage


def test_metadata_omitted_when_not_supplied(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path / "transcript.jsonl")
    transcript.append(ChatMessage(role="user", content="hello"))

    record = transcript.read_records()[0]
    assert "ts" not in record
    assert "usage" not in record


def test_metadata_does_not_affect_message_equality(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path / "transcript.jsonl")
    message = ChatMessage(role="user", content="hello")
    transcript.append(message, ts="2026-05-29T00:00:00Z", usage={"input_tokens": 1})

    assert transcript.read() == [message]


def test_read_missing_file_returns_empty(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path / "does_not_exist.jsonl")
    assert transcript.read() == []
    assert transcript.read_records() == []


def test_blank_and_whitespace_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    transcript = Transcript(path)
    transcript.append(ChatMessage(role="user", content="one"))
    transcript.append(ChatMessage(role="assistant", content="two"))

    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write("   \n")
        fh.write("\t\n")

    messages = transcript.read()
    assert [m.content for m in messages] == ["one", "two"]


def test_durability_across_separate_instances(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"

    writer_a = Transcript(path)
    writer_a.append(ChatMessage(role="user", content="first"))

    writer_b = Transcript(path)
    writer_b.append(ChatMessage(role="assistant", content="second"))

    assert [m.content for m in writer_a.read()] == ["first", "second"]
    assert [m.content for m in writer_b.read()] == ["first", "second"]


def test_append_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "transcript.jsonl"
    transcript = Transcript(path)
    transcript.append(ChatMessage(role="user", content="hi"))

    assert path.exists()
    assert transcript.read()[0].content == "hi"


def test_malformed_trailing_line_does_not_lose_prior_history(tmp_path: Path) -> None:
    """A partial final line (crash mid-write) must not make prior lines unreadable."""
    path = tmp_path / "transcript.jsonl"
    transcript = Transcript(path)
    transcript.append(ChatMessage(role="user", content="one"))
    transcript.append(ChatMessage(role="assistant", content="two"))
    # Simulate a torn final write: append a partial JSON line by hand.
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"role": "user", "content": "thr')  # no newline, truncated

    messages = transcript.read()
    assert [m.content for m in messages] == ["one", "two"]


def test_malformed_interior_line_is_skipped(tmp_path: Path) -> None:
    """A corrupt non-final line is skipped, not fatal; good lines still load."""
    path = tmp_path / "transcript.jsonl"
    path.write_text(
        '{"role": "user", "content": "a"}\n}{ not json\n{"role": "assistant", "content": "b"}\n',
        encoding="utf-8",
    )
    transcript = Transcript(path)
    assert [m.content for m in transcript.read()] == ["a", "b"]
