"""JSONL transcript: the on-disk, append-only record of a conversation.

A :class:`Transcript` is the durable half of SPEC §9.1's session format: one
JSON object per line in ``transcript.jsonl``, each carrying a
:class:`~genie.providers.base.ChatMessage` plus optional per-message metadata
(``ts``, ``usage``). The writer appends and flushes one line at a time so an
interrupted run still leaves every committed message on disk — the property the
replay/resume primitive (SPEC §9.3) depends on.

Timestamps are **never** generated here. ``ts`` is supplied by the caller (or
omitted), which keeps reads byte-for-byte reproducible and lets tests stay
deterministic without monkeypatching the clock.
"""

from __future__ import annotations

import json
from pathlib import Path

from genie.providers.base import ChatMessage
from genie.utils.logger import get_logger

_log = get_logger("genie.session.transcript")


class Transcript:
    """An append-only JSONL reader/writer over :class:`ChatMessage` records.

    Each line is a JSON object with the message fields ``role``, ``content``,
    ``tool_calls``, ``tool_call_id`` plus optional ``ts`` (caller-supplied
    timestamp) and ``usage`` (token accounting). Reconstruction round-trips: a
    message written with :meth:`append` and read back with :meth:`read` is
    equal to the original.
    """

    def __init__(self, path: str | Path) -> None:
        """Bind the transcript to a ``transcript.jsonl`` path.

        The file is not created or opened here; it is touched lazily on the
        first :meth:`append`, and a missing file reads as an empty history.

        Args:
            path: Filesystem path to the transcript JSONL file.
        """
        self.path = Path(path)

    def append(
        self,
        message: ChatMessage,
        *,
        ts: str | None = None,
        usage: dict | None = None,
    ) -> None:
        """Append one message as a single JSON line, flushed to disk.

        The file is opened in append mode and flushed after the write so the
        line survives a crash immediately after this call returns (SPEC §9.1
        crash-durability). ``ts`` and ``usage`` are written only when provided;
        when ``None`` they are omitted from the line rather than stored as null.

        Args:
            message: The :class:`ChatMessage` to persist.
            ts: Optional caller-supplied timestamp string. Not generated here.
            usage: Optional token-accounting dict for this message.
        """
        record: dict[str, object] = {
            "role": message.role,
            "content": message.content,
            "tool_calls": message.tool_calls,
            "tool_call_id": message.tool_call_id,
        }
        if ts is not None:
            record["ts"] = ts
        if usage is not None:
            record["usage"] = usage

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

    def read_records(self) -> list[dict]:
        """Return every line parsed as a raw record dict, metadata included.

        Blank and whitespace-only lines are skipped. A missing file yields an
        empty list. Each record carries the message fields plus any ``ts`` /
        ``usage`` that were persisted, which is how callers inspect metadata
        that :meth:`read` intentionally drops.

        Robustness: a malformed line (e.g. a partial final line left by a crash
        mid-write) is skipped and logged rather than raising — so a single
        interrupted append can never make the prior, fully-committed history
        unreadable. This is what makes the SPEC §9.3 resume primitive safe.

        Returns:
            The list of parsed JSON objects in file order.
        """
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        records: list[dict] = []
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                _log.warning(
                    "transcript_skipped_malformed_line",
                    path=str(self.path),
                    line_no=line_no,
                    trailing=line_no == len(lines),
                )
        return records

    def read(self) -> list[ChatMessage]:
        """Return the persisted messages as :class:`ChatMessage` objects.

        The ``ts`` / ``usage`` metadata is dropped during reconstruction so the
        result is exactly what was appended; use :meth:`read_records` to recover
        the metadata. A missing file yields an empty list.

        Returns:
            The reconstructed messages in file order.
        """
        return [_record_to_message(r) for r in self.read_records()]


def _record_to_message(record: dict) -> ChatMessage:
    """Rebuild a :class:`ChatMessage` from a raw record, ignoring metadata."""
    return ChatMessage(
        role=record["role"],
        content=record["content"],
        tool_calls=record.get("tool_calls"),
        tool_call_id=record.get("tool_call_id"),
    )
