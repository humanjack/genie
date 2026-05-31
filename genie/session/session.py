"""Session: the per-conversation state and its on-disk home (SPEC §9).

A :class:`Session` ties together the in-memory message list the loop hands the
provider and the durable artifacts under ``<root>/<id>/``:

- ``meta.json`` — ``id``, ``parent_id``, ``model``, ``working_dir`` and an
  optional ``started_at`` (SPEC §9.1).
- ``transcript.jsonl`` — the append-only message log (see
  :class:`~genie.session.transcript.Transcript`).

The ``parent_id`` field is the seed of SPEC §9.2 tree sessions: a forked child
records its parent here. :meth:`resume` is the SPEC §9.3 replay primitive — it
rebuilds a live :class:`Session` from disk so a saved conversation can be
inspected or re-run.

Identity and time are kept **injectable**: ``id`` and ``started_at`` are passed
in, never generated with ``uuid4``/``datetime.now`` inside this module, so
construction and replay are deterministic and testable.
"""

from __future__ import annotations

import json
from pathlib import Path

from genie.providers.base import ChatMessage
from genie.session.transcript import Transcript


class Session:
    """A single conversation's state plus its durable transcript and metadata.

    Attributes:
        id: Stable session identifier; also the directory name under ``root``.
        parent_id: The id this session was forked from, or ``None`` for a root
            session (SPEC §9.2 tree sessions).
        working_dir: The directory the agent operates in for this session.
        model: The provider/model identifier in effect at creation.
        messages: The in-memory conversation the loop materializes for the
            provider; kept in sync with the on-disk transcript by :meth:`append`.
        transcript: The append-only JSONL log backing :attr:`messages`.
    """

    def __init__(
        self,
        *,
        id: str,
        model: str,
        working_dir: str,
        transcript: Transcript,
        parent_id: str | None = None,
        messages: list[ChatMessage] | None = None,
    ) -> None:
        """Construct a session from already-resolved parts.

        Prefer the :meth:`create` and :meth:`resume` factories; this initializer
        wires up an instance once its directory layout and message history are
        known.

        Args:
            id: Stable session identifier.
            model: Provider/model identifier.
            working_dir: Directory the agent operates in.
            transcript: The transcript backing this session's messages.
            parent_id: Optional parent session id (tree sessions).
            messages: Optional initial in-memory messages (used by
                :meth:`resume`); defaults to empty.
        """
        self.id = id
        self.model = model
        self.working_dir = working_dir
        self.parent_id = parent_id
        self.transcript = transcript
        self.messages: list[ChatMessage] = list(messages) if messages else []

    @staticmethod
    def dir_for(root: Path, id: str) -> Path:
        """Return the session directory ``<root>/<id>`` for ``id``."""
        return Path(root) / id

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        id: str,
        model: str,
        working_dir: str | None = None,
        parent_id: str | None = None,
        started_at: str | None = None,
    ) -> Session:
        """Create a fresh session directory and return the live session.

        Builds ``<root>/<id>/`` containing ``meta.json`` and an empty
        ``transcript.jsonl``. ``id`` and ``started_at`` are caller-supplied so
        creation is deterministic; nothing here reads the clock or generates a
        uuid. When ``working_dir`` is ``None`` it defaults to the session
        directory itself, so the field is always populated.

        Args:
            root: Parent directory under which the session dir is created.
            id: Stable session identifier and directory name.
            model: Provider/model identifier to record.
            working_dir: Directory the agent operates in; defaults to the
                session directory when omitted.
            parent_id: Optional parent session id (tree sessions).
            started_at: Optional caller-supplied creation timestamp; omitted
                from ``meta.json`` when ``None``.

        Returns:
            A :class:`Session` with an empty message list and transcript.
        """
        session_dir = cls.dir_for(root, id)
        session_dir.mkdir(parents=True, exist_ok=True)
        resolved_working_dir = working_dir if working_dir is not None else str(session_dir)

        meta = _Meta(
            id=id,
            parent_id=parent_id,
            model=model,
            working_dir=resolved_working_dir,
            started_at=started_at,
        )
        _write_meta(session_dir, meta)

        transcript = Transcript(session_dir / "transcript.jsonl")
        transcript.path.touch()

        return cls(
            id=id,
            model=model,
            working_dir=resolved_working_dir,
            transcript=transcript,
            parent_id=parent_id,
        )

    @classmethod
    def resume(cls, root: Path, id: str) -> Session:
        """Rebuild a session from disk — the SPEC §9.3 replay primitive.

        Reads ``<root>/<id>/meta.json`` for identity/model/working_dir/parent
        and replays ``transcript.jsonl`` back into :attr:`messages`, yielding a
        session equal (modulo metadata) to the one that was persisted.

        Args:
            root: Parent directory the session lives under.
            id: Identifier of the session to resume.

        Returns:
            A :class:`Session` populated with the persisted messages.

        Raises:
            FileNotFoundError: If the session's ``meta.json`` does not exist.
        """
        session_dir = cls.dir_for(root, id)
        meta = _read_meta(session_dir)
        transcript = Transcript(session_dir / "transcript.jsonl")
        return cls(
            id=meta.id,
            model=meta.model,
            working_dir=meta.working_dir,
            transcript=transcript,
            parent_id=meta.parent_id,
            messages=transcript.read(),
        )

    def append(
        self,
        message: ChatMessage,
        *,
        ts: str | None = None,
        usage: dict | None = None,
    ) -> None:
        """Record a message in memory and durably in the transcript.

        Appends to :attr:`messages` (what the loop will send next) and writes
        the same message — with optional ``ts``/``usage`` metadata — to the
        on-disk transcript, keeping the two in lockstep.

        Args:
            message: The :class:`ChatMessage` to record.
            ts: Optional caller-supplied timestamp for the transcript line.
            usage: Optional token-accounting dict for the transcript line.
        """
        self.messages.append(message)
        self.transcript.append(message, ts=ts, usage=usage)

    def materialize_messages(self) -> list[ChatMessage]:
        """Return the message list the loop hands the provider.

        For Phase 1 this is simply a copy of the in-memory :attr:`messages`.
        System-prompt and memory injection (SPEC §8) are later phases; keeping
        this a method lets the context pipeline wrap it without touching the
        loop or callers.

        Returns:
            A shallow copy of the current in-memory messages.
        """
        return list(self.messages)


class _Meta:
    """The contents of ``meta.json`` (SPEC §9.1), without a generated clock."""

    __slots__ = ("id", "model", "parent_id", "started_at", "working_dir")

    def __init__(
        self,
        *,
        id: str,
        model: str,
        working_dir: str,
        parent_id: str | None = None,
        started_at: str | None = None,
    ) -> None:
        self.id = id
        self.model = model
        self.working_dir = working_dir
        self.parent_id = parent_id
        self.started_at = started_at


def _meta_path(session_dir: Path) -> Path:
    """Return the ``meta.json`` path inside ``session_dir``."""
    return session_dir / "meta.json"


def _write_meta(session_dir: Path, meta: _Meta) -> None:
    """Write ``meta`` to ``meta.json``, omitting ``started_at`` when unset."""
    record: dict[str, object] = {
        "id": meta.id,
        "parent_id": meta.parent_id,
        "model": meta.model,
        "working_dir": meta.working_dir,
    }
    if meta.started_at is not None:
        record["started_at"] = meta.started_at
    _meta_path(session_dir).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_meta(session_dir: Path) -> _Meta:
    """Read and parse ``meta.json`` from ``session_dir``."""
    record = json.loads(_meta_path(session_dir).read_text(encoding="utf-8"))
    return _Meta(
        id=record["id"],
        model=record["model"],
        working_dir=record["working_dir"],
        parent_id=record.get("parent_id"),
        started_at=record.get("started_at"),
    )
