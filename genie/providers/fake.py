"""A scripted, deterministic :class:`ProviderClient` for tests.

``FakeProvider`` is the second implementation of the provider contract (the
first being the real SDK adapters). Its existence is the proof that the
abstraction holds: the loop can be driven end-to-end with no network, no SDK,
and fully predictable output. See SPEC operating principle "Replaceability is
testable".
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from genie.providers.base import ChatChunk, ChatMessage, ProviderClient


def _tool_call_fragments(
    index: int, call_id: str, name: str, args: dict, *, arg_chunks: int = 2
) -> list[ChatChunk]:
    """Build streamed fragments for one tool call at slot ``index``.

    The opening fragment carries ``id`` and ``name``; ``arguments`` are split
    into ``arg_chunks`` partial-JSON ``arguments_delta`` pieces — exactly the
    shape a real provider streams (id/name once, args as string fragments).
    """
    arg_json = json.dumps(args)
    n = max(1, arg_chunks)
    size = max(1, -(-len(arg_json) // n))
    pieces = [arg_json[i : i + size] for i in range(0, len(arg_json), size)] or [""]
    fragments = [
        ChatChunk(
            tool_call_delta={
                "index": index,
                "id": call_id,
                "name": name,
                "arguments_delta": pieces[0],
            }
        )
    ]
    fragments += [
        ChatChunk(tool_call_delta={"index": index, "arguments_delta": piece})
        for piece in pieces[1:]
    ]
    return fragments


class FakeProvider(ProviderClient):
    """A provider that replays a pre-scripted sequence of streamed turns.

    Construct with ``turns`` as either a list of turns (where each turn is a
    ``list[ChatChunk]``) for multi-turn conversations, or a flat
    ``list[ChatChunk]`` for a single turn. Each call to :meth:`stream` replays
    the next scripted turn in order, advancing an internal cursor so the loop
    can drive a multi-turn conversation deterministically.

    Calling :meth:`stream` more times than there are scripted turns raises
    :class:`IndexError` with a clear message, so over-driving the script fails
    loudly rather than hanging.

    Every call records its ``messages``, ``tools``, and ``system`` into the
    public :attr:`calls` list, letting tests assert exactly what the loop sent.
    """

    name = "fake"

    def __init__(
        self,
        turns: list[list[ChatChunk]] | list[ChatChunk] | None = None,
        *,
        model: str = "fake-1",
    ) -> None:
        """Build a fake provider.

        Args:
            turns: Either a list of turns (each a ``list[ChatChunk]``) or a
                flat ``list[ChatChunk]`` treated as a single turn. ``None`` is
                an empty script.
            model: The model identifier reported by this provider.
        """
        self.model = model
        self._turns: list[list[ChatChunk]] = self._normalize(turns)
        self._cursor = 0
        self.calls: list[dict] = []

    @staticmethod
    def _normalize(
        turns: list[list[ChatChunk]] | list[ChatChunk] | None,
    ) -> list[list[ChatChunk]]:
        """Coerce the constructor argument into a list of turns."""
        if turns is None:
            return []
        if turns and isinstance(turns[0], ChatChunk):
            # A flat list of chunks is a single turn.
            return [turns]  # type: ignore[list-item]
        return turns  # type: ignore[return-value]

    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system: str | None = None,
        cache_breakpoints: list[int] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Replay the next scripted turn, recording the call first.

        Raises:
            IndexError: If invoked more times than there are scripted turns.
        """
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "cache_breakpoints": cache_breakpoints,
            }
        )
        if self._cursor >= len(self._turns):
            raise IndexError(
                f"FakeProvider exhausted: {len(self._turns)} scripted turn(s), "
                f"but stream() was called {self._cursor + 1} time(s)"
            )
        turn = self._turns[self._cursor]
        self._cursor += 1
        for chunk in turn:
            yield chunk

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        """Return a deterministic cheap estimate: ~4 chars per token, min 1."""
        chars = sum(len(str(m.content)) for m in messages)
        return max(1, chars // 4)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        model: str = "fake-1",
        chunks: int = 3,
        usage: dict | None = None,
    ) -> FakeProvider:
        """Build a single-turn provider that streams ``text`` then stops.

        The text is split into roughly ``chunks`` ``delta_text`` pieces,
        followed by a terminal chunk with ``finish_reason="stop"``. The
        concatenated ``delta_text`` of the streamed chunks equals ``text``. When
        ``usage`` is given it rides the terminal chunk, so cost/ledger paths can
        be exercised.
        """
        n = max(1, chunks)
        size = max(1, -(-len(text) // n))
        pieces = [text[i : i + size] for i in range(0, len(text), size)] or [""]
        turn = [ChatChunk(delta_text=p) for p in pieces]
        turn.append(ChatChunk(finish_reason="stop", usage=usage))
        return cls([turn], model=model)

    @classmethod
    def with_tool_call(
        cls,
        name: str,
        args: dict,
        *,
        call_id: str = "call_1",
        model: str = "fake-1",
        usage: dict | None = None,
    ) -> FakeProvider:
        """Build a single-turn provider that streams one tool call then stops.

        The arguments are streamed as partial-JSON ``arguments_delta`` fragments
        at slot ``index=0`` (id/name on the first fragment), exactly as a real
        provider streams them, followed by a chunk with
        ``finish_reason="tool_calls"`` carrying optional ``usage``.
        """
        return cls.with_tool_calls([(name, args)], call_ids=[call_id], model=model, usage=usage)

    @classmethod
    def with_tool_calls(
        cls,
        calls: list[tuple[str, dict]],
        *,
        call_ids: list[str] | None = None,
        model: str = "fake-1",
        usage: dict | None = None,
    ) -> FakeProvider:
        """Build a single-turn provider that streams *parallel* tool calls.

        Each ``(name, args)`` in ``calls`` is assigned a distinct slot
        ``index`` and its fragments are emitted in order; this is the fixture
        that exercises the loop's parallel tool-call dispatch (SPEC §5.3). A
        terminal ``finish_reason="tool_calls"`` chunk (with optional ``usage``)
        closes the turn.
        """
        ids = call_ids or [f"call_{i + 1}" for i in range(len(calls))]
        turn: list[ChatChunk] = []
        for index, ((name, args), call_id) in enumerate(zip(calls, ids, strict=True)):
            turn += _tool_call_fragments(index, call_id, name, args)
        turn.append(ChatChunk(finish_reason="tool_calls", usage=usage))
        return cls([turn], model=model)
