"""Tests for FakeProvider — the proof the provider abstraction holds."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from genie.providers.base import ChatChunk, ChatMessage
from genie.providers.fake import FakeProvider


async def collect(agen: AsyncIterator[ChatChunk]) -> list[ChatChunk]:
    """Drain an async generator of chunks into a list."""
    return [c async for c in agen]


def _accumulate_args(chunks: list[ChatChunk]) -> dict[int, dict]:
    """Reassemble per-slot tool-call arguments the way the loop will.

    Joins ``arguments_delta`` fragments by ``index`` and ``json.loads`` them —
    the canonical consumer of the streaming contract.
    """
    buffers: dict[int, str] = {}
    for c in chunks:
        d = c.tool_call_delta
        if d is None:
            continue
        buffers.setdefault(d["index"], "")
        if d.get("arguments_delta"):
            buffers[d["index"]] += d["arguments_delta"]
    return {i: json.loads(b) for i, b in buffers.items()}


async def test_from_text_streams_and_stops() -> None:
    provider = FakeProvider.from_text("hello world")
    chunks = await collect(provider.stream([], []))

    assert chunks[-1].finish_reason == "stop"
    text = "".join(c.delta_text for c in chunks if c.delta_text is not None)
    assert text == "hello world"
    assert provider.model == "fake-1"
    assert provider.name == "fake"


async def test_from_text_empty_string() -> None:
    provider = FakeProvider.from_text("")
    chunks = await collect(provider.stream([], []))
    text = "".join(c.delta_text or "" for c in chunks)
    assert text == ""
    assert chunks[-1].finish_reason == "stop"


async def test_with_tool_call_streams_fragments_and_finishes() -> None:
    provider = FakeProvider.with_tool_call("read_file", {"path": "a.py"}, call_id="c9")
    chunks = await collect(provider.stream([], []))

    # First fragment carries id + name at slot 0; args arrive as JSON fragments.
    first = chunks[0].tool_call_delta
    assert first is not None
    assert first["index"] == 0
    assert first["name"] == "read_file"
    assert first["id"] == "c9"
    assert "arguments" not in first  # never a pre-parsed dict

    # Reassembling the streamed arguments_delta yields the original args.
    reassembled = _accumulate_args(chunks)
    assert reassembled[0] == {"path": "a.py"}
    assert chunks[-1].finish_reason == "tool_calls"


async def test_with_tool_calls_assigns_distinct_indices() -> None:
    """Parallel tool calls each get their own slot index (drives §5.3 dispatch)."""
    provider = FakeProvider.with_tool_calls(
        [("read_file", {"path": "a.py"}), ("bash", {"cmd": "ls"})],
        call_ids=["c1", "c2"],
    )
    chunks = await collect(provider.stream([], []))

    indices = {c.tool_call_delta["index"] for c in chunks if c.tool_call_delta is not None}
    assert indices == {0, 1}
    reassembled = _accumulate_args(chunks)
    assert reassembled == {0: {"path": "a.py"}, 1: {"cmd": "ls"}}
    assert chunks[-1].finish_reason == "tool_calls"


async def test_usage_rides_terminal_chunk() -> None:
    usage = {"input_tokens": 10, "output_tokens": 5}
    text_provider = FakeProvider.from_text("hi", usage=usage)
    chunks = await collect(text_provider.stream([], []))
    assert chunks[-1].usage == usage

    tool_provider = FakeProvider.with_tool_call("noop", {}, usage=usage)
    tchunks = await collect(tool_provider.stream([], []))
    assert tchunks[-1].usage == usage


async def test_multi_turn_advances_per_call() -> None:
    turn1 = [ChatChunk(delta_text="one"), ChatChunk(finish_reason="stop")]
    turn2 = [ChatChunk(delta_text="two"), ChatChunk(finish_reason="stop")]
    provider = FakeProvider([turn1, turn2])

    first = await collect(provider.stream([], []))
    second = await collect(provider.stream([], []))

    assert first[0].delta_text == "one"
    assert second[0].delta_text == "two"


async def test_exhausted_script_raises() -> None:
    provider = FakeProvider([[ChatChunk(finish_reason="stop")]])
    await collect(provider.stream([], []))
    with pytest.raises(IndexError, match="exhausted"):
        await collect(provider.stream([], []))


async def test_flat_chunk_list_is_single_turn() -> None:
    provider = FakeProvider([ChatChunk(delta_text="flat"), ChatChunk(finish_reason="stop")])
    chunks = await collect(provider.stream([], []))
    assert chunks[0].delta_text == "flat"


async def test_calls_records_arguments() -> None:
    provider = FakeProvider.from_text("hi")
    messages = [ChatMessage(role="user", content="ping")]
    tools = [{"name": "noop"}]
    await collect(provider.stream(messages, tools, system="be nice", cache_breakpoints=[0]))

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["messages"] is messages
    assert call["tools"] is tools
    assert call["system"] == "be nice"
    assert call["cache_breakpoints"] == [0]


def test_count_tokens_positive() -> None:
    provider = FakeProvider()
    n = provider.count_tokens([ChatMessage(role="user", content="some content here")])
    assert isinstance(n, int)
    assert n > 0


def test_count_tokens_empty_is_at_least_one() -> None:
    provider = FakeProvider()
    assert provider.count_tokens([]) == 1


def test_custom_model() -> None:
    provider = FakeProvider(model="fake-xl")
    assert provider.model == "fake-xl"


def test_none_script_is_empty() -> None:
    provider = FakeProvider(None)
    assert provider._turns == []
