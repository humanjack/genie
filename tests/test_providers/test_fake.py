"""Tests for FakeProvider — the proof the provider abstraction holds."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from genie.providers.base import ChatChunk, ChatMessage
from genie.providers.fake import FakeProvider


async def collect(agen: AsyncIterator[ChatChunk]) -> list[ChatChunk]:
    """Drain an async generator of chunks into a list."""
    return [c async for c in agen]


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


async def test_with_tool_call_emits_delta_and_finish() -> None:
    provider = FakeProvider.with_tool_call("read_file", {"path": "a.py"}, call_id="c9")
    chunks = await collect(provider.stream([], []))

    delta = chunks[0].tool_call_delta
    assert delta is not None
    assert delta["name"] == "read_file"
    assert delta["arguments"] == {"path": "a.py"}
    assert delta["id"] == "c9"
    assert chunks[-1].finish_reason == "tool_calls"


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
