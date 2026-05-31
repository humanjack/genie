"""Tests for the provider contract dataclasses and ABC."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from genie.providers.base import ChatChunk, ChatMessage, ProviderClient


def test_provider_client_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ProviderClient()  # type: ignore[abstract]


def test_minimal_concrete_subclass_works() -> None:
    class Minimal(ProviderClient):
        name = "minimal"
        model = "m-1"

        async def stream(
            self,
            messages,
            tools,
            *,
            max_tokens=4096,
            temperature=0.0,
            system=None,
            cache_breakpoints=None,
        ) -> AsyncIterator[ChatChunk]:
            yield ChatChunk(delta_text="hi")

        def count_tokens(self, messages) -> int:
            return len(messages)

    client = Minimal()
    assert client.name == "minimal"
    assert client.model == "m-1"
    assert client.count_tokens([ChatMessage(role="user", content="x")]) == 1


def test_chat_message_defaults() -> None:
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


def test_chat_message_accepts_content_blocks() -> None:
    blocks = [{"type": "text", "text": "hi"}]
    msg = ChatMessage(role="assistant", content=blocks, tool_calls=[{"id": "c1"}])
    assert msg.content == blocks
    assert msg.tool_calls == [{"id": "c1"}]


def test_chat_chunk_defaults() -> None:
    chunk = ChatChunk()
    assert chunk.delta_text is None
    assert chunk.tool_call_delta is None
    assert chunk.finish_reason is None
    assert chunk.usage is None


def test_chat_chunk_fields_set() -> None:
    usage = {"input_tokens": 1, "output_tokens": 2, "cache_read": 0, "cache_write": 0}
    chunk = ChatChunk(delta_text="x", finish_reason="stop", usage=usage)
    assert chunk.delta_text == "x"
    assert chunk.finish_reason == "stop"
    assert chunk.usage == usage
