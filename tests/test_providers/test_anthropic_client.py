"""Tests for :class:`AnthropicClient` — offline, with the SDK fully mocked.

A fake async-streaming context manager replays ``SimpleNamespace`` events shaped
exactly like the real ``anthropic.types`` raw stream events (``.type``,
``content_block.type == "tool_use"``, ``delta.partial_json``/``delta.text``,
``message.usage``, ``delta.stop_reason``). Injecting it via the ``client=``
kwarg lets :meth:`AnthropicClient.stream` run with no network.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from genie.providers.anthropic_client import AnthropicClient
from genie.providers.base import ChatChunk, ChatMessage


async def collect(agen: AsyncIterator[ChatChunk]) -> list[ChatChunk]:
    """Drain an async generator of chunks into a list."""
    return [c async for c in agen]


def _accumulate_args(chunks: list[ChatChunk]) -> dict[int, dict]:
    """Reassemble per-slot tool-call arguments the way the loop will."""
    buffers: dict[int, str] = {}
    for c in chunks:
        d = c.tool_call_delta
        if d is None:
            continue
        buffers.setdefault(d["index"], "")
        if d.get("arguments_delta"):
            buffers[d["index"]] += d["arguments_delta"]
    return {i: json.loads(b) for i, b in buffers.items()}


# --- fake SDK event/stream builders -------------------------------------------


def ev_message_start(*, input_tokens=10, cache_read=None, cache_write=None) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )
    return SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage))


def ev_text(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        index=0,
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def ev_tool_start(index: int, call_id: str, name: str) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", id=call_id, name=name, input={})
    return SimpleNamespace(type="content_block_start", index=index, content_block=block)


def ev_tool_json(index: int, partial_json: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="input_json_delta", partial_json=partial_json),
    )


def ev_message_delta(stop_reason: str, *, output_tokens=5) -> SimpleNamespace:
    return SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason=stop_reason),
        usage=SimpleNamespace(output_tokens=output_tokens),
    )


class FakeStream:
    """Async context manager + async iterator over a fixed list of events."""

    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = events

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        for e in self._events:
            yield e


class FakeMessages:
    """Mimics ``client.messages`` and records the params it was streamed with."""

    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = events
        self.captured_params: dict | None = None

    def stream(self, **params) -> FakeStream:
        self.captured_params = params
        return FakeStream(self._events)


class FakeClient:
    """Mimics ``anthropic.AsyncAnthropic`` for injection via ``client=``."""

    def __init__(self, events: list[SimpleNamespace]) -> None:
        self.messages = FakeMessages(events)


def _make_client(events: list[SimpleNamespace], *, model="claude-sonnet-4-6") -> AnthropicClient:
    return AnthropicClient(model=model, client=FakeClient(events))


# --- streaming behavior -------------------------------------------------------


async def test_text_only_turn_reassembles_and_stops() -> None:
    events = [
        ev_message_start(input_tokens=12),
        ev_text("Hello"),
        ev_text(", world"),
        ev_message_delta("end_turn", output_tokens=7),
    ]
    client = _make_client(events)
    chunks = await collect(client.stream([ChatMessage(role="user", content="hi")], []))

    text = "".join(c.delta_text for c in chunks if c.delta_text is not None)
    assert text == "Hello, world"
    terminal = chunks[-1]
    assert terminal.finish_reason == "stop"
    assert terminal.usage == {"input_tokens": 12, "output_tokens": 7}
    assert client.name == "anthropic"
    assert client.model == "claude-sonnet-4-6"


async def test_single_tool_call_streams_index_id_name_and_args() -> None:
    args = {"path": "a.py", "limit": 50}
    arg_json = json.dumps(args)
    events = [
        ev_message_start(),
        ev_tool_start(0, "tool_abc", "read_file"),
        ev_tool_json(0, arg_json[:5]),
        ev_tool_json(0, arg_json[5:]),
        ev_message_delta("tool_use"),
    ]
    chunks = await collect(_make_client(events).stream([], []))

    first = chunks[0].tool_call_delta
    assert first is not None
    assert first["index"] == 0
    assert first["id"] == "tool_abc"
    assert first["name"] == "read_file"
    assert first["arguments_delta"] == ""

    # Subsequent fragments carry only the index + a partial-JSON string.
    second = chunks[1].tool_call_delta
    assert second is not None
    assert "id" not in second and "name" not in second
    assert isinstance(second["arguments_delta"], str)

    assert _accumulate_args(chunks) == {0: args}
    assert chunks[-1].finish_reason == "tool_calls"


async def test_two_parallel_tool_calls_use_distinct_indices() -> None:
    args0 = {"path": "a.py"}
    args1 = {"cmd": "ls -la"}
    j0, j1 = json.dumps(args0), json.dumps(args1)
    events = [
        ev_message_start(),
        ev_tool_start(0, "call_0", "read_file"),
        ev_tool_start(1, "call_1", "bash"),
        ev_tool_json(0, j0[:4]),
        ev_tool_json(1, j1[:4]),
        ev_tool_json(0, j0[4:]),
        ev_tool_json(1, j1[4:]),
        ev_message_delta("tool_use"),
    ]
    chunks = await collect(_make_client(events).stream([], []))

    indices = {c.tool_call_delta["index"] for c in chunks if c.tool_call_delta is not None}
    assert indices == {0, 1}
    assert _accumulate_args(chunks) == {0: args0, 1: args1}
    assert chunks[-1].finish_reason == "tool_calls"


async def test_stop_reason_mapping_passthrough() -> None:
    # end_turn -> stop
    end = await collect(_make_client([ev_message_delta("end_turn")]).stream([], []))
    assert end[-1].finish_reason == "stop"
    # tool_use -> tool_calls
    tool = await collect(_make_client([ev_message_delta("tool_use")]).stream([], []))
    assert tool[-1].finish_reason == "tool_calls"
    # unknown reason passes through unchanged
    other = await collect(_make_client([ev_message_delta("max_tokens")]).stream([], []))
    assert other[-1].finish_reason == "max_tokens"


async def test_usage_includes_cache_tokens_when_present() -> None:
    events = [
        ev_message_start(input_tokens=20, cache_read=8, cache_write=3),
        ev_message_delta("end_turn", output_tokens=4),
    ]
    chunks = await collect(_make_client(events).stream([], []))
    assert chunks[-1].usage == {
        "input_tokens": 20,
        "cache_read": 8,
        "cache_write": 3,
        "output_tokens": 4,
    }


# --- param translation --------------------------------------------------------


def _client() -> AnthropicClient:
    return AnthropicClient(model="claude-sonnet-4-6", client=FakeClient([]))


def test_tools_and_system_forwarded_unchanged() -> None:
    tools = [{"name": "read_file", "description": "read", "input_schema": {"type": "object"}}]
    params = _client()._build_params(
        [ChatMessage(role="user", content="hi")],
        tools,
        max_tokens=100,
        temperature=0.2,
        system="be terse",
        cache_breakpoints=None,
    )
    assert params["tools"] is tools
    assert params["system"] == "be terse"
    assert params["model"] == "claude-sonnet-4-6"
    assert params["max_tokens"] == 100
    assert params["temperature"] == 0.2
    assert params["messages"] == [{"role": "user", "content": "hi"}]


def test_tool_call_and_result_messages_translate_to_blocks() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[{"id": "t1", "name": "read_file", "arguments": {"path": "a.py"}}],
        ),
        ChatMessage(role="tool", content="file contents", tool_call_id="t1"),
    ]
    params = _client()._build_params(
        messages, [], max_tokens=10, temperature=0.0, system=None, cache_breakpoints=None
    )
    assert params["messages"][0] == {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}}
        ],
    }
    assert params["messages"][1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file contents"}],
    }
    assert "system" not in params  # None system omitted


def test_cache_breakpoint_applied_to_right_message() -> None:
    messages = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="user", content="second"),
    ]
    params = _client()._build_params(
        messages, [], max_tokens=10, temperature=0.0, system=None, cache_breakpoints=[1]
    )
    # Index 0 untouched (plain string), index 1 promoted to a cached text block.
    assert params["messages"][0] == {"role": "user", "content": "first"}
    assert params["messages"][1]["content"] == [
        {"type": "text", "text": "second", "cache_control": {"type": "ephemeral"}}
    ]


def test_assistant_text_preserved_alongside_tool_calls() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content="thinking out loud",
            tool_calls=[{"id": "t1", "name": "bash", "arguments": {"cmd": "ls"}}],
        )
    ]
    params = _client()._build_params(
        messages, [], max_tokens=10, temperature=0.0, system=None, cache_breakpoints=None
    )
    assert params["messages"][0]["content"] == [
        {"type": "text", "text": "thinking out loud"},
        {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
    ]


def test_assistant_list_content_preserved_alongside_tool_calls() -> None:
    blocks = [{"type": "text", "text": "pre-existing block"}]
    messages = [
        ChatMessage(
            role="assistant",
            content=blocks,
            tool_calls=[{"id": "t2", "name": "noop", "arguments": {}}],
        )
    ]
    params = _client()._build_params(
        messages, [], max_tokens=10, temperature=0.0, system=None, cache_breakpoints=None
    )
    assert params["messages"][0]["content"] == [
        {"type": "text", "text": "pre-existing block"},
        {"type": "tool_use", "id": "t2", "name": "noop", "input": {}},
    ]


def test_out_of_range_cache_breakpoint_is_ignored() -> None:
    params = _client()._build_params(
        [ChatMessage(role="user", content="only")],
        [],
        max_tokens=10,
        temperature=0.0,
        system=None,
        cache_breakpoints=[5],
    )
    assert params["messages"][0] == {"role": "user", "content": "only"}


# --- count_tokens & lazy client / key resolution ------------------------------


def test_count_tokens_positive() -> None:
    client = _client()
    n = client.count_tokens([ChatMessage(role="user", content="some content here")])
    assert isinstance(n, int)
    assert n > 0


def test_count_tokens_empty_is_at_least_one() -> None:
    assert _client().count_tokens([]) == 1


def test_construct_without_key_does_not_explode() -> None:
    # No client, no key — construction must succeed; only stream() would need one.
    client = AnthropicClient(model="claude-sonnet-4-6")
    assert client.model == "claude-sonnet-4-6"


async def test_stream_without_key_raises_actionable_error(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient(model="claude-sonnet-4-6")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        await collect(client.stream([], []))


def test_env_api_key_resolved_without_settings(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    assert AnthropicClient(model="claude-sonnet-4-6")._resolve_api_key() == "sk-from-env"


async def test_settings_require_api_key_consulted(monkeypatch) -> None:
    class FakeSettings:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def require_api_key(self, name, env):
            self.calls.append((name, env))
            return "sk-from-settings"

    settings = FakeSettings()
    client = AnthropicClient(model="claude-sonnet-4-6", settings=settings)
    # _resolve_api_key should route through settings, not the env var.
    assert client._resolve_api_key() == "sk-from-settings"
    assert settings.calls[0][0] == "anthropic"


# --- live (skipped offline) ---------------------------------------------------


async def test_live_anthropic() -> None:
    if not os.getenv("RUN_LIVE_API"):
        pytest.skip("set RUN_LIVE_API to run the live Anthropic smoke test")
    client = AnthropicClient(model="claude-sonnet-4-6")
    chunks = await collect(
        client.stream(
            [ChatMessage(role="user", content="Say hi in one word.")],
            [],
            max_tokens=1,
        )
    )
    assert chunks[-1].finish_reason is not None
