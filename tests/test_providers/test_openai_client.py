"""Tests for OpenAIClient — the chat-completions adapter for the contract.

No network: the OpenAI SDK is replaced by a fake async-iterable stream injected
via ``client=``. Fixtures mimic ``ChatCompletionChunk`` (``choices[].delta`` /
``.finish_reason`` and the separate usage chunk) with ``SimpleNamespace`` so the
chunk-mapping logic is exercised against the exact attribute shapes the real SDK
emits.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from genie.providers.base import ChatChunk, ChatMessage
from genie.providers.openai_client import (
    OpenAIClient,
    _translate_messages,
    _translate_tools,
)


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


def _text_delta(content: str) -> SimpleNamespace:
    """Build a chunk with a single choice carrying ``delta.content``."""
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=None)
    return SimpleNamespace(choices=[choice], usage=None)


def _tool_call_chunk(
    index: int,
    *,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    """Build a chunk carrying one ``ChoiceDeltaToolCall`` fragment."""
    function = SimpleNamespace(name=name, arguments=arguments)
    tool_call = SimpleNamespace(index=index, id=id, function=function, type="function")
    delta = SimpleNamespace(content=None, tool_calls=[tool_call])
    choice = SimpleNamespace(delta=delta, finish_reason=None)
    return SimpleNamespace(choices=[choice], usage=None)


def _finish_chunk(reason: str) -> SimpleNamespace:
    """Build a chunk whose first choice carries a ``finish_reason``."""
    delta = SimpleNamespace(content=None, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=reason)
    return SimpleNamespace(choices=[choice], usage=None)


def _usage_chunk(prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    """Build the trailing usage chunk that OpenAI emits with empty choices."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[], usage=usage)


class _FakeStream:
    """An async-iterable over a fixed list of fake chunks."""

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> _FakeStream:
        self._it = iter(self._chunks)
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


class _FakeCompletions:
    """Captures the ``create`` kwargs and returns a scripted stream."""

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks
        self.create_kwargs: dict | None = None

    async def create(self, **kwargs) -> _FakeStream:
        self.create_kwargs = kwargs
        return _FakeStream(self._chunks)


class _FakeClient:
    """Minimal stand-in for ``AsyncOpenAI`` exposing ``chat.completions``."""

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self.completions = _FakeCompletions(chunks)
        self.chat = SimpleNamespace(completions=self.completions)


def _make_client(chunks: list[SimpleNamespace], **kwargs) -> tuple[OpenAIClient, _FakeClient]:
    """Build an OpenAIClient wired to a fake SDK client over ``chunks``."""
    fake = _FakeClient(chunks)
    client = OpenAIClient(model="gpt-4o-mini", client=fake, **kwargs)
    return client, fake


async def test_text_turn_reassembles_with_stop_and_usage() -> None:
    chunks = [
        _text_delta("Hello"),
        _text_delta(", "),
        _text_delta("world"),
        _finish_chunk("stop"),
        _usage_chunk(10, 4),
    ]
    client, _ = _make_client(chunks)

    out = await collect(client.stream([ChatMessage(role="user", content="hi")], []))

    text = "".join(c.delta_text for c in out if c.delta_text is not None)
    assert text == "Hello, world"
    finish = [c for c in out if c.finish_reason is not None]
    assert finish[0].finish_reason == "stop"
    usage = [c for c in out if c.usage is not None]
    assert usage[0].usage == {"input_tokens": 10, "output_tokens": 4}


async def test_single_tool_call_streams_fragments() -> None:
    chunks = [
        _tool_call_chunk(0, id="call_abc", name="read_file", arguments=""),
        _tool_call_chunk(0, arguments='{"path": '),
        _tool_call_chunk(0, arguments='"a.py"}'),
        _finish_chunk("tool_calls"),
        _usage_chunk(7, 3),
    ]
    client, _ = _make_client(chunks)

    out = await collect(client.stream([], [{"name": "read_file"}]))

    first = out[0].tool_call_delta
    assert first is not None
    assert first["index"] == 0
    assert first["id"] == "call_abc"
    assert first["name"] == "read_file"
    # Later fragments carry neither id nor name (set once on the opener).
    second = out[1].tool_call_delta
    assert second is not None
    assert "id" not in second
    assert "name" not in second

    assert _accumulate_args(out) == {0: {"path": "a.py"}}
    finish = [c for c in out if c.finish_reason is not None]
    assert finish[0].finish_reason == "tool_calls"


async def test_two_parallel_tool_calls_distinct_indices() -> None:
    chunks = [
        _tool_call_chunk(0, id="c1", name="read_file", arguments='{"path":'),
        _tool_call_chunk(1, id="c2", name="bash", arguments='{"cmd":'),
        _tool_call_chunk(0, arguments=' "a.py"}'),
        _tool_call_chunk(1, arguments=' "ls"}'),
        _finish_chunk("tool_calls"),
    ]
    client, _ = _make_client(chunks)

    out = await collect(client.stream([], []))

    indices = {c.tool_call_delta["index"] for c in out if c.tool_call_delta is not None}
    assert indices == {0, 1}
    assert _accumulate_args(out) == {0: {"path": "a.py"}, 1: {"cmd": "ls"}}


def test_translate_tools_wraps_as_function_tools() -> None:
    tools = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]
    wrapped = _translate_tools(tools)
    assert wrapped == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]


def test_translate_tools_empty_is_none() -> None:
    assert _translate_tools([]) is None


def test_translate_messages_system_and_tool_result() -> None:
    messages = [
        ChatMessage(role="user", content="run it"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "bash"}}],
        ),
        ChatMessage(role="tool", content="done", tool_call_id="c1"),
    ]
    out = _translate_messages(messages, system="be terse")

    assert out[0] == {"role": "system", "content": "be terse"}
    assert out[1]["role"] == "user"
    assert out[2]["tool_calls"][0]["id"] == "c1"
    assert out[3] == {"role": "tool", "content": "done", "tool_call_id": "c1"}


def test_translate_messages_maps_neutral_tool_calls_to_native() -> None:
    """The loop appends NEUTRAL tool_calls; the adapter must produce OpenAI-native.

    Regression for #63: passing the neutral shape through verbatim made
    multi-turn tool conversations 400 on the 2nd OpenAI call.
    """
    # This mirrors exactly what genie.loop appends after one tool round-trip.
    messages = [
        ChatMessage(role="user", content="read a.txt"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "name": "read_file", "arguments": {"path": "a.txt"}}],
        ),
        ChatMessage(role="tool", content="contents", tool_call_id="call_1"),
    ]
    out = _translate_messages(messages, system=None)

    tc = out[1]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "read_file"
    # arguments must be a JSON-encoded STRING, not a dict.
    assert tc["function"]["arguments"] == '{"path": "a.txt"}'
    assert out[2] == {"role": "tool", "content": "contents", "tool_call_id": "call_1"}


async def test_stream_passes_tools_and_stream_options() -> None:
    client, fake = _make_client([_finish_chunk("stop")])
    await collect(
        client.stream(
            [ChatMessage(role="user", content="hi")],
            [{"name": "noop", "description": "d", "input_schema": {"type": "object"}}],
            max_tokens=128,
            temperature=0.2,
            system="sys",
        )
    )
    kwargs = fake.completions.create_kwargs
    assert kwargs is not None
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["max_tokens"] == 128
    assert kwargs["temperature"] == 0.2
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert kwargs["tools"][0]["type"] == "function"


async def test_cache_breakpoints_is_noop() -> None:
    client, fake = _make_client([_finish_chunk("stop")])
    await collect(client.stream([], [], cache_breakpoints=[0, 1]))
    # cache_breakpoints must not leak into the OpenAI request.
    assert "cache_breakpoints" not in (fake.completions.create_kwargs or {})


async def test_responses_api_mode_raises() -> None:
    client = OpenAIClient(model="gpt-4o-mini", client=_FakeClient([]), api="responses")
    with pytest.raises(NotImplementedError, match=r"provider\.openai\.api"):
        await collect(client.stream([], []))


def test_api_mode_resolved_from_settings() -> None:
    settings = SimpleNamespace(provider=SimpleNamespace(openai=SimpleNamespace(api="responses")))
    client = OpenAIClient(model="gpt-4o-mini", settings=settings, client=_FakeClient([]))
    assert client._api == "responses"


def test_api_mode_defaults_without_settings() -> None:
    client = OpenAIClient(model="gpt-4o-mini", client=_FakeClient([]))
    assert client._api == "chat_completions"


async def test_real_default_config_streams_without_raising() -> None:
    """Regression: a default-configured OpenAI run must reach chat_completions, not raise.

    Drives the *real* ``config.Settings`` default through ``stream()`` — guards
    against the config default ever drifting back to an unimplemented ``api``.
    """
    from genie.config import Settings

    settings = Settings()
    assert settings.provider.openai.api == "chat_completions"
    finish = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="stop"
            )
        ],
        usage=None,
    )
    client = OpenAIClient(
        model="gpt-4o-mini", settings=settings, client=_FakeClient([_text_delta("ok"), finish])
    )
    out = await collect(client.stream([ChatMessage(role="user", content="hi")], []))
    assert any(c.finish_reason == "stop" for c in out)


def test_count_tokens_positive() -> None:
    client = OpenAIClient(model="gpt-4o-mini", client=_FakeClient([]))
    n = client.count_tokens([ChatMessage(role="user", content="some content here")])
    assert isinstance(n, int)
    assert n > 0


def test_count_tokens_empty_is_at_least_one() -> None:
    client = OpenAIClient(model="gpt-4o-mini", client=_FakeClient([]))
    assert client.count_tokens([]) == 1


def test_lazy_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAIClient(model="gpt-4o-mini")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        client._ensure_client()


def test_lazy_client_uses_settings_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-settings")
    calls: list[tuple] = []

    class _Settings:
        def require_api_key(self, name: str, env) -> str:
            calls.append((name, env.get("OPENAI_API_KEY")))
            return env["OPENAI_API_KEY"]

    client = OpenAIClient(model="gpt-4o-mini", settings=_Settings())
    built = client._ensure_client()

    from openai import AsyncOpenAI

    assert isinstance(built, AsyncOpenAI)
    assert calls == [("openai", "sk-from-settings")]


def test_lazy_client_built_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient(model="gpt-4o-mini")
    built = client._ensure_client()
    # A real AsyncOpenAI was constructed (no network call made).
    from openai import AsyncOpenAI

    assert isinstance(built, AsyncOpenAI)


def test_name_attribute() -> None:
    client = OpenAIClient(model="gpt-4o-mini", client=_FakeClient([]))
    assert client.name == "openai"
    assert client.model == "gpt-4o-mini"


@pytest.mark.skipif(not os.getenv("RUN_LIVE_API"), reason="set RUN_LIVE_API to hit the network")
async def test_live_openai() -> None:
    """Smoke-test a real call when RUN_LIVE_API is set (manual, networked)."""
    client = OpenAIClient(model="gpt-4o-mini")
    out = await collect(
        client.stream([ChatMessage(role="user", content="Say 'pong'.")], [], max_tokens=16)
    )
    text = "".join(c.delta_text for c in out if c.delta_text is not None)
    assert text
    assert any(c.finish_reason is not None for c in out)
