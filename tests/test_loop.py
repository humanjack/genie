"""Tests for the ReAct agent loop (:mod:`genie.loop`).

The loop is the integration point of the merged seams, so these tests wire up
the *real* :class:`~genie.tools.registry.ToolRegistry`,
:class:`~genie.hooks.manager.HookManager`, and
:class:`~genie.session.session.Session` (under ``tmp_path``) and drive them with
the scripted :class:`~genie.providers.fake.FakeProvider` — no network, fully
deterministic. They cover: a text-only stop, single/parallel/sequential tool
dispatch, malformed-argument and unknown-tool tolerance, a blocking
``before_tool_call`` hook (US-3), the ``before_model_call`` veto, the
``max_iterations`` safety budget, and ``usage`` propagation to the transcript.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from genie.hooks.manager import HookManager, HookOutcome
from genie.loop import (
    ToolCall,
    TurnResult,
    dispatch_tool_calls,
    run_turn,
)
from genie.providers.base import ChatChunk, ChatMessage
from genie.providers.fake import FakeProvider
from genie.session.session import Session
from genie.tools.base import tool
from genie.tools.registry import ToolRegistry
from genie.tools.result import ToolResult


def _session(tmp_path: Path) -> Session:
    """A fresh on-disk session rooted under ``tmp_path``."""
    return Session.create(tmp_path, id="s1", model="fake-1", working_dir=str(tmp_path))


def _registry(*tools) -> ToolRegistry:
    """A registry holding ``tools`` in registration order."""
    registry = ToolRegistry()
    registry.register_all(tools)
    return registry


# --------------------------------------------------------------------------- #
# text-only turn
# --------------------------------------------------------------------------- #


async def test_text_only_turn_stops_and_streams(tmp_path: Path) -> None:
    session = _session(tmp_path)
    provider = FakeProvider.from_text("hello world")
    registry = ToolRegistry()
    hooks = HookManager()
    seen: list[str] = []

    result = await run_turn(session, provider, registry, hooks, on_text_delta=seen.append)

    assert isinstance(result, TurnResult)
    assert result.stopped is True
    assert result.stop_reason == "model_stopped"
    assert result.iterations == 1
    assert result.last_message is not None
    assert result.last_message.content == "hello world"
    # The assistant message was appended to the session.
    messages = session.materialize_messages()
    assert messages[-1].role == "assistant"
    assert messages[-1].content == "hello world"
    # on_text_delta received the streamed fragments, concatenating to the text.
    assert "".join(seen) == "hello world"


# --------------------------------------------------------------------------- #
# single tool call then a stopping second turn
# --------------------------------------------------------------------------- #


async def test_single_tool_call_executes_then_stops(tmp_path: Path) -> None:
    written: list[str] = []

    @tool(name="write_note")
    async def write_note(text: str) -> ToolResult:
        """Record a note (sentinel side effect)."""
        written.append(text)
        return ToolResult.text(f"wrote: {text}")

    session = _session(tmp_path)
    registry = _registry(write_note)
    hooks = HookManager()
    # Turn 1 calls the tool; turn 2 stops with text.
    provider = FakeProvider(
        [
            FakeProvider.with_tool_call("write_note", {"text": "hi"})._turns[0],
            FakeProvider.from_text("done")._turns[0],
        ]
    )

    result = await run_turn(session, provider, registry, hooks)

    # The tool actually ran (sentinel observed).
    assert written == ["hi"]
    assert result.stop_reason == "model_stopped"
    assert result.iterations == 2

    messages = session.materialize_messages()
    # assistant(tool_call) -> tool(result) -> assistant(text)
    assert [m.role for m in messages] == ["assistant", "tool", "assistant"]
    tool_msg = messages[1]
    assert tool_msg.tool_call_id == "call_1"
    assert tool_msg.content == "wrote: hi"
    assert messages[2].content == "done"


# --------------------------------------------------------------------------- #
# parallel tool calls
# --------------------------------------------------------------------------- #


async def test_parallel_tool_calls_both_execute_in_order(tmp_path: Path) -> None:
    @tool(name="upper")
    async def upper(text: str) -> ToolResult:
        """Uppercase text."""
        return ToolResult.text(text.upper())

    @tool(name="reverse")
    async def reverse(text: str) -> ToolResult:
        """Reverse text."""
        return ToolResult.text(text[::-1])

    session = _session(tmp_path)
    registry = _registry(upper, reverse)
    hooks = HookManager()
    provider = FakeProvider(
        [
            FakeProvider.with_tool_calls(
                [("upper", {"text": "ab"}), ("reverse", {"text": "cd"})]
            )._turns[0],
            FakeProvider.from_text("ok")._turns[0],
        ]
    )

    await run_turn(session, provider, registry, hooks)

    messages = session.materialize_messages()
    assert [m.role for m in messages] == ["assistant", "tool", "tool", "assistant"]
    # Results appended in the order of the calls (call_1 then call_2).
    assert messages[1].tool_call_id == "call_1"
    assert messages[1].content == "AB"
    assert messages[2].tool_call_id == "call_2"
    assert messages[2].content == "dc"


# --------------------------------------------------------------------------- #
# sequential serialization
# --------------------------------------------------------------------------- #


async def test_sequential_tool_forces_serial_execution(tmp_path: Path) -> None:
    events: list[str] = []

    @tool(name="seq", sequential=True)
    async def seq(label: str) -> ToolResult:
        """A sequential tool that records its start and end."""
        events.append(f"start:{label}")
        await asyncio.sleep(0.01)
        events.append(f"end:{label}")
        return ToolResult.text(label)

    calls = [
        ToolCall(id="c1", name="seq", args={"label": "a"}),
        ToolCall(id="c2", name="seq", args={"label": "b"}),
    ]
    results = await dispatch_tool_calls(calls, _registry(seq), HookManager(), _session(tmp_path))

    assert [r.content for r in results] == ["a", "b"]
    # Serial execution => each call fully completes before the next starts; a
    # concurrent run would interleave the starts ahead of the first end.
    assert events == ["start:a", "end:a", "start:b", "end:b"]


async def test_parallel_tools_actually_interleave(tmp_path: Path) -> None:
    """Counterpart to the serial test: non-sequential calls run concurrently."""
    events: list[str] = []

    @tool(name="par")
    async def par(label: str) -> ToolResult:
        """A concurrent tool that records its start and end."""
        events.append(f"start:{label}")
        await asyncio.sleep(0.01)
        events.append(f"end:{label}")
        return ToolResult.text(label)

    calls = [
        ToolCall(id="c1", name="par", args={"label": "a"}),
        ToolCall(id="c2", name="par", args={"label": "b"}),
    ]
    await dispatch_tool_calls(calls, _registry(par), HookManager(), _session(tmp_path))

    # Both started before either ended (interleaved under asyncio.gather).
    assert events[:2] == ["start:a", "start:b"]


# --------------------------------------------------------------------------- #
# malformed tool arguments
# --------------------------------------------------------------------------- #


def _malformed_args_turn(name: str, call_id: str = "call_1") -> list[ChatChunk]:
    """A hand-built turn whose ``arguments_delta`` is invalid JSON."""
    return [
        ChatChunk(
            tool_call_delta={
                "index": 0,
                "id": call_id,
                "name": name,
                "arguments_delta": "{not: valid",
            }
        ),
        ChatChunk(finish_reason="tool_calls"),
    ]


async def test_malformed_tool_args_feeds_error_and_continues(tmp_path: Path) -> None:
    ran: list[str] = []

    @tool(name="noop")
    async def noop(text: str) -> ToolResult:
        """Should never run with bad args."""
        ran.append(text)
        return ToolResult.text("ran")

    session = _session(tmp_path)
    registry = _registry(noop)
    hooks = HookManager()
    provider = FakeProvider(
        [_malformed_args_turn("noop"), FakeProvider.from_text("recovered")._turns[0]]
    )

    result = await run_turn(session, provider, registry, hooks)

    # The handler was never invoked; the loop did not crash and went on.
    assert ran == []
    assert result.stop_reason == "model_stopped"
    messages = session.materialize_messages()
    assert messages[1].role == "tool"
    assert "could not parse arguments" in messages[1].content


async def test_empty_argument_stream_parses_to_empty_dict(tmp_path: Path) -> None:
    """A tool call whose argument stream is empty/whitespace becomes ``args={}``."""
    ran: list[bool] = []

    @tool(name="ping")
    async def ping() -> ToolResult:
        """A zero-argument tool."""
        ran.append(True)
        return ToolResult.text("pong")

    empty_args_turn = [
        ChatChunk(
            tool_call_delta={
                "index": 0,
                "id": "call_1",
                "name": "ping",
                "arguments_delta": "   ",
            }
        ),
        ChatChunk(finish_reason="tool_calls"),
    ]
    session = _session(tmp_path)
    provider = FakeProvider([empty_args_turn, FakeProvider.from_text("ok")._turns[0]])

    await run_turn(session, provider, _registry(ping), HookManager())

    assert ran == [True]
    messages = session.materialize_messages()
    assert messages[1].role == "tool"
    assert messages[1].content == "pong"


# --------------------------------------------------------------------------- #
# blocking before_tool_call hook (US-3)
# --------------------------------------------------------------------------- #


async def test_blocking_before_tool_call_hook_denies_but_continues(
    tmp_path: Path,
) -> None:
    ran: list[str] = []

    @tool(name="danger")
    async def danger(text: str) -> ToolResult:
        """A tool the policy hook will deny."""
        ran.append(text)
        return ToolResult.text("ran")

    async def deny(event: str, **payload: object) -> HookOutcome:
        return HookOutcome.blocked("not allowed")

    session = _session(tmp_path)
    registry = _registry(danger)
    hooks = HookManager()
    hooks.on("before_tool_call", deny)
    provider = FakeProvider(
        [
            FakeProvider.with_tool_call("danger", {"text": "x"})._turns[0],
            FakeProvider.from_text("acknowledged")._turns[0],
        ]
    )

    result = await run_turn(session, provider, registry, hooks)

    # The tool did NOT run; the denial was fed back; the session continued.
    assert ran == []
    assert result.stop_reason == "model_stopped"
    messages = session.materialize_messages()
    assert messages[1].role == "tool"
    assert "blocked" in messages[1].content
    assert "not allowed" in messages[1].content


# --------------------------------------------------------------------------- #
# unknown tool name from the model
# --------------------------------------------------------------------------- #


async def test_unknown_tool_name_yields_error_no_crash(tmp_path: Path) -> None:
    session = _session(tmp_path)
    registry = ToolRegistry()  # empty: every tool name is unknown
    hooks = HookManager()
    provider = FakeProvider(
        [
            FakeProvider.with_tool_call("ghost", {"a": 1})._turns[0],
            FakeProvider.from_text("ok")._turns[0],
        ]
    )

    result = await run_turn(session, provider, registry, hooks)

    assert result.stop_reason == "model_stopped"
    messages = session.materialize_messages()
    assert messages[1].role == "tool"
    assert messages[1].content == "unknown tool: ghost"


# --------------------------------------------------------------------------- #
# max_iterations safety budget
# --------------------------------------------------------------------------- #


async def test_max_iterations_stops_a_runaway_loop(tmp_path: Path) -> None:
    @tool(name="loop_tool")
    async def loop_tool(n: int) -> ToolResult:
        """A tool that never lets the model settle."""
        return ToolResult.text(str(n))

    # A provider that always returns the same tool call, every turn.
    one_turn = FakeProvider.with_tool_call("loop_tool", {"n": 1})._turns[0]
    provider = FakeProvider([list(one_turn) for _ in range(10)])
    session = _session(tmp_path)
    registry = _registry(loop_tool)
    hooks = HookManager()

    result = await run_turn(session, provider, registry, hooks, max_iterations=3)

    assert result.stopped is False
    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3


# --------------------------------------------------------------------------- #
# before_model_call veto
# --------------------------------------------------------------------------- #


async def test_before_model_call_block_vetoes_run(tmp_path: Path) -> None:
    async def veto(event: str, **payload: object) -> HookOutcome:
        return HookOutcome.blocked("no model calls")

    session = _session(tmp_path)
    provider = FakeProvider.from_text("never reached")
    hooks = HookManager()
    hooks.on("before_model_call", veto)

    result = await run_turn(session, provider, ToolRegistry(), hooks)

    assert result.stopped is True
    assert result.stop_reason == "model_call_blocked"
    assert result.last_message is None
    assert result.iterations == 0
    # Nothing was appended and the provider was never streamed.
    assert session.materialize_messages() == []
    assert provider.calls == []


# --------------------------------------------------------------------------- #
# usage propagation
# --------------------------------------------------------------------------- #


async def test_usage_propagates_to_transcript(tmp_path: Path) -> None:
    usage = {"input_tokens": 12, "output_tokens": 3, "cache_read": 0, "cache_write": 0}
    session = _session(tmp_path)
    provider = FakeProvider.from_text("hi", usage=usage)
    hooks = HookManager()

    await run_turn(session, provider, ToolRegistry(), hooks)

    # The terminal-chunk usage rode through to the persisted transcript record.
    records = session.transcript.read_records()
    assert records[-1]["usage"] == usage


# --------------------------------------------------------------------------- #
# hook payload visibility (after_model_call observation)
# --------------------------------------------------------------------------- #


async def test_after_model_call_hook_observes_message_and_usage(
    tmp_path: Path,
) -> None:
    usage = {"input_tokens": 1, "output_tokens": 1, "cache_read": 0, "cache_write": 0}
    observed: list[dict] = []

    async def observe(event: str, **payload: object) -> None:
        observed.append(dict(payload))

    session = _session(tmp_path)
    provider = FakeProvider.from_text("hi", usage=usage)
    hooks = HookManager()
    hooks.on("after_model_call", observe)

    await run_turn(session, provider, ToolRegistry(), hooks)

    assert len(observed) == 1
    assert observed[0]["usage"] == usage
    message = observed[0]["message"]
    assert isinstance(message, ChatMessage)
    assert message.content == "hi"


# --------------------------------------------------------------------------- #
# dispatch returns ordered results for mixed valid/invalid calls
# --------------------------------------------------------------------------- #


async def test_dispatch_preserves_order_with_mixed_calls(tmp_path: Path) -> None:
    @tool(name="ok")
    async def ok(text: str) -> ToolResult:
        """A working tool."""
        return ToolResult.text(text)

    calls = [
        ToolCall(id="c1", name="ok", args={"text": "first"}),
        ToolCall(id="c2", name="missing", args={}),
        ToolCall(id="c3", name="bad", parse_error="could not parse arguments: junk"),
    ]
    results = await dispatch_tool_calls(calls, _registry(ok), HookManager(), _session(tmp_path))

    assert [r.content for r in results] == [
        "first",
        "unknown tool: missing",
        "could not parse arguments: junk",
    ]
    assert results[0].is_error is False
    assert results[1].is_error is True
    assert results[2].is_error is True


# --------------------------------------------------------------------------- #
# SLOC budget guard
# --------------------------------------------------------------------------- #


def test_loop_under_300_sloc() -> None:
    """The loop is the centerpiece but must stay tight (≤300 SLOC)."""
    source = Path(__file__).resolve().parent.parent / "genie" / "loop.py"
    assert len(source.read_text(encoding="utf-8").splitlines()) <= 300
