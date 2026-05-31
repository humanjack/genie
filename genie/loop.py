"""The ReAct agent loop: the centerpiece that drives a conversation to completion (SPEC §4, §5).

This module owns the *Reason → Act → Observe* cycle. It holds no policy and no
provider knowledge; it composes the four merged seams instead:

- the :class:`~genie.providers.base.ProviderClient` it streams a turn from,
- the :class:`~genie.tools.registry.ToolRegistry` it dispatches tool calls
  through,
- the :class:`~genie.hooks.manager.HookManager` it announces every lifecycle
  event to (the single chokepoint of SPEC §7), and
- the :class:`~genie.session.session.Session` it materializes history from and
  appends each new message to.

Because every dependency is reached through its abstract contract, the loop is
unaware of who is on the other end — the same code runs against a real SDK
adapter and the scripted ``FakeProvider`` with zero edits.

The loop is intentionally tolerant of model misbehaviour: malformed tool
arguments, hallucinated tool names, and hook denials are each turned into a
tool-result the model reads and reacts to, never an exception that aborts the
run. The only hard stops are a blocked ``before_model_call`` (the run is vetoed
before it begins) and the ``max_iterations`` safety budget.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field

from genie.hooks.manager import BlockedError, HookManager
from genie.providers.base import ChatMessage, ProviderClient
from genie.session.session import Session
from genie.tools.registry import ToolRegistry
from genie.tools.result import ToolResult


@dataclass
class ToolCall:
    """A single tool invocation requested by an assistant turn.

    Attributes:
        id: The provider-assigned id this call's result must reference.
        name: The tool name the model asked to run.
        args: The parsed arguments dict; ``{}`` when arguments were empty or
            could not be parsed (see :attr:`parse_error`).
        parse_error: A human-readable message when the streamed arguments were
            not valid JSON, else ``None``. The loop feeds this back to the model
            as a tool error instead of raising, so the model can correct itself.
    """

    id: str
    name: str
    args: dict = field(default_factory=dict)
    parse_error: str | None = None


@dataclass
class TurnResult:
    """The outcome of one :func:`run_turn` invocation.

    Attributes:
        stopped: ``True`` when the loop ended because the model itself stopped
            (or was vetoed before starting); ``False`` when it was cut off by
            the ``max_iterations`` budget with work still pending.
        last_message: The final assistant message produced, or ``None`` when the
            run never got a model turn (a blocked ``before_model_call``).
        iterations: How many model turns the loop ran.
        stop_reason: Why the loop ended — one of ``"model_stopped"``,
            ``"max_iterations"``, or ``"model_call_blocked"``.
    """

    stopped: bool
    last_message: ChatMessage | None
    iterations: int
    stop_reason: str


async def _collect_assistant_turn(
    provider: ProviderClient,
    messages: list[ChatMessage],
    tools: list[dict],
    *,
    system: str | None,
    max_tokens: int,
    on_text_delta: Callable[[str], None] | None,
) -> tuple[ChatMessage, list[ToolCall], dict | None]:
    """Consume one streamed turn and reassemble it into an assistant message.

    Text fragments are concatenated (and forwarded to ``on_text_delta`` as they
    arrive, for live REPL display). Tool-call fragments are accumulated by their
    integer slot ``index``: ``id`` and ``name`` are captured from the slot's
    first fragment and each ``arguments_delta`` string is appended to that
    slot's running argument buffer — exactly the index-addressed shape
    :class:`~genie.providers.base.ChatChunk` documents. ``usage`` rides the
    terminal chunk.

    Once the stream ends, each slot's argument buffer is parsed with
    :func:`json.loads` (empty/whitespace → ``{}``). A
    :class:`json.JSONDecodeError` is *not* raised: the call keeps ``args={}`` and
    records a :attr:`ToolCall.parse_error` so the model can see and fix its
    malformed arguments. Slots are emitted in first-seen order.

    Returns:
        A ``(assistant_message, tool_calls, usage)`` triple. The assistant
        message carries ``tool_calls`` as ``{"id", "name", "arguments"}`` dicts
        (or ``None`` when the turn made no calls).
    """
    text_parts: list[str] = []
    # Ordered by first appearance of each slot index.
    slots: dict[int, dict] = {}
    usage: dict | None = None

    async for chunk in provider.stream(messages, tools, max_tokens=max_tokens, system=system):
        if chunk.delta_text:
            text_parts.append(chunk.delta_text)
            if on_text_delta is not None:
                on_text_delta(chunk.delta_text)
        if chunk.tool_call_delta is not None:
            _accumulate_tool_delta(slots, chunk.tool_call_delta)
        if chunk.usage is not None:
            usage = chunk.usage

    tool_calls = [_finalize_tool_call(slot) for slot in slots.values()]
    text = "".join(text_parts)
    message = ChatMessage(
        role="assistant",
        content=text,
        tool_calls=[{"id": c.id, "name": c.name, "arguments": c.args} for c in tool_calls] or None,
    )
    return message, tool_calls, usage


def _accumulate_tool_delta(slots: dict[int, dict], delta: dict) -> None:
    """Fold one ``tool_call_delta`` fragment into its slot's running buffer."""
    index = delta["index"]
    slot = slots.setdefault(index, {"id": None, "name": None, "args": ""})
    if delta.get("id") is not None:
        slot["id"] = delta["id"]
    if delta.get("name") is not None:
        slot["name"] = delta["name"]
    if delta.get("arguments_delta") is not None:
        slot["args"] += delta["arguments_delta"]


def _finalize_tool_call(slot: dict) -> ToolCall:
    """Parse one accumulated slot into a :class:`ToolCall`, capturing parse errors."""
    raw = slot["args"]
    if not raw.strip():
        return ToolCall(id=slot["id"], name=slot["name"], args={})
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ToolCall(
            id=slot["id"],
            name=slot["name"],
            args={},
            parse_error=f"could not parse arguments: {raw}",
        )
    return ToolCall(id=slot["id"], name=slot["name"], args=parsed)


async def dispatch_tool_calls(
    calls: list[ToolCall],
    registry: ToolRegistry,
    hooks: HookManager,
    session: Session,
) -> list[ToolResult]:
    """Run a batch of tool calls and return one :class:`ToolResult` per call, in order.

    Concurrency follows the Pi rule (SPEC §5.3): if *any* call targets a
    ``sequential`` tool, the whole batch runs serially (unknown tools are not
    sequential); otherwise the calls run concurrently via :func:`asyncio.gather`.
    Either way the results preserve the order of ``calls`` so each can be paired
    with the assistant's request.

    Each call is resolved by :func:`_call_one`, which turns a parse error, an
    unknown tool, or a hook veto into an error result rather than an exception —
    so a single bad call never aborts the batch or the loop.
    """

    async def _call_one(call: ToolCall) -> ToolResult:
        if call.parse_error is not None:
            return ToolResult.error(call.parse_error)
        if call.name not in registry:
            return ToolResult.error(f"unknown tool: {call.name}")
        try:
            await hooks.run_or_raise("before_tool_call", call=call, session=session)
        except BlockedError as exc:
            # A hook denied the call (e.g. approval/policy). Feed the denial back
            # to the model as an error result so the session continues (US-3).
            return ToolResult.error(f"blocked: {exc}")
        result = await registry.call(call.name, call.args)
        await hooks.run("after_tool_call", call=call, result=result, session=session)
        return result

    if any(call.name in registry and registry.get(call.name).sequential for call in calls):
        return [await _call_one(call) for call in calls]
    return list(await asyncio.gather(*(_call_one(call) for call in calls)))


async def run_turn(
    session: Session,
    provider: ProviderClient,
    registry: ToolRegistry,
    hooks: HookManager,
    *,
    system: str | None = None,
    max_tokens: int = 4096,
    max_iterations: int = 50,
    on_text_delta: Callable[[str], None] | None = None,
) -> TurnResult:
    """Drive the ReAct loop until the model stops or the iteration budget is hit.

    Each iteration: announce ``before_model_call`` (a veto aborts the run with
    ``stop_reason="model_call_blocked"``), stream and assemble the assistant turn
    via :func:`_collect_assistant_turn`, persist it, then announce
    ``after_model_call``. If the turn requested no tools the loop stops with
    ``stop_reason="model_stopped"``. Otherwise the requested calls are dispatched
    through :func:`dispatch_tool_calls`, each result is appended as a
    ``role="tool"`` message keyed to its call id, and the loop iterates.

    The ``max_iterations`` guard is a Phase-1 stopgap: SPEC §4 makes the turn
    budget an ``iteration_budget`` hook in Phase 2, at which point this counter
    is removed in favour of the hook owning the policy. Until then it bounds a
    runaway tool-calling loop, returning ``stopped=False`` with
    ``stop_reason="max_iterations"`` when the budget is exhausted with work still
    pending.

    Args:
        session: The conversation state; history is read from
            :meth:`~genie.session.session.Session.materialize_messages` and every
            new message is appended back.
        provider: The model client to stream each turn from.
        registry: The tools the model may call, and the dispatcher for them.
        hooks: The chokepoint every lifecycle event is announced through.
        system: Optional system prompt passed to the provider each turn.
        max_tokens: Per-turn generation cap forwarded to the provider.
        max_iterations: Maximum model turns before the safety budget stops the
            loop (Phase-1 stopgap; becomes a hook in Phase 2).
        on_text_delta: Optional callback invoked with each streamed text
            fragment, for live display.

    Returns:
        A :class:`TurnResult` describing how and why the loop ended.
    """
    iterations = 0
    while True:
        messages = session.materialize_messages()
        tools = registry.specs_for(provider.name)

        try:
            await hooks.run_or_raise("before_model_call", session=session, messages=messages)
        except BlockedError:
            return TurnResult(
                stopped=True,
                last_message=None,
                iterations=iterations,
                stop_reason="model_call_blocked",
            )

        assistant_msg, tool_calls, usage = await _collect_assistant_turn(
            provider,
            messages,
            tools,
            system=system,
            max_tokens=max_tokens,
            on_text_delta=on_text_delta,
        )
        session.append(assistant_msg, usage=usage)
        await hooks.run("after_model_call", session=session, message=assistant_msg, usage=usage)

        if not tool_calls:
            return TurnResult(
                stopped=True,
                last_message=assistant_msg,
                iterations=iterations + 1,
                stop_reason="model_stopped",
            )

        results = await dispatch_tool_calls(tool_calls, registry, hooks, session)
        for call, result in zip(tool_calls, results, strict=True):
            session.append(ChatMessage(role="tool", tool_call_id=call.id, content=result.content))

        iterations += 1
        if iterations >= max_iterations:
            return TurnResult(
                stopped=False,
                last_message=assistant_msg,
                iterations=iterations,
                stop_reason="max_iterations",
            )
