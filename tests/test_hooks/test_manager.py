"""Tests for the hook chokepoint (SPEC §7.1, §7.2).

These cover the contract the loop and Phase-2 hooks rely on: event filtering,
registration-order dispatch, the first-block-wins cascade (via both ``run`` and
``run_or_raise``), payload-mutation merge visibility, the empty-chain and
unknown-event edges, and the documented "a raising hook propagates" policy.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from genie.hooks.manager import (
    EVENTS,
    BlockedError,
    Hook,
    HookManager,
    HookOutcome,
)

OutcomeFn = Callable[..., Awaitable[HookOutcome | None]]


class RecordingHook:
    """A configurable :class:`Hook` that records each call and returns a scripted outcome."""

    def __init__(
        self,
        name: str,
        events: list[str],
        log: list[str],
        result: HookOutcome | None = None,
        *,
        on_call: OutcomeFn | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self._log = log
        self._result = result
        self._on_call = on_call
        self.calls: list[dict] = []

    async def __call__(self, event: str, **payload: object) -> HookOutcome | None:
        self._log.append(self.name)
        self.calls.append(dict(payload))
        if self._on_call is not None:
            return await self._on_call(event, **payload)
        return self._result


def test_recording_hook_satisfies_protocol() -> None:
    assert isinstance(RecordingHook("h", ["session_start"], []), Hook)


def test_events_are_the_eight_spec_events() -> None:
    assert set(EVENTS) == {
        "session_start",
        "session_end",
        "before_model_call",
        "after_model_call",
        "before_tool_call",
        "after_tool_call",
        "model_error",
        "tool_error",
    }
    assert isinstance(EVENTS, frozenset)


async def test_hook_fires_only_for_declared_events() -> None:
    log: list[str] = []
    mgr = HookManager()
    mgr.register(RecordingHook("only_tool", ["before_tool_call"], log))

    await mgr.run("before_model_call", model="m")
    assert log == []

    await mgr.run("before_tool_call", call={})
    assert log == ["only_tool"]


async def test_hooks_run_in_registration_order() -> None:
    log: list[str] = []
    mgr = HookManager()
    mgr.register(RecordingHook("first", ["session_start"], log))
    mgr.register(RecordingHook("second", ["session_start"], log))
    mgr.register(RecordingHook("third", ["session_start"], log))

    await mgr.run("session_start", session=object())

    assert log == ["first", "second", "third"]


async def test_first_block_stops_cascade_and_run_returns_block() -> None:
    log: list[str] = []
    mgr = HookManager()
    mgr.register(RecordingHook("pass", ["before_tool_call"], log))
    mgr.register(RecordingHook("veto", ["before_tool_call"], log, HookOutcome.blocked("nope")))
    later = RecordingHook("later", ["before_tool_call"], log)
    mgr.register(later)

    outcome = await mgr.run("before_tool_call", call={})

    assert outcome.block is True
    assert outcome.block_reason == "nope"
    assert log == ["pass", "veto"]
    assert later.calls == []


async def test_run_or_raise_raises_blocked_error_with_reason() -> None:
    mgr = HookManager()
    mgr.register(RecordingHook("veto", ["before_model_call"], [], HookOutcome.blocked("denied")))

    with pytest.raises(BlockedError, match="denied") as exc:
        await mgr.run_or_raise("before_model_call", model="m")

    assert exc.value.reason == "denied"


async def test_run_or_raise_returns_outcome_when_not_blocked() -> None:
    mgr = HookManager()
    mgr.register(
        RecordingHook(
            "mutate",
            ["before_tool_call"],
            [],
            HookOutcome.proceed({"approved": True}),
        )
    )

    outcome = await mgr.run_or_raise("before_tool_call", call={})

    assert outcome.block is False
    assert outcome.mutated_payload == {"approved": True}


async def test_blocked_error_default_message_when_reason_none() -> None:
    assert str(BlockedError(None)) == "blocked by hook"


async def test_mutated_payload_visible_to_later_hook_and_in_outcome() -> None:
    log: list[str] = []
    mgr = HookManager()
    mgr.register(
        RecordingHook(
            "mutator",
            ["before_model_call"],
            log,
            HookOutcome.proceed({"model": "swapped", "added": 1}),
        )
    )
    observer = RecordingHook("observer", ["before_model_call"], log)
    mgr.register(observer)

    outcome = await mgr.run("before_model_call", model="original")

    assert observer.calls == [{"model": "swapped", "added": 1}]
    assert outcome.block is False
    assert outcome.mutated_payload == {"model": "swapped", "added": 1}


async def test_later_mutation_overrides_earlier_on_same_key() -> None:
    mgr = HookManager()
    mgr.register(RecordingHook("a", ["after_tool_call"], [], HookOutcome.proceed({"k": 1})))
    mgr.register(RecordingHook("b", ["after_tool_call"], [], HookOutcome.proceed({"k": 2})))

    outcome = await mgr.run("after_tool_call", call={}, result="r")

    assert outcome.mutated_payload == {"k": 2}


async def test_hook_returning_none_proceeds() -> None:
    mgr = HookManager()
    mgr.register(RecordingHook("noop", ["session_end"], [], None))

    outcome = await mgr.run("session_end", session=object(), reason="done")

    assert outcome.block is False
    assert outcome.mutated_payload is None


async def test_no_registered_hooks_proceeds() -> None:
    mgr = HookManager()

    outcome = await mgr.run("before_tool_call", call={})

    assert outcome.block is False
    assert outcome.mutated_payload is None


async def test_unknown_event_raises_value_error() -> None:
    mgr = HookManager()

    with pytest.raises(ValueError, match="unknown hook event"):
        await mgr.run("not_an_event")


async def test_run_or_raise_unknown_event_raises_value_error() -> None:
    mgr = HookManager()

    with pytest.raises(ValueError, match="unknown hook event"):
        await mgr.run_or_raise("not_an_event")


async def test_raising_hook_propagates() -> None:
    """Documented policy: a hook that raises is a bug, not a block — it propagates."""

    async def boom(event: str, **payload: object) -> HookOutcome | None:
        raise RuntimeError("hook is broken")

    mgr = HookManager()
    mgr.register(RecordingHook("boom", ["before_tool_call"], [], on_call=boom))

    with pytest.raises(RuntimeError, match="hook is broken"):
        await mgr.run("before_tool_call", call={})


async def test_async_hooks_are_awaited() -> None:
    awaited: list[str] = []

    async def slow(event: str, **payload: object) -> HookOutcome | None:
        awaited.append("ran")
        return HookOutcome.proceed()

    mgr = HookManager()
    mgr.register(RecordingHook("slow", ["session_start"], [], on_call=slow))

    await mgr.run("session_start", session=object())

    assert awaited == ["ran"]


async def test_on_registers_callable_for_single_event() -> None:
    seen: list[str] = []

    async def my_hook(event: str, **payload: object) -> HookOutcome | None:
        seen.append(event)
        return None

    mgr = HookManager()
    mgr.on("before_tool_call", my_hook)

    assert mgr.hooks[0].name == "my_hook"
    assert mgr.hooks[0].events == ["before_tool_call"]

    await mgr.run("before_tool_call", call={})
    await mgr.run("session_start", session=object())

    assert seen == ["before_tool_call"]


async def test_on_can_block() -> None:
    async def deny(event: str, **payload: object) -> HookOutcome | None:
        return HookOutcome.blocked("via on")

    mgr = HookManager()
    mgr.on("before_model_call", deny)

    with pytest.raises(BlockedError, match="via on"):
        await mgr.run_or_raise("before_model_call", model="m")


def test_on_rejects_unknown_event() -> None:
    mgr = HookManager()

    async def fn(event: str, **payload: object) -> HookOutcome | None:
        return None

    with pytest.raises(ValueError, match="unknown hook event"):
        mgr.on("not_an_event", fn)


def test_on_name_falls_back_to_repr_for_anonymous_callable() -> None:
    mgr = HookManager()

    class Callable:
        async def __call__(self, event: str, **payload: object) -> HookOutcome | None:
            return None

    instance = Callable()
    mgr.on("session_start", instance)

    assert mgr.hooks[0].name == repr(instance)
