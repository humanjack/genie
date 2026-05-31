"""Hook / middleware system: the single chokepoint every lifecycle event passes through (SPEC §7).

The agent loop never acts on the world directly. Before it calls the model,
calls a tool, or reaches a session boundary, it announces the event here and
lets registered :class:`Hook` objects observe it, mutate its payload, or veto
it outright. This is the mechanism behind the "secure by default" operating
principle: every action that touches disk, network, or shell is gated by a
hook, so policy lives in one place instead of being scattered through the loop.

This module is **pure mechanism**. The built-in policy hooks of SPEC §7.3
(``approval``, ``iteration_budget``, ``cost_ledger``, ``policy``) are Phase 2
and live elsewhere; here we only define the contract and the dispatcher they
plug into.

Key behaviours, all decided here so the loop and Phase-2 hooks can rely on them:

- **Event filtering.** A hook declares the events it cares about via
  :attr:`Hook.events`; :meth:`HookManager.run` only invokes a hook for an event
  in that list. Non-matching events skip it entirely.
- **Registration order is run order.** Hooks fire in the order they were
  registered. The first one to register for an event runs first.
- **Blocking cascade.** The first hook to return an outcome with ``block=True``
  short-circuits the cascade: later hooks for that event are **not** called and
  that blocking outcome is returned. :meth:`HookManager.run_or_raise` turns it
  into a :class:`BlockedError`.
- **Payload mutation merge.** A non-blocking hook may return
  ``mutated_payload`` to revise the payload. Those keys are merged
  (``dict.update`` semantics — last writer wins) into a running payload that
  later hooks for the same event see, and the accumulated mutation is carried on
  the final outcome.
- **A raising hook propagates.** A hook that raises (as opposed to cleanly
  blocking) is treated as a bug, not a policy decision: the exception is left to
  propagate. At a ``before_*`` chokepoint this fails closed — a broken policy
  hook stops the action rather than silently letting it through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

EVENTS: frozenset[str] = frozenset(
    {
        "session_start",
        "session_end",
        "before_model_call",
        "after_model_call",
        "before_tool_call",
        "after_tool_call",
        "model_error",
        "tool_error",
    }
)
"""The eight lifecycle events a hook may subscribe to (SPEC §7.1).

Intentionally compatible with Anthropic's Claude Code hook schema. The payload
each event carries is documented in SPEC §7.1; this module is agnostic to
payload shape and forwards it as keyword arguments unchanged.
"""


@dataclass
class HookOutcome:
    """The verdict a hook returns for one event (SPEC §7.2).

    A hook may also return ``None`` instead of an instance, which
    :meth:`HookManager.run` treats as an implicit :meth:`proceed`.

    Attributes:
        block: When ``True``, short-circuit the cascade — the loop must not
            perform the gated action (the model/tool call) and no later hook for
            this event runs.
        block_reason: Human-readable explanation for a block, surfaced to the
            model via :class:`BlockedError`; ``None`` when not blocking.
        mutated_payload: Keys to merge into the event payload so later hooks and
            the loop observe the revision; ``None`` when the hook leaves the
            payload untouched.
    """

    block: bool = False
    block_reason: str | None = None
    mutated_payload: dict | None = None

    @classmethod
    def proceed(cls, mutated_payload: dict | None = None) -> HookOutcome:
        """Return a non-blocking outcome, optionally carrying payload mutations."""
        return cls(block=False, mutated_payload=mutated_payload)

    @classmethod
    def blocked(cls, reason: str) -> HookOutcome:
        """Return a blocking outcome that stops the cascade with ``reason``."""
        return cls(block=True, block_reason=reason)


@runtime_checkable
class Hook(Protocol):
    """The contract a hook satisfies to plug into the chokepoint (SPEC §7.2).

    A hook is any object exposing a :attr:`name`, the :attr:`events` it
    subscribes to, and an async ``__call__``. Implementations may be classes or
    any object with these members; structural typing keeps the contract light.

    Attributes:
        name: Stable identifier, used in logs and diagnostics.
        events: The event names this hook should be invoked for; events outside
            this collection are skipped for this hook.
    """

    name: str
    events: list[str]

    async def __call__(self, event: str, **payload: object) -> HookOutcome | None:
        """Observe ``event`` and its ``payload``; return a verdict or ``None``.

        Returning ``None`` (or a non-blocking :class:`HookOutcome`) lets the
        action proceed. Returning :meth:`HookOutcome.blocked` vetoes it.
        """
        ...


class BlockedError(Exception):
    """Raised when a hook blocks an action, carrying the block reason (SPEC §7.2).

    :meth:`HookManager.run_or_raise` raises this so the loop can catch it at a
    ``before_*`` chokepoint and format it as a tool-result-equivalent for the
    model to read. The reason is available both as the exception message and as
    :attr:`reason`.
    """

    def __init__(self, reason: str | None) -> None:
        super().__init__(reason or "blocked by hook")
        self.reason = reason


@dataclass
class HookManager:
    """Registers hooks and dispatches lifecycle events to them (SPEC §7.2).

    The manager is the only object the loop talks to about hooks. It preserves
    registration order, filters by subscribed event, merges payload mutations,
    and enforces the first-block-wins cascade.

    Attributes:
        hooks: Registered hooks in registration (run) order.
    """

    hooks: list[Hook] = field(default_factory=list)

    def register(self, hook: Hook) -> None:
        """Append ``hook`` to the chain; its position fixes its run order."""
        self.hooks.append(hook)

    def on(self, event: str, fn: object) -> None:
        """Register a bare async callable ``fn`` for a single ``event``.

        Convenience for one-off hooks that do not warrant a class: ``fn`` is
        wrapped in an adapter exposing the :class:`Hook` shape (``name`` derived
        from ``fn``, ``events == [event]``). ``event`` must be a known event.
        """
        if event not in EVENTS:
            raise ValueError(f"unknown hook event: {event!r}")
        self.register(_CallableHook(event, fn))

    async def run(self, event: str, **payload: object) -> HookOutcome:
        """Dispatch ``event`` to every subscribed hook and return the outcome.

        Hooks fire in registration order. Each subscribed hook is awaited with
        ``event`` and the (possibly already-mutated) payload as keyword
        arguments. The first hook to return ``block=True`` stops the cascade and
        its outcome is returned immediately — later hooks do not run. Otherwise
        each non-blocking hook's ``mutated_payload`` is merged into the running
        payload (``dict.update`` — last writer wins) so subsequent hooks see the
        revision, and the accumulated mutation is returned on a non-blocking
        outcome (``None`` if nothing mutated).

        This method does not raise on a block; it returns the blocking outcome
        so callers that want to inspect rather than abort can. Use
        :meth:`run_or_raise` at ``before_*`` chokepoints to fail closed.

        Args:
            event: One of :data:`EVENTS`.
            **payload: The event payload (SPEC §7.1), forwarded unchanged.

        Returns:
            The first blocking outcome, or a non-blocking outcome carrying any
            merged ``mutated_payload``.

        Raises:
            ValueError: If ``event`` is not a known event.
        """
        if event not in EVENTS:
            raise ValueError(f"unknown hook event: {event!r}")

        running = dict(payload)
        merged: dict = {}
        for hook in self.hooks:
            if event not in hook.events:
                continue
            outcome = await hook(event, **running)
            if outcome is None:
                continue
            if outcome.block:
                return outcome
            if outcome.mutated_payload:
                running.update(outcome.mutated_payload)
                merged.update(outcome.mutated_payload)

        return HookOutcome.proceed(merged or None)

    async def run_or_raise(self, event: str, **payload: object) -> HookOutcome:
        """Run ``event`` and raise :class:`BlockedError` if a hook blocks it.

        The fail-closed variant the loop should use at the ``before_*``
        chokepoints (``before_model_call``, ``before_tool_call``): a block must
        prevent the action, so it is surfaced as an exception the loop catches
        and reports to the model. For observational events
        (``after_*``/``session_*``/``*_error``) prefer :meth:`run`, where a
        block has no action to stop.

        Returns:
            The non-blocking outcome (carrying any merged ``mutated_payload``).

        Raises:
            BlockedError: If a hook returns ``block=True``.
            ValueError: If ``event`` is not a known event.
        """
        outcome = await self.run(event, **payload)
        if outcome.block:
            raise BlockedError(outcome.block_reason)
        return outcome


class _CallableHook:
    """Adapts a bare async callable into the :class:`Hook` shape for :meth:`HookManager.on`."""

    def __init__(self, event: str, fn: object) -> None:
        self.fn = fn
        self.name: str = getattr(fn, "__name__", repr(fn))
        self.events: list[str] = [event]

    async def __call__(self, event: str, **payload: object) -> HookOutcome | None:
        """Delegate to the wrapped callable."""
        return await self.fn(event, **payload)  # type: ignore[operator]
