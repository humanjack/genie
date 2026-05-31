"""Hook system: the lifecycle-event chokepoint every action is gated by (SPEC §7)."""

from __future__ import annotations

from genie.hooks.manager import (
    EVENTS,
    BlockedError,
    Hook,
    HookManager,
    HookOutcome,
)

__all__ = [
    "EVENTS",
    "BlockedError",
    "Hook",
    "HookManager",
    "HookOutcome",
]
