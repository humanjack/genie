"""Phase 1 end-to-end user-story tests — driving the WHOLE stack as a user would.

Unlike the per-component unit tests, these seed a real repo on disk, build the
real tool registry (real sandbox, real file tools), and drive the real
:func:`~genie.loop.run_turn` with a scripted :class:`FakeProvider` standing in
for the model's decisions. The assertions are about *real outcomes*: files
actually edited, real subprocesses actually run, the policy hook actually
denying a dangerous command, and a transcript that actually replays.

The scenarios map to docs/DEVELOPMENT_PLAN.md (S1-S5). S1-S3 + S5 are
deterministic and run on every CI build; S4 hits the live Anthropic API and is
skipped unless ``RUN_LIVE_API`` is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from genie.agent import build_registry
from genie.hooks.manager import HookManager, HookOutcome
from genie.loop import run_turn
from genie.providers.base import ChatChunk, ChatMessage
from genie.providers.fake import FakeProvider
from genie.session.session import Session

# --- helpers: build deterministic provider turns in the real wire shape ------


def _tool_turn(name: str, args: dict, *, call_id: str = "call_1") -> list[ChatChunk]:
    """One scripted assistant turn that streams a single tool call then stops.

    Uses the real index-addressed ``tool_call_delta`` contract so the loop's
    reassembly path is exercised exactly as a live provider would drive it.
    """
    return [
        ChatChunk(
            tool_call_delta={
                "index": 0,
                "id": call_id,
                "name": name,
                "arguments_delta": json.dumps(args),
            }
        ),
        ChatChunk(finish_reason="tool_calls"),
    ]


def _text_turn(text: str) -> list[ChatChunk]:
    """One scripted assistant turn that streams text and stops."""
    return [ChatChunk(delta_text=text), ChatChunk(finish_reason="stop")]


def _session(tmp_path: Path, *, prompt: str) -> Session:
    session = Session.create(
        tmp_path / ".sessions", id="e2e", model="fake:fake-1", working_dir=str(tmp_path)
    )
    session.append(ChatMessage(role="user", content=prompt))
    return session


def _tool_messages(session: Session) -> list[ChatMessage]:
    return [m for m in session.materialize_messages() if m.role == "tool"]


# --- S1: close a real bug end-to-end (the north-star) ------------------------


async def test_s1_fix_bug_read_edit_bash(tmp_path: Path) -> None:
    """US-1: read the buggy file, edit it, run the check, report — all for real."""
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b  # BUG\n")
    (tmp_path / "check.py").write_text(
        "from calc import add\nassert add(2, 3) == 5, 'FAIL'\nprint('TESTS PASS')\n"
    )

    registry = build_registry(tmp_path)
    session = _session(tmp_path, prompt="Fix the bug in add() and confirm the check passes.")
    provider = FakeProvider(
        [
            _tool_turn("read_file", {"path": "calc.py"}),
            _tool_turn(
                "edit_file",
                {"path": "calc.py", "old": "return a - b  # BUG", "new": "return a + b"},
            ),
            _tool_turn("bash", {"command": "python3 check.py"}),
            _text_turn("Fixed: add() now returns a + b; the check passes."),
        ]
    )

    result = await run_turn(session, provider, registry, HookManager())

    # The fix actually landed on disk.
    assert "return a + b" in (tmp_path / "calc.py").read_text()
    # All three tool calls really happened, in order.
    tool_msgs = _tool_messages(session)
    assert len(tool_msgs) == 3
    # The real subprocess observed the passing check.
    assert "TESTS PASS" in str(tool_msgs[2].content)
    assert result.stopped and result.stop_reason == "model_stopped"


# --- S2: create a file and run it --------------------------------------------


async def test_s2_create_file_and_run(tmp_path: Path) -> None:
    """US: 'create hello.py that prints hi' → write_file then bash prints hi."""
    registry = build_registry(tmp_path)
    session = _session(tmp_path, prompt="Create hello.py that prints hi, then run it.")
    provider = FakeProvider(
        [
            _tool_turn("write_file", {"path": "hello.py", "content": "print('hi')\n"}),
            _tool_turn("bash", {"command": "python3 hello.py"}),
            _text_turn("Created hello.py and ran it."),
        ]
    )

    result = await run_turn(session, provider, registry, HookManager())

    assert (tmp_path / "hello.py").read_text() == "print('hi')\n"
    assert "hi" in str(_tool_messages(session)[1].content)
    assert result.stop_reason == "model_stopped"


# --- S3: a dangerous command is denied by the hook, session continues (US-3) -


class _DenyDangerousBash:
    """A before_tool_call policy hook standing in for the Phase-2 approval hook.

    Blocks a ``bash`` call whose command matches a dangerous pattern. The loop
    feeds the denial back to the model as a tool result, so the session is never
    killed — exactly the US-3 fail-safe behaviour.
    """

    name = "deny_dangerous_bash"

    def __init__(self) -> None:
        self.events = ["before_tool_call"]

    async def __call__(self, event: str, **payload: object) -> HookOutcome | None:
        call = payload.get("call")
        command = getattr(call, "args", {}).get("command", "") if call else ""
        if getattr(call, "name", None) == "bash" and "rm -rf" in command:
            return HookOutcome.blocked("dangerous command blocked by policy")
        return None


async def test_s3_dangerous_command_denied_session_continues(tmp_path: Path) -> None:
    (tmp_path / "precious.txt").write_text("do not delete me")
    registry = build_registry(tmp_path)
    hooks = HookManager()
    hooks.register(_DenyDangerousBash())
    session = _session(tmp_path, prompt="Delete everything in this directory.")
    provider = FakeProvider(
        [
            _tool_turn("bash", {"command": "rm -rf ."}),
            _text_turn("Understood — I won't delete anything."),
        ]
    )

    result = await run_turn(session, provider, registry, hooks)

    # The destructive command never ran: the sentinel survives.
    assert (tmp_path / "precious.txt").read_text() == "do not delete me"
    # The denial was fed back to the model as a tool result.
    tool_msgs = _tool_messages(session)
    assert len(tool_msgs) == 1
    assert "blocked" in str(tool_msgs[0].content).lower()
    # The session continued and ended cleanly rather than crashing.
    assert result.stopped and result.stop_reason == "model_stopped"


# --- S5: a saved session resumes and replays identically ---------------------


async def test_s5_resume_replays_transcript(tmp_path: Path) -> None:
    registry = build_registry(tmp_path)
    session = _session(tmp_path, prompt="Create note.txt saying hello.")
    provider = FakeProvider(
        [
            _tool_turn("write_file", {"path": "note.txt", "content": "hello"}),
            _text_turn("Done."),
        ]
    )
    await run_turn(session, provider, registry, HookManager())
    original = session.materialize_messages()

    # Resume from disk into a fresh Session and assert byte-for-byte message parity.
    resumed = Session.resume(tmp_path / ".sessions", "e2e")
    assert resumed.materialize_messages() == original
    # And the resumed session knows its workspace + model.
    assert resumed.working_dir == str(tmp_path)
    assert resumed.model == "fake:fake-1"


# --- S4: live Anthropic end-to-end (gated) -----------------------------------


@pytest.mark.skipif(not os.getenv("RUN_LIVE_API"), reason="set RUN_LIVE_API for the live e2e")
async def test_s4_live_anthropic_fix_bug(tmp_path: Path) -> None:  # pragma: no cover - networked
    """The real model, real tools, real subprocess: close a trivial bug live."""
    from genie.agent import load_system_prompt
    from genie.providers.factory import provider_factory

    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "check.py").write_text(
        "from calc import add\nassert add(2, 3) == 5\nprint('TESTS PASS')\n"
    )
    registry = build_registry(tmp_path)
    session = _session(
        tmp_path,
        prompt="calc.py's add() is wrong. Fix it, then run `python3 check.py` to confirm.",
    )
    provider = provider_factory("anthropic:claude-haiku-4-5-20251001")

    result = await run_turn(
        session, provider, registry, HookManager(), system=load_system_prompt(), max_iterations=12
    )

    assert "return a + b" in (tmp_path / "calc.py").read_text()
    assert result.stopped
