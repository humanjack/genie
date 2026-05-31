"""Tests for the coding-agent wiring and the REPL core (genie.agent).

All offline: the loop is driven by a scripted FakeProvider and input by a
scripted ``read_input`` callable, so no terminal or network is involved.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path

from genie.agent import (
    ToolCallDisplay,
    build_registry,
    load_system_prompt,
    run_code_session,
)
from genie.hooks.manager import HookManager
from genie.loop import ToolCall
from genie.providers.base import ChatChunk, ProviderClient
from genie.providers.fake import FakeProvider
from genie.session.session import Session
from genie.tools.result import ToolResult


def _scripted_input(*lines: str | None) -> Callable[[], str | None]:
    """A read_input that yields each arg in turn, then None forever."""
    it = iter(lines)

    def read() -> str | None:
        return next(it, None)

    return read


def _session(tmp_path: Path) -> Session:
    return Session.create(tmp_path / ".genie", id="t", model="fake:1", working_dir=str(tmp_path))


# --- build_registry ----------------------------------------------------------


def test_build_registry_registers_the_four_tools(tmp_path: Path) -> None:
    registry = build_registry(tmp_path)
    assert set(registry.names()) == {"read_file", "write_file", "edit_file", "bash"}


def test_build_registry_marks_dangerous_tools(tmp_path: Path) -> None:
    registry = build_registry(tmp_path)
    assert registry.get("bash").dangerous is True
    assert registry.get("write_file").dangerous is True
    assert registry.get("read_file").dangerous is False


# --- load_system_prompt ------------------------------------------------------


def test_load_system_prompt_returns_nonempty_text() -> None:
    prompt = load_system_prompt()
    assert isinstance(prompt, str)
    assert prompt.strip()


# --- ToolCallDisplay ---------------------------------------------------------


async def test_display_hook_renders_before_and_after() -> None:
    lines: list[str] = []
    hook = ToolCallDisplay(write=lines.append)
    call = ToolCall(id="c1", name="read_file", args={"path": "a.py"})

    await hook("before_tool_call", call=call)
    await hook("after_tool_call", call=call, result=ToolResult.text("contents"))

    assert any(line.startswith("→ read_file(") for line in lines)
    assert any("✓ read_file" in line for line in lines)


async def test_display_hook_marks_error_result() -> None:
    lines: list[str] = []
    hook = ToolCallDisplay(write=lines.append)
    call = ToolCall(id="c1", name="bash", args={"command": "false"})

    await hook("after_tool_call", call=call, result=ToolResult.error("boom"))

    assert any(line.startswith("✗ bash") for line in lines)


async def test_display_hook_never_blocks() -> None:
    hook = ToolCallDisplay(write=lambda _: None)
    call = ToolCall(id="c1", name="read_file", args={})
    assert await hook("before_tool_call", call=call) is None


# --- run_code_session --------------------------------------------------------


async def test_run_code_session_one_text_turn(tmp_path: Path, capsys) -> None:
    session = _session(tmp_path)
    provider = FakeProvider.from_text("hi there")
    registry = build_registry(tmp_path)

    await run_code_session(
        provider,
        registry,
        HookManager(),
        session,
        system="s",
        read_input=_scripted_input("hello", None),
    )

    out = capsys.readouterr().out
    assert "hi there" in out
    roles = [m.role for m in session.materialize_messages()]
    assert roles == ["user", "assistant"]


async def test_run_code_session_executes_tool_then_stops(tmp_path: Path, capsys) -> None:
    (tmp_path / "f.txt").write_text("file body")
    session = _session(tmp_path)
    registry = build_registry(tmp_path)
    hooks = HookManager()
    hooks.register(ToolCallDisplay())
    # Turn 1: model calls read_file; turn 2: model replies and stops.
    provider = FakeProvider(
        [
            FakeProvider.with_tool_call("read_file", {"path": "f.txt"})._turns[0],
            FakeProvider.from_text("the file says: file body")._turns[0],
        ]
    )

    await run_code_session(
        provider, registry, hooks, session, system="s", read_input=_scripted_input("read it", None)
    )

    out = capsys.readouterr().out
    assert "→ read_file(" in out  # display hook rendered the call
    roles = [m.role for m in session.materialize_messages()]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_msg = session.materialize_messages()[2]
    assert "file body" in str(tool_msg.content)


async def test_run_code_session_writes_a_file_end_to_end(tmp_path: Path) -> None:
    """The REPL drives a real write_file tool through the loop onto disk."""
    session = _session(tmp_path)
    registry = build_registry(tmp_path)
    provider = FakeProvider(
        [
            FakeProvider.with_tool_call(
                "write_file", {"path": "hello.py", "content": "print('hi')"}
            )._turns[0],
            FakeProvider.from_text("done")._turns[0],
        ]
    )

    await run_code_session(
        provider,
        registry,
        HookManager(),
        session,
        system="s",
        read_input=_scripted_input("make hello.py", None),
        write=lambda _: None,
    )

    created = tmp_path / "hello.py"
    assert created.exists()
    assert created.read_text() == "print('hi')"


async def test_run_code_session_eof_exits_without_turns(tmp_path: Path) -> None:
    session = _session(tmp_path)
    await run_code_session(
        FakeProvider(),  # empty script — never streamed because no input
        build_registry(tmp_path),
        HookManager(),
        session,
        system="s",
        read_input=_scripted_input(None),
    )
    assert session.materialize_messages() == []


async def test_run_code_session_exit_command(tmp_path: Path) -> None:
    session = _session(tmp_path)
    await run_code_session(
        FakeProvider(),
        build_registry(tmp_path),
        HookManager(),
        session,
        system="s",
        read_input=_scripted_input("/exit"),
    )
    assert session.materialize_messages() == []


async def test_run_code_session_ctrl_c_during_turn_continues(tmp_path: Path) -> None:
    """Ctrl-C during a turn is caught and reported; the REPL does not crash (US headline)."""

    class _InterruptingProvider(ProviderClient):
        name = "fake"
        model = "fake-1"

        async def stream(self, messages, tools, **kwargs) -> AsyncIterator[ChatChunk]:
            raise KeyboardInterrupt
            yield ChatChunk()  # pragma: no cover - unreachable, makes this a generator

        def count_tokens(self, messages) -> int:  # pragma: no cover
            return 0

    notices: list[str] = []
    # One line triggers the interrupt mid-turn; EOF then ends the loop cleanly.
    await run_code_session(
        _InterruptingProvider(),
        build_registry(tmp_path),
        HookManager(),
        _session(tmp_path),
        system="s",
        read_input=_scripted_input("do something", None),
        write=notices.append,
    )

    assert any("interrupted" in n for n in notices)


async def test_run_code_session_skips_blank_lines(tmp_path: Path) -> None:
    session = _session(tmp_path)
    provider = FakeProvider.from_text("ok")
    await run_code_session(
        provider,
        build_registry(tmp_path),
        HookManager(),
        session,
        system="s",
        read_input=_scripted_input("   ", "go", None),
        write=lambda _: None,
    )
    # Only the non-blank "go" produced a turn.
    assert [m.role for m in session.materialize_messages()] == ["user", "assistant"]
