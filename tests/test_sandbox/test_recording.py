"""Tests for RecordingBackend — the proof the sandbox abstraction holds."""

from __future__ import annotations

import pytest

from genie.sandbox import (
    ExecResult,
    RecordingBackend,
    SandboxBackend,
)


def test_is_sandbox_backend() -> None:
    assert isinstance(RecordingBackend(), SandboxBackend)


async def test_replays_list_in_order() -> None:
    backend = RecordingBackend(
        results=[
            ExecResult(0, "first", ""),
            ExecResult(1, "", "second"),
        ]
    )
    r1 = await backend.exec("cmd-a")
    r2 = await backend.exec("cmd-b")
    assert r1.stdout == "first"
    assert r2.returncode == 1
    assert r2.stderr == "second"


async def test_records_every_call() -> None:
    backend = RecordingBackend(results=[ExecResult(0, "ok", "")])
    await backend.exec("echo hi", cwd="/work", env={"FOO": "bar"}, timeout=12.0)
    assert backend.calls == [
        {
            "command": "echo hi",
            "cwd": "/work",
            "env": {"FOO": "bar"},
            "timeout": 12.0,
        }
    ]


async def test_records_defaults_when_omitted() -> None:
    backend = RecordingBackend(results=[ExecResult(0, "ok", "")])
    await backend.exec("ls")
    call = backend.calls[0]
    assert call["cwd"] is None
    assert call["env"] is None
    assert call["timeout"] == 30.0


async def test_list_overdrive_raises() -> None:
    backend = RecordingBackend(results=[ExecResult(0, "only", "")])
    await backend.exec("once")
    with pytest.raises(IndexError, match="exhausted"):
        await backend.exec("twice")


async def test_empty_script_overdrives_immediately() -> None:
    backend = RecordingBackend()
    with pytest.raises(IndexError, match="exhausted"):
        await backend.exec("anything")
    # The over-driving call is still recorded for diagnostics.
    assert backend.calls[0]["command"] == "anything"


async def test_dict_mode_keyed_by_command() -> None:
    backend = RecordingBackend(
        results={
            "pytest": ExecResult(0, "passed", ""),
            "ruff check": ExecResult(1, "", "lint error"),
        }
    )
    # Order-independent: a later key can be requested first.
    lint = await backend.exec("ruff check")
    tests = await backend.exec("pytest")
    assert lint.returncode == 1
    assert tests.stdout == "passed"


async def test_dict_mode_unknown_command_raises() -> None:
    backend = RecordingBackend(results={"pytest": ExecResult(0, "passed", "")})
    with pytest.raises(KeyError, match="no scripted result"):
        await backend.exec("rm -rf /")


async def test_dict_mode_repeatable() -> None:
    backend = RecordingBackend(results={"date": ExecResult(0, "today", "")})
    first = await backend.exec("date")
    second = await backend.exec("date")
    assert first.stdout == second.stdout == "today"
    assert len(backend.calls) == 2
