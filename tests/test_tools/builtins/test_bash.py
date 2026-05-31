"""Tests for the ``bash`` builtin: a thin, safe wrapper over the sandbox.

Most cases drive the deterministic :class:`RecordingBackend` so behavior is
asserted without a real process; one smoke test runs a real
:class:`LocalSubprocessBackend` to prove the wrapper composes with the live
backend. The load-bearing distinction under test: a nonzero command exit is
``is_error=False`` (returned to the model), while a sandbox exception becomes
:meth:`ToolResult.error`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from genie.sandbox import (
    ExecResult,
    LocalSubprocessBackend,
    RecordingBackend,
    SandboxBackend,
    SandboxError,
)
from genie.tools.base import Tool
from genie.tools.builtins.bash import make_bash
from genie.tools.result import ToolResult


class _RaisingBackend(SandboxBackend):
    """A backend whose :meth:`exec` always raises :class:`SandboxError`."""

    async def exec(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> ExecResult:
        raise SandboxError("cwd /etc escapes sandbox root")


def test_factory_returns_dangerous_shell_tool() -> None:
    bash = make_bash(RecordingBackend())
    assert isinstance(bash, Tool)
    assert bash.name == "bash"
    assert bash.dangerous is True
    assert "shell" in bash.tags


def test_input_schema_requires_command() -> None:
    schema = make_bash(RecordingBackend()).input_schema
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"command"}
    assert schema["properties"]["command"]["type"] == "string"
    assert schema["required"] == ["command"]


async def test_success_returns_output_and_records_command() -> None:
    backend = RecordingBackend(results=[ExecResult(0, "hello\n", "")])
    bash = make_bash(backend)

    result = await bash.handler(command="echo hello")

    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert "hello" in result.content
    assert backend.calls[0]["command"] == "echo hello"


async def test_success_includes_stderr_when_present() -> None:
    backend = RecordingBackend(results=[ExecResult(0, "out\n", "warn\n")])
    bash = make_bash(backend)

    result = await bash.handler(command="noisy")

    assert result.is_error is False
    assert "out" in result.content
    assert "warn" in result.content


async def test_nonzero_exit_is_not_a_tool_error() -> None:
    backend = RecordingBackend(results=[ExecResult(3, "", "boom")])
    bash = make_bash(backend)

    result = await bash.handler(command="false")

    # A failed command is feedback for the model, not a retried tool error.
    assert result.is_error is False
    assert "exit code: 3" in result.content
    assert "boom" in result.content


async def test_timed_out_is_noted_and_not_a_tool_error() -> None:
    backend = RecordingBackend(results=[ExecResult(124, "", "", timed_out=True)])
    bash = make_bash(backend, timeout=30.0)

    result = await bash.handler(command="sleep 999")

    assert result.is_error is False
    assert "timed out after 30.0s" in result.content


async def test_truncated_output_is_flagged() -> None:
    backend = RecordingBackend(results=[ExecResult(0, "lots of data", "", truncated=True)])
    bash = make_bash(backend)

    result = await bash.handler(command="cat big")

    assert result.is_error is False
    assert "truncated" in result.content
    assert "lots of data" in result.content


async def test_sandbox_error_becomes_tool_error() -> None:
    bash = make_bash(_RaisingBackend())

    result = await bash.handler(command="cd /etc")

    # Only a sandbox failure is a tool error; the traceback never escapes.
    assert result.is_error is True
    assert "sandbox error" in result.content
    assert "escapes sandbox root" in result.content


async def test_timeout_is_forwarded_to_sandbox() -> None:
    backend = RecordingBackend(results=[ExecResult(0, "ok", "")])
    bash = make_bash(backend, timeout=5)

    await bash.handler(command="echo ok")

    assert backend.calls[0]["timeout"] == 5


async def test_real_local_subprocess_echo(tmp_path: Path) -> None:
    bash = make_bash(LocalSubprocessBackend(tmp_path))

    result = await bash.handler(command="echo hi")

    assert result.is_error is False
    assert "hi" in result.content
