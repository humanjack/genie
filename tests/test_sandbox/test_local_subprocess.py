"""Tests for LocalSubprocessBackend — the v1 host runner (SPEC §6.1).

These exercise the real subprocess path (fast, local) and focus on the two
security guarantees — cwd confinement and env curation — plus timeout and
output caps.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from genie.sandbox import (
    ExecResult,
    LocalSubprocessBackend,
    SandboxBackend,
    SandboxError,
)


def test_is_sandbox_backend(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    assert isinstance(backend, SandboxBackend)


def test_root_is_resolved_absolute(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(str(tmp_path))
    assert backend.root.is_absolute()
    assert backend.root == tmp_path.resolve()


async def test_echo_hello(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    result = await backend.exec("echo hello")
    assert result.returncode == 0
    assert "hello" in result.stdout
    assert result.timed_out is False
    assert result.truncated is False


async def test_nonzero_exit(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    result = await backend.exec("exit 3")
    assert result.returncode == 3


async def test_stderr_captured_separately(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    result = await backend.exec("echo out; echo err 1>&2")
    assert "out" in result.stdout
    assert "err" in result.stderr
    assert "err" not in result.stdout
    # output property combines the two streams.
    assert "out" in result.output
    assert "err" in result.output


async def test_default_cwd_is_root(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    result = await backend.exec("pwd")
    assert result.stdout.strip() == str(tmp_path.resolve())


# --- cwd confinement (SECURITY) -------------------------------------------


async def test_cwd_parent_escape_rejected(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    with pytest.raises(SandboxError):
        await backend.exec("pwd", cwd=str(tmp_path / ".."))


async def test_cwd_absolute_etc_rejected(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    with pytest.raises(SandboxError):
        await backend.exec("pwd", cwd="/etc")


async def test_cwd_relative_dotdot_rejected(tmp_path: Path) -> None:
    sub = tmp_path / "work"
    sub.mkdir()
    backend = LocalSubprocessBackend(sub)
    with pytest.raises(SandboxError):
        await backend.exec("pwd", cwd="..")


async def test_cwd_legit_subdir_allowed(tmp_path: Path) -> None:
    sub = tmp_path / "pkg" / "src"
    sub.mkdir(parents=True)
    backend = LocalSubprocessBackend(tmp_path)
    result = await backend.exec("pwd", cwd=sub)
    assert result.returncode == 0
    assert result.stdout.strip() == str(sub.resolve())


async def test_cwd_relative_subdir_allowed(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    backend = LocalSubprocessBackend(tmp_path)
    result = await backend.exec("pwd", cwd="sub")
    assert result.stdout.strip() == str((tmp_path / "sub").resolve())


async def test_cwd_root_itself_allowed(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    result = await backend.exec("pwd", cwd=str(tmp_path))
    assert result.returncode == 0


async def test_cwd_prefix_sibling_rejected(tmp_path: Path) -> None:
    """A sibling sharing a name prefix (root /a/b, cwd /a/bb) must NOT pass.

    Guards against a future regression to naive str.startswith confinement.
    """
    root = tmp_path / "b"
    sibling = tmp_path / "bb"
    root.mkdir()
    sibling.mkdir()
    backend = LocalSubprocessBackend(root)
    with pytest.raises(SandboxError):
        await backend.exec("pwd", cwd=sibling)


async def test_missing_cwd_raises_sandbox_error(tmp_path: Path) -> None:
    """A cwd inside root but not existing is a clean SandboxError, not a raw OSError."""
    backend = LocalSubprocessBackend(tmp_path)
    with pytest.raises(SandboxError, match="does not exist"):
        await backend.exec("pwd", cwd=tmp_path / "ghost")


async def test_cwd_symlink_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "escape"
    link.symlink_to(outside)
    backend = LocalSubprocessBackend(root)
    with pytest.raises(SandboxError):
        await backend.exec("pwd", cwd=str(link))


# --- env curation (SECURITY) ----------------------------------------------


async def test_secret_env_not_leaked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SECRET", "topsecret")
    backend = LocalSubprocessBackend(tmp_path)
    result = await backend.exec("printf '%s' \"${MY_SECRET:-ABSENT}\"")
    assert result.stdout == "ABSENT"
    assert "topsecret" not in result.stdout


async def test_allowlisted_path_present(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    result = await backend.exec("printf '%s' \"${PATH:-MISSING}\"")
    assert result.stdout != "MISSING"
    assert result.stdout != ""


async def test_explicit_env_overlaid(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    result = await backend.exec("printf '%s' \"$INJECTED\"", env={"INJECTED": "hello-from-caller"})
    assert result.stdout == "hello-from-caller"


async def test_explicit_env_can_add_secret_deliberately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_SECRET", "leaked")
    backend = LocalSubprocessBackend(tmp_path)
    result = await backend.exec("printf '%s' \"$MY_SECRET\"", env={"MY_SECRET": "explicit"})
    assert result.stdout == "explicit"


async def test_custom_allowlist_projects_only_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALLOWED_VAR", "yes")
    monkeypatch.setenv("DENIED_VAR", "no")
    backend = LocalSubprocessBackend(tmp_path, env_allowlist=("ALLOWED_VAR",))
    allowed = await backend.exec("printf '%s' \"${ALLOWED_VAR:-X}\"")
    denied = await backend.exec("printf '%s' \"${DENIED_VAR:-X}\"")
    assert allowed.stdout == "yes"
    assert denied.stdout == "X"


# --- timeout ---------------------------------------------------------------


async def test_timeout_kills_and_reports(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    start = time.monotonic()
    result = await backend.exec("sleep 5", timeout=0.3)
    elapsed = time.monotonic() - start
    assert result.timed_out is True
    assert result.returncode == 124
    assert elapsed < 2.0


async def test_timeout_kills_process_group(tmp_path: Path) -> None:
    # A child that spawns a grandchild then waits: killing only the shell would
    # orphan the grandchild. The group kill must reap promptly.
    backend = LocalSubprocessBackend(tmp_path)
    start = time.monotonic()
    result = await backend.exec("sleep 30 & sleep 30", timeout=0.3)
    elapsed = time.monotonic() - start
    assert result.timed_out is True
    assert elapsed < 2.0


async def test_timeout_returns_promptly_despite_session_escape(tmp_path: Path) -> None:
    """A child that escapes the process group (setsid) must not hang exec().

    Without a bounded post-kill drain, the escaped child holding the stdout pipe
    blocks communicate() for its whole lifetime. Must return within the drain
    bound, well under the child's sleep.
    """
    backend = LocalSubprocessBackend(tmp_path)
    # setsid detaches the sleeper into its own session so killpg(group) misses
    # it; it inherits and holds the stdout pipe.
    command = "python3 -c 'import os,time; os.setsid(); time.sleep(20)' & sleep 20"
    start = time.monotonic()
    result = await backend.exec(command, timeout=0.3)
    elapsed = time.monotonic() - start
    assert result.timed_out is True
    assert result.returncode == 124
    # timeout (0.3) + drain bound (2.0) + slack — far below the 20s sleeps.
    assert elapsed < 5.0


async def test_fast_command_not_marked_timed_out(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    result = await backend.exec("true", timeout=5.0)
    assert result.timed_out is False
    assert result.returncode == 0


# --- output caps -----------------------------------------------------------


async def test_stdout_truncated_to_cap(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path, max_output_bytes=16)
    result = await backend.exec("printf 'x%.0s' $(seq 1 100)")
    assert result.truncated is True
    assert len(result.stdout) <= 16


async def test_stderr_truncated_to_cap(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path, max_output_bytes=16)
    result = await backend.exec("printf 'y%.0s' $(seq 1 100) 1>&2")
    assert result.truncated is True
    assert len(result.stderr) <= 16


async def test_small_output_not_truncated(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path, max_output_bytes=16)
    result = await backend.exec("printf 'short'")
    assert result.truncated is False
    assert result.stdout == "short"


async def test_non_utf8_output_decoded_with_replacement(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend(tmp_path)
    # Emit a lone 0xFF byte, which is not valid UTF-8.
    result = await backend.exec(r"printf '\377'")
    assert result.returncode == 0
    assert "�" in result.stdout


# --- ExecResult.output property -------------------------------------------


def test_output_property_combines() -> None:
    assert ExecResult(0, "out", "err").output == "outerr"
    assert ExecResult(0, "out", "").output == "out"
    assert ExecResult(0, "", "err").output == "err"
    assert ExecResult(0, "", "").output == ""
