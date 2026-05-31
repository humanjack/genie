"""Local subprocess sandbox backend (SPEC §6.1, v1).

:class:`LocalSubprocessBackend` runs commands on the host via
``asyncio.create_subprocess_shell`` under two security guarantees — a confined
working directory and a curated environment — plus a timeout and per-stream
output caps. It is the default v1 runner; the heavier Docker backend (SPEC §6.3)
and the bash-AST guard (SPEC §6.2) are deferred to later phases.

**Scope of "confinement" (read before trusting it).** cwd confinement controls
only the command's *starting directory*. It does **not** sandbox the filesystem
or network: the command string still runs an arbitrary shell, so
``cd / && cat /etc/passwd``, writes to absolute paths, and outbound network
calls all succeed. Real filesystem/network isolation is the Docker backend's job
(SPEC §6.3); per-command allow/deny is the bash-AST guard's (SPEC §6.2). For v1,
the dangerous-tool approval hook (SPEC §7.3) is what gates risky commands — this
backend is only the cwd/env/timeout primitive beneath it.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path

from genie.sandbox.base import ExecResult, SandboxBackend, SandboxError

DEFAULT_ENV_ALLOWLIST: tuple[str, ...] = ("PATH", "HOME", "LANG", "LC_ALL", "TERM")
"""Environment variables projected from the parent by default.

Deliberately minimal: enough to find executables (``PATH``), resolve ``~``
(``HOME``), and behave sanely in a locale/terminal — and nothing that carries
credentials. Anything else must be passed explicitly per call.
"""

TIMEOUT_RETURNCODE = 124
"""Exit status reported on timeout, matching the shell ``timeout(1)`` convention."""

DRAIN_TIMEOUT = 2.0
"""Seconds to wait draining output after a kill before giving up.

A child that escaped the process group (e.g. via ``setsid``) can hold the
stdout/stderr pipe open and make the post-kill ``communicate()`` block for its
whole lifetime. Bounding the drain guarantees a timed-out :meth:`exec` still
returns promptly, even if it means dropping a stuck child's trailing output.
"""


class LocalSubprocessBackend(SandboxBackend):
    """Run commands as local subprocesses, confined to a root directory.

    Args:
        root: The session working directory and confinement boundary. Every
            command runs here or in a subpath; a ``cwd`` outside it is rejected.
            Resolved to an absolute real path once at construction.
        env_allowlist: Names of parent-environment variables to project into the
            child's base environment. Defaults to :data:`DEFAULT_ENV_ALLOWLIST`
            (no credential-bearing variables).
        max_output_bytes: Per-stream cap; ``stdout`` and ``stderr`` are each
            truncated to this many bytes and :attr:`ExecResult.truncated` is set
            when a stream exceeds it.

    The two security guarantees (see :class:`SandboxBackend`) are enforced here:

    - **cwd confinement** — :meth:`exec` resolves the requested ``cwd`` (default
      = root) to an absolute real path with symlinks expanded, then requires it
      to be the root or a subpath; otherwise it raises :class:`SandboxError`
      before spawning anything. This rejects ``..`` traversal, absolute paths
      like ``/etc``, and symlinks pointing outside the root.
    - **env curation** — the child environment is built as an allowlist
      projection of the current ``os.environ`` (only :attr:`env_allowlist`
      keys), with any explicit ``env`` overlaid on top. The parent's full
      environment is never inherited, so secrets do not leak by default.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        env_allowlist: Sequence[str] = DEFAULT_ENV_ALLOWLIST,
        max_output_bytes: int = 8192,
    ) -> None:
        self.root: Path = Path(root).resolve()
        self.env_allowlist: tuple[str, ...] = tuple(env_allowlist)
        self.max_output_bytes: int = max_output_bytes

    def _resolve_cwd(self, cwd: str | Path | None) -> Path:
        """Resolve ``cwd`` and confine it to the root, or raise.

        ``None`` resolves to the root itself. Symlinks are expanded via
        ``Path.resolve()`` so a symlink inside the root that points outside it
        is rejected.

        Raises:
            SandboxError: If the resolved path is neither the root nor a
                subpath of it.
        """
        target = self.root if cwd is None else Path(cwd)
        if not target.is_absolute():
            target = self.root / target
        resolved = target.resolve()
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise SandboxError(
                f"cwd {resolved} escapes sandbox root {self.root}; "
                "commands are confined to the root or a subpath"
            )
        return resolved

    def _build_env(self, env: Mapping[str, str] | None) -> dict[str, str]:
        """Build the child environment: allowlist projection plus overlay.

        Only :attr:`env_allowlist` keys are projected from the current
        ``os.environ``; the caller's explicit ``env`` (if any) is layered on
        top. Nothing else from the parent process is inherited.
        """
        curated = {k: os.environ[k] for k in self.env_allowlist if k in os.environ}
        if env:
            curated.update(env)
        return curated

    def _cap(self, raw: bytes) -> tuple[str, bool]:
        """Decode ``raw`` to text, truncating to :attr:`max_output_bytes`.

        Returns the decoded (possibly truncated) text and whether truncation
        occurred. Bytes are decoded with ``errors="replace"`` so binary or
        non-UTF-8 output never raises.
        """
        truncated = len(raw) > self.max_output_bytes
        clipped = raw[: self.max_output_bytes] if truncated else raw
        return clipped.decode("utf-8", errors="replace"), truncated

    async def exec(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> ExecResult:
        """Run ``command`` confined to the root with a curated environment.

        See :meth:`SandboxBackend.exec` for the full contract. On timeout the
        process (and its process group) is killed and an :class:`ExecResult`
        with ``returncode=124`` and ``timed_out=True`` is returned, carrying
        whatever output was captured before the kill — the underlying
        ``TimeoutError`` is never surfaced to the caller.

        Raises:
            SandboxError: If ``cwd`` escapes the confinement root.
        """
        run_cwd = self._resolve_cwd(cwd)
        run_env = self._build_env(env)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(run_cwd),
                env=run_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except (FileNotFoundError, NotADirectoryError) as exc:
            # cwd passed confinement but does not exist / is not a directory.
            raise SandboxError(f"working directory does not exist: {run_cwd}") from exc

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            stdout_bytes, stderr_bytes = await self._terminate(proc)

        stdout, out_trunc = self._cap(stdout_bytes)
        stderr, err_trunc = self._cap(stderr_bytes)
        returncode = TIMEOUT_RETURNCODE if timed_out else (proc.returncode or 0)
        return ExecResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            truncated=out_trunc or err_trunc,
            timed_out=timed_out,
        )

    @staticmethod
    async def _terminate(proc: asyncio.subprocess.Process) -> tuple[bytes, bytes]:
        """Kill a timed-out process group and drain whatever it produced.

        The child is spawned with ``start_new_session=True`` so it leads its own
        process group; killing the group (``killpg``) reaps grandchildren a bare
        ``proc.kill()`` would orphan. Best-effort: races where the process has
        already exited are ignored.
        """
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        with suppress(ProcessLookupError):
            proc.kill()
        # Bound the drain: a session-escaped child holding the pipe must not be
        # able to block exec() past the timeout (see DRAIN_TIMEOUT).
        try:
            return await asyncio.wait_for(proc.communicate(), timeout=DRAIN_TIMEOUT)
        except Exception:  # noqa: BLE001 — drain is best-effort; never block exec()
            return b"", b""
