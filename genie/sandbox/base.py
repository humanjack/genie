"""Sandbox contract: the single seam every command execution is reached through.

A sandbox backend runs a shell command on behalf of a tool (``bash``,
``write_file``, ...) under a security policy the caller can rely on. The
contract is deliberately small — one async :meth:`SandboxBackend.exec` method
returning a plain :class:`ExecResult` dataclass — so the agent loop and tools
never learn whether the command ran in a local subprocess, a recording fake, or
(later) a container. See SPEC §6.1 and the operating principle "Secure by
default": every action that touches the shell goes through this chokepoint, and
implementations confine the working directory and curate the environment rather
than inheriting the parent's full ambient authority.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class SandboxError(Exception):
    """Raised when a request violates the sandbox security policy.

    The canonical case is a working-directory escape: a ``cwd`` that resolves
    outside the backend's confinement root (via ``..`` traversal, an absolute
    path, or a symlink pointing out). Backends raise this *before* spawning any
    process, so a rejected command never executes.
    """


@dataclass
class ExecResult:
    """The outcome of running one command in a sandbox.

    ``stdout`` and ``stderr`` are kept separate (never pre-merged) so callers
    can route them independently; use :attr:`output` when a single combined
    stream is wanted. Both streams are already decoded text — backends decode
    captured bytes with ``errors="replace"`` so the result is always valid
    ``str`` even for non-UTF-8 output.

    Attributes:
        returncode: The process exit status. By convention a backend that
            timed the command out reports ``124`` (the shell ``timeout``
            convention) and sets :attr:`timed_out`.
        stdout: Captured standard output, decoded and possibly truncated.
        stderr: Captured standard error, decoded and possibly truncated.
        truncated: ``True`` if either stream exceeded the backend's output cap
            and was cut short.
        timed_out: ``True`` if the command was killed for exceeding its
            timeout.
    """

    returncode: int
    stdout: str
    stderr: str
    truncated: bool = False
    timed_out: bool = False

    @property
    def output(self) -> str:
        """Return ``stdout`` and ``stderr`` joined into one stream.

        A convenience for callers that want a single block of text; the
        non-empty parts are concatenated with stdout first, then stderr.
        """
        if self.stdout and self.stderr:
            return self.stdout + self.stderr
        return self.stdout or self.stderr


class SandboxBackend(ABC):
    """Abstract base every command-execution backend must satisfy.

    Concrete subclasses implement :meth:`exec`. Tools depend only on this
    interface, which is what makes the runner swappable with zero edits to the
    loop or tools (SPEC operating principle "Pluggable everywhere"): a real
    :class:`~genie.sandbox.local_subprocess.LocalSubprocessBackend` and a test
    :class:`~genie.sandbox.recording.RecordingBackend` are interchangeable.

    **Security contract** that every implementation MUST honor:

    - **Working-directory confinement.** ``cwd`` defaults to the backend's root
      and is confined to that root or a subpath of it. A ``cwd`` resolving
      outside the root (parent-dir traversal, absolute path, or symlink escape)
      raises :class:`SandboxError` *before* any process is spawned. Symlinks are
      resolved when checking confinement.
    - **Environment curation.** The child process does **not** inherit the full
      parent environment. The backend starts from a curated allowlist of safe
      variables and overlays only what the caller explicitly passes, so secrets
      such as ``AWS_*`` or ``GH_TOKEN`` never leak unless deliberately provided.
    """

    @abstractmethod
    async def exec(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> ExecResult:
        """Run ``command`` under the sandbox policy and return its result.

        Args:
            command: The shell command line to execute.
            cwd: Working directory for the command. ``None`` means the
                backend's confinement root. Any value must resolve to the root
                or a subpath of it, or :class:`SandboxError` is raised.
            env: Environment variables to overlay on the backend's curated
                base environment. ``None`` runs with the curated base only.
            timeout: Seconds before the command is killed; on expiry the
                backend returns an :class:`ExecResult` with ``timed_out=True``
                rather than raising.

        Returns:
            An :class:`ExecResult` capturing exit status, output, and the
            ``truncated`` / ``timed_out`` flags.

        Raises:
            SandboxError: If ``cwd`` escapes the confinement root.
        """
        raise NotImplementedError
