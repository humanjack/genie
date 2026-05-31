"""A scripted, deterministic :class:`SandboxBackend` for tests.

``RecordingBackend`` is the second implementation of the sandbox contract (the
first being :class:`~genie.sandbox.local_subprocess.LocalSubprocessBackend`).
Its existence is the proof the abstraction holds: tools that shell out
(``bash``, ``write_file``, ...) can be tested with no real process — every call
is recorded and every result is pre-scripted. See SPEC operating principle
"Replaceability is testable".
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from genie.sandbox.base import ExecResult, SandboxBackend


class RecordingBackend(SandboxBackend):
    """Replay scripted :class:`ExecResult`s and record every call.

    Construct with ``results`` as either:

    - a **list** of :class:`ExecResult`, replayed in call order (one per
      :meth:`exec`); or
    - a **dict** keyed by command string, where each :meth:`exec` returns the
      result mapped to its exact ``command``.

    Every call's ``command``, ``cwd``, ``env``, and ``timeout`` are appended to
    the public :attr:`calls` list, so tests can assert exactly what a tool
    asked the sandbox to run. Over-driving the script — exhausting the list or
    requesting an unmapped command — raises a clear error rather than returning
    a silent default.

    Args:
        results: The scripted results, as an ordered list or a command-keyed
            dict. ``None`` is an empty script (any :meth:`exec` over-drives).
    """

    def __init__(
        self,
        results: list[ExecResult] | Mapping[str, ExecResult] | None = None,
    ) -> None:
        self._keyed: dict[str, ExecResult] | None
        self._queue: list[ExecResult]
        if results is None:
            self._keyed = None
            self._queue = []
        elif isinstance(results, Mapping):
            self._keyed = dict(results)
            self._queue = []
        else:
            self._keyed = None
            self._queue = list(results)
        self._cursor = 0
        self.calls: list[dict] = []

    async def exec(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> ExecResult:
        """Record the call, then return the next (or command-keyed) result.

        Raises:
            KeyError: In dict mode, if ``command`` has no scripted result.
            IndexError: In list mode, if invoked more times than there are
                scripted results.
        """
        self.calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": dict(env) if env is not None else None,
                "timeout": timeout,
            }
        )
        if self._keyed is not None:
            try:
                return self._keyed[command]
            except KeyError:
                raise KeyError(
                    f"RecordingBackend has no scripted result for command {command!r}; "
                    f"known commands: {sorted(self._keyed)}"
                ) from None
        if self._cursor >= len(self._queue):
            raise IndexError(
                f"RecordingBackend exhausted: {len(self._queue)} scripted result(s), "
                f"but exec() was called {self._cursor + 1} time(s)"
            )
        result = self._queue[self._cursor]
        self._cursor += 1
        return result
