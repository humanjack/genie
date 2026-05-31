"""The ``bash`` builtin: run a shell command through the sandbox.

The sandbox backend is not a model argument, so this module exports a *factory*,
:func:`make_bash`, that binds a :class:`~genie.sandbox.base.SandboxBackend` (and
a per-call timeout) and returns a configured :class:`~genie.tools.base.Tool`.
The handler then receives only the model-provided ``command``.

The tool is a thin, safe wrapper: the sandbox owns confinement, the curated
environment, the timeout, and output caps (SPEC §6.1). The handler's only jobs
are to format the result for the model and to draw the line between a *command*
failure and a *tool* failure. A nonzero exit is **not** a tool error — the model
should see the failure output and react, matching "tool errors are returned to
the model, not retried". Only the sandbox raising (e.g.
:class:`~genie.sandbox.base.SandboxError` for a bad cwd) becomes
:meth:`ToolResult.error`. The tool is marked ``dangerous`` so the Phase-2
approval hook gates it before execution (SPEC §7.3).
"""

from __future__ import annotations

from genie.sandbox.base import ExecResult, SandboxBackend, SandboxError
from genie.tools.base import Tool, tool
from genie.tools.result import ToolResult


def _format(result: ExecResult, timeout: float) -> str:
    """Render an :class:`ExecResult` into text the model can act on.

    A zero exit returns the merged output alone. A nonzero exit is prefixed with
    an ``exit code:`` line so the failure is unmistakable even when the command
    printed nothing. Timeout and truncation are surfaced as their own notices so
    the model knows output was cut short or the command never finished.

    Args:
        result: The sandbox's outcome for the command.
        timeout: The timeout the command was run with, named in the timeout
            notice.

    Returns:
        The text for the model's observation.
    """
    parts: list[str] = []
    if result.timed_out:
        parts.append(f"timed out after {timeout}s")
    elif result.returncode != 0:
        parts.append(f"exit code: {result.returncode}")
    output = result.output
    if output:
        parts.append(output)
    if result.truncated:
        parts.append("[output truncated]")
    return "\n".join(parts)


def make_bash(sandbox: SandboxBackend, *, timeout: float = 30.0) -> Tool:
    """Build a ``bash`` tool bound to ``sandbox``.

    Args:
        sandbox: The backend every command is executed through. Its policy
            (cwd confinement, env curation, output caps) is what makes the tool
            safe; this wrapper adds none of its own.
        timeout: Seconds to allow each command before the sandbox kills it.
            Forwarded to :meth:`SandboxBackend.exec` on every call.

    Returns:
        A :class:`~genie.tools.base.Tool` whose handler runs a shell command in
        the workspace and returns its output.
    """

    @tool(name="bash", tags=["shell"], dangerous=True)
    async def bash(command: str) -> ToolResult:
        """Run a shell command in the workspace and return its output.

        A nonzero exit status is reported in the result (prefixed with an
        ``exit code:`` line) but is **not** a tool error: the output is returned
        for the model to read and react to. Only a sandbox failure — such as a
        working directory that escapes the workspace — yields an error result.
        """
        try:
            result = await sandbox.exec(command, timeout=timeout)
        except SandboxError as exc:
            return ToolResult.error(f"sandbox error: {exc}")
        return ToolResult.text(_format(result, timeout))

    return bash
