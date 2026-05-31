"""The ``read_file`` built-in: read a UTF-8 text file confined to the workspace.

The session's working directory is not a model argument, so this module exports
a *factory*, :func:`make_read_file`, that binds the workspace root and returns a
configured :class:`~genie.tools.base.Tool`. The handler then receives only the
model-provided arguments (``path`` and an optional line slice).

Confinement is secure by default: every ``path`` is resolved against the bound
root and rejected if it escapes (an absolute path or one climbing out via
``..``). Reading a spill file is just an ordinary read of a real path within the
workspace — no special handling is required (SPEC §5.4).
"""

from __future__ import annotations

from pathlib import Path

from genie.tools.base import Tool, tool
from genie.tools.builtins._workspace import WorkspaceEscape, confine
from genie.tools.result import ToolResult


def make_read_file(root: str | Path) -> Tool:
    """Build a ``read_file`` tool bound to the workspace ``root``.

    Args:
        root: The session's working directory. Resolved once so confinement
            comparisons are against a canonical absolute path.

    Returns:
        A :class:`~genie.tools.base.Tool` whose handler reads a UTF-8 text file
        within the workspace, optionally returning only a slice of its lines.
    """
    base = Path(root).resolve()

    @tool(name="read_file", tags=["fs"], max_result_chars=16000)
    async def read_file(
        path: str, offset: int | None = None, limit: int | None = None
    ) -> ToolResult:
        """Read a UTF-8 text file within the workspace. Optionally a line slice.

        ``offset`` is a 0-based line index and ``limit`` a maximum line count;
        out-of-range values are clamped. With neither given, the whole file is
        returned. Paths that escape the workspace, missing files, and
        non-regular files each return an error result.
        """
        try:
            target = confine(base, path)
        except WorkspaceEscape as exc:
            return ToolResult.error(str(exc))

        if not target.exists():
            return ToolResult.error(f"no such file: {path}")
        if not target.is_file():
            return ToolResult.error(f"not a regular file: {path}")

        content = target.read_text(encoding="utf-8", errors="replace")

        if offset is not None or limit is not None:
            lines = content.splitlines(keepends=True)
            start = max(0, offset or 0)
            end = len(lines) if limit is None else start + max(0, limit)
            content = "".join(lines[start:end])

        return ToolResult.text(content)

    return read_file
