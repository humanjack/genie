"""The ``write_file`` builtin: create or overwrite a text file in the workspace.

Exposed via the :func:`make_write_file` factory, which binds the workspace root
so the model-facing handler receives only ``path`` and ``content``. Every write
is confined to that root: a ``path`` that resolves outside the workspace (via
``..`` segments or an absolute path) is refused with an error result rather than
touching the filesystem. The tool is marked ``dangerous`` so the Phase-2
approval hook gates it before execution (SPEC §7.3).
"""

from __future__ import annotations

from pathlib import Path

from genie.tools.base import Tool, tool
from genie.tools.builtins._workspace import WorkspaceEscape, confine
from genie.tools.result import ToolResult


def make_write_file(root: str | Path) -> Tool:
    """Build a ``write_file`` tool confined to the ``root`` workspace.

    Args:
        root: The workspace directory all writes are confined to. It is
            resolved once at construction time and captured by the handler.

    Returns:
        A :class:`~genie.tools.base.Tool` whose handler creates or overwrites a
        UTF-8 text file at a path relative to ``root``.
    """
    base = Path(root).resolve()

    @tool(name="write_file", tags=["fs"], dangerous=True)
    async def write_file(path: str, content: str) -> ToolResult:
        """Create or overwrite a UTF-8 text file within the workspace.

        Parent directories are created as needed. ``path`` is interpreted
        relative to the workspace root; paths that escape it are rejected.
        """
        try:
            target = confine(base, path)
        except WorkspaceEscape as exc:
            return ToolResult.error(str(exc))

        if target.is_dir():
            return ToolResult.error(f"path {path!r} is a directory, not a file")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        byte_count = len(content.encode("utf-8"))
        return ToolResult.text(f"wrote {byte_count} bytes to {path}")

    return write_file
