"""The ``edit_file`` builtin: anchor-based, exactly-once find/replace.

This tool deliberately avoids patch/diff application (SPEC Phase 1 risks §): a
unified diff is brittle against line drift and fuzz, and a fuzzy apply can land
a hunk in the wrong place. Instead the model supplies an exact ``old`` snippet
that must occur **exactly once** in the file; that single occurrence is replaced
with ``new``. The exactly-once rule is the core safety property — an ambiguous
anchor is refused rather than silently rewriting several places.

Like every filesystem builtin, the workspace root is bound at construction via
:func:`make_edit_file` so the model-facing handler takes only its own arguments,
and every path is confined under that root before any I/O happens.
"""

from __future__ import annotations

from pathlib import Path

from genie.tools.base import Tool, tool
from genie.tools.builtins._workspace import WorkspaceEscape, confine
from genie.tools.result import ToolResult

_CONTEXT_LINES = 2
_MAX_SNIPPET_CHARS = 600


def _clip(text: str) -> str:
    """Bound a snippet so the result never echoes an enormous region.

    Args:
        text: The snippet to bound.

    Returns:
        ``text`` unchanged when short, else a head slice with an elision marker.
    """
    if len(text) <= _MAX_SNIPPET_CHARS:
        return text
    return text[:_MAX_SNIPPET_CHARS] + "\n…[snippet truncated]…"


def _changed_region(new_text: str, index: int, new: str) -> str:
    """Render a short confirmation of the edited region with surrounding context.

    Args:
        new_text: The full file contents after the replacement.
        index: Character offset in ``new_text`` where ``new`` was written.
        new: The replacement text that now lives at ``index``.

    Returns:
        A few context lines before and after the change, framed by a header,
        suitable as the model-facing observation (never the whole file).
    """
    start_line = new_text[:index].count("\n")
    end_line = start_line + new.count("\n")

    lines = new_text.splitlines()
    first = max(0, start_line - _CONTEXT_LINES)
    last = min(len(lines), end_line + _CONTEXT_LINES + 1)
    window = "\n".join(lines[first:last])
    return f"@@ lines {first + 1}-{last} @@\n{_clip(window)}"


def make_edit_file(root: str | Path) -> Tool:
    """Build an :func:`edit_file` tool confined to ``root``.

    The returned tool replaces a single, unambiguous ``old`` snippet with
    ``new`` in a file under ``root``. It is marked ``dangerous`` because it
    mutates the workspace; a Phase-2 approval hook can gate it.

    Args:
        root: The workspace root. All edited paths must resolve within it.

    Returns:
        A configured :class:`~genie.tools.base.Tool` named ``edit_file``.
    """
    base = Path(root).resolve()

    @tool(name="edit_file", tags=["fs"], dangerous=True)
    async def edit_file(path: str, old: str, new: str) -> ToolResult:
        """Replace an exact unique snippet in a workspace file (anchor-based).

        ``old`` must appear exactly once in the file. Zero matches, multiple
        matches, or an empty ``old`` are refused without writing. On success the
        single occurrence becomes ``new`` and a short view of the changed region
        is returned.
        """
        try:
            target = confine(base, path)
        except WorkspaceEscape as exc:
            return ToolResult.error(str(exc))

        if old == "":
            return ToolResult.error("empty anchor: 'old' must be a non-empty snippet")

        if not target.is_file():
            return ToolResult.error(f"file not found: {path!r}")

        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult.error(f"could not read {path!r}: {exc}")

        count = text.count(old)
        if count == 0:
            return ToolResult.error(f"anchor not found: {_clip(old)!r}")
        if count > 1:
            return ToolResult.error(
                f"anchor is ambiguous (found {count} times); include more context"
            )

        index = text.index(old)
        new_text = text[:index] + new + text[index + len(old) :]
        try:
            target.write_text(new_text, encoding="utf-8")
        except OSError as exc:
            return ToolResult.error(f"could not write {path!r}: {exc}")

        region = _changed_region(new_text, index, new)
        return ToolResult.text(
            f"edited {path} (1 replacement)\n{region}",
            path=str(target),
            replacements=1,
        )

    return edit_file
