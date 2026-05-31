"""Shared workspace-path confinement for the file tools (secure by default).

read_file / write_file / edit_file all confine a model-supplied path to the
session's workspace root. Centralizing it here keeps the security check in one
audited place rather than three diverging copies, and locks the symlink
guarantee with one test suite.
"""

from __future__ import annotations

from pathlib import Path


class WorkspaceEscape(Exception):
    """Raised when a model-supplied path resolves outside the workspace root."""


def confine(root: str | Path, path: str) -> Path:
    """Resolve ``path`` relative to ``root`` and confine it to that root.

    The candidate is resolved with :meth:`Path.resolve` — which expands symlinks
    and ``..`` segments — *before* the containment check, so a symlink inside the
    workspace that points outside it, a ``../`` traversal, and an absolute path
    elsewhere are all rejected. ``root`` is resolved too, so the comparison is
    symlink-stable on both sides.

    Args:
        root: The workspace boundary (the session working directory).
        path: A model-supplied path, relative to ``root`` or absolute.

    Returns:
        The resolved absolute path, guaranteed to be ``root`` or a descendant.

    Raises:
        WorkspaceEscape: If the resolved path is not within ``root``. The message
            names only the supplied ``path`` (never the absolute root) to avoid
            leaking host paths to the model.
    """
    base = Path(root).resolve()
    candidate = (base / path).resolve()
    if not candidate.is_relative_to(base):
        raise WorkspaceEscape(f"path escapes workspace: {path!r}")
    return candidate
