"""Tests for the shared workspace-path confinement helper.

This is the single audited home for the file tools' security boundary, so the
escape vectors — including the symlink case that motivates using ``resolve()``
— are regression-locked here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from genie.tools.builtins._workspace import WorkspaceEscape, confine


def test_relative_path_resolved_within_root(tmp_path: Path) -> None:
    target = confine(tmp_path, "sub/file.txt")
    assert target == (tmp_path / "sub" / "file.txt").resolve()
    assert target.is_relative_to(tmp_path.resolve())


def test_root_itself_is_allowed(tmp_path: Path) -> None:
    assert confine(tmp_path, ".") == tmp_path.resolve()


def test_parent_traversal_rejected(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceEscape):
        confine(tmp_path, "../outside.txt")


def test_deep_traversal_back_out_rejected(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    with pytest.raises(WorkspaceEscape):
        confine(tmp_path, "sub/../../escape.txt")


def test_absolute_path_outside_rejected(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceEscape):
        confine(tmp_path, "/etc/hostname")


def test_prefix_sibling_rejected(tmp_path: Path) -> None:
    """root /a/ws must not admit sibling /a/ws-evil (segment-aware, not string)."""
    root = tmp_path / "ws"
    sibling = tmp_path / "ws-evil"
    root.mkdir()
    sibling.mkdir()
    with pytest.raises(WorkspaceEscape):
        confine(root, str(sibling / "x.txt"))


def test_symlink_inside_pointing_outside_rejected(tmp_path: Path) -> None:
    """The motivating case: a symlink within root that targets outside it.

    resolve() expands the link before the containment check, so following it is
    rejected. A regression to a non-symlink-expanding resolver would fail here.
    """
    root = tmp_path / "ws"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret")
    (root / "link").symlink_to(outside)

    with pytest.raises(WorkspaceEscape):
        confine(root, "link/secret.txt")


def test_symlinked_file_inside_pointing_outside_rejected(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("top secret")
    (root / "alias.txt").symlink_to(secret)

    with pytest.raises(WorkspaceEscape):
        confine(root, "alias.txt")


def test_message_does_not_leak_absolute_root(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceEscape) as exc:
        confine(tmp_path, "../escape.txt")
    assert str(tmp_path.resolve()) not in str(exc.value)
    assert "../escape.txt" in str(exc.value)
