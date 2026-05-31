"""Tests for the :func:`make_edit_file` anchor-based find/replace builtin."""

from __future__ import annotations

from pathlib import Path

from genie.tools.base import Tool
from genie.tools.builtins.edit_file import make_edit_file


def _write(root: Path, name: str, content: str) -> Path:
    target = root / name
    target.write_text(content, encoding="utf-8")
    return target


async def test_unique_match_replaces_once_and_updates_disk(tmp_path: Path) -> None:
    target = _write(tmp_path, "code.py", "alpha\nbeta\ngamma\n")
    tool = make_edit_file(tmp_path)

    result = await tool.handler(path="code.py", old="beta", new="BETA")

    assert result.is_error is False
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert "BETA" in result.content
    assert result.metadata["replacements"] == 1


async def test_result_shows_change_but_not_whole_file(tmp_path: Path) -> None:
    body = "\n".join(f"line{i}" for i in range(200))
    _write(tmp_path, "big.txt", body + "\nNEEDLE\n" + body)
    tool = make_edit_file(tmp_path)

    result = await tool.handler(path="big.txt", old="NEEDLE", new="FOUND")

    assert result.is_error is False
    assert "FOUND" in result.content
    # A short region, not the multi-hundred-line file.
    assert len(result.content.splitlines()) < 20


def test_tool_is_dangerous_and_tagged_fs(tmp_path: Path) -> None:
    tool = make_edit_file(tmp_path)
    assert tool.dangerous is True
    assert "fs" in tool.tags


async def test_anchor_not_found_errors_and_leaves_file_unchanged(tmp_path: Path) -> None:
    target = _write(tmp_path, "code.py", "alpha\nbeta\n")
    before = target.read_bytes()
    tool = make_edit_file(tmp_path)

    result = await tool.handler(path="code.py", old="zeta", new="X")

    assert result.is_error is True
    assert "not found" in result.content
    assert target.read_bytes() == before


async def test_ambiguous_anchor_errors_with_count_and_leaves_file_unchanged(
    tmp_path: Path,
) -> None:
    target = _write(tmp_path, "code.py", "dup\nmiddle\ndup\n")
    before = target.read_bytes()
    tool = make_edit_file(tmp_path)

    result = await tool.handler(path="code.py", old="dup", new="X")

    assert result.is_error is True
    assert "ambiguous" in result.content
    assert "2" in result.content
    # The key safety property: nothing is rewritten when the anchor is ambiguous.
    assert target.read_bytes() == before


async def test_empty_old_errors_and_leaves_file_unchanged(tmp_path: Path) -> None:
    target = _write(tmp_path, "code.py", "content\n")
    before = target.read_bytes()
    tool = make_edit_file(tmp_path)

    result = await tool.handler(path="code.py", old="", new="X")

    assert result.is_error is True
    assert "empty" in result.content
    assert target.read_bytes() == before


async def test_missing_file_errors(tmp_path: Path) -> None:
    tool = make_edit_file(tmp_path)

    result = await tool.handler(path="nope.py", old="a", new="b")

    assert result.is_error is True
    assert "not found" in result.content


async def test_relative_escape_errors_without_raising(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret\n", encoding="utf-8")
    tool = make_edit_file(tmp_path)

    result = await tool.handler(path="../outside_secret.txt", old="secret", new="X")

    assert result.is_error is True
    assert "escapes workspace" in result.content
    assert outside.read_text(encoding="utf-8") == "secret\n"


async def test_absolute_path_outside_root_errors(tmp_path: Path) -> None:
    outside = tmp_path.parent / "abs_outside.txt"
    outside.write_text("data\n", encoding="utf-8")
    tool = make_edit_file(tmp_path)

    result = await tool.handler(path=str(outside), old="data", new="X")

    assert result.is_error is True
    assert "escapes workspace" in result.content
    assert outside.read_text(encoding="utf-8") == "data\n"


def test_returns_tool_named_edit_file_with_required_params(tmp_path: Path) -> None:
    tool = make_edit_file(tmp_path)

    assert isinstance(tool, Tool)
    assert tool.name == "edit_file"
    schema = tool.input_schema
    assert set(schema["properties"]) == {"path", "old", "new"}
    assert set(schema["required"]) == {"path", "old", "new"}


async def test_factory_accepts_str_root(tmp_path: Path) -> None:
    target = _write(tmp_path, "s.txt", "hello world\n")
    tool = make_edit_file(str(tmp_path))

    result = await tool.handler(path="s.txt", old="world", new="there")

    assert result.is_error is False
    assert target.read_text(encoding="utf-8") == "hello there\n"


async def test_replacement_at_start_of_file_renders_region(tmp_path: Path) -> None:
    target = _write(tmp_path, "head.txt", "first\nsecond\nthird\n")
    tool = make_edit_file(tmp_path)

    result = await tool.handler(path="head.txt", old="first", new="FIRST")

    assert result.is_error is False
    assert target.read_text(encoding="utf-8") == "FIRST\nsecond\nthird\n"
    assert "FIRST" in result.content
