"""Tests for the ``write_file`` builtin and its workspace confinement."""

from __future__ import annotations

from pathlib import Path

from genie.tools.base import Tool
from genie.tools.builtins.write_file import make_write_file
from genie.tools.result import ToolResult


def test_factory_returns_dangerous_fs_tool(tmp_path: Path) -> None:
    write_file = make_write_file(tmp_path)
    assert isinstance(write_file, Tool)
    assert write_file.name == "write_file"
    assert write_file.dangerous is True
    assert "fs" in write_file.tags


def test_input_schema_requires_path_and_content(tmp_path: Path) -> None:
    schema = make_write_file(tmp_path).input_schema
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"path", "content"}
    assert schema["properties"]["path"]["type"] == "string"
    assert schema["properties"]["content"]["type"] == "string"
    assert set(schema["required"]) == {"path", "content"}


async def test_writes_new_file(tmp_path: Path) -> None:
    write_file = make_write_file(tmp_path)

    result = await write_file.handler(path="hello.txt", content="hi there")

    assert isinstance(result, ToolResult)
    assert result.is_error is False
    target = tmp_path / "hello.txt"
    assert target.read_text(encoding="utf-8") == "hi there"
    assert "8 bytes" in result.content
    assert "hello.txt" in result.content


async def test_reports_utf8_byte_count_not_char_count(tmp_path: Path) -> None:
    write_file = make_write_file(tmp_path)

    # "é" is one character but two bytes in UTF-8.
    result = await write_file.handler(path="accent.txt", content="é")

    assert result.is_error is False
    assert "2 bytes" in result.content
    assert (tmp_path / "accent.txt").read_text(encoding="utf-8") == "é"


async def test_creates_nested_parent_dirs(tmp_path: Path) -> None:
    write_file = make_write_file(tmp_path)

    result = await write_file.handler(path="a/b/c.txt", content="deep")

    assert result.is_error is False
    target = tmp_path / "a" / "b" / "c.txt"
    assert target.read_text(encoding="utf-8") == "deep"


async def test_overwrites_existing_file(tmp_path: Path) -> None:
    write_file = make_write_file(tmp_path)
    existing = tmp_path / "note.txt"
    existing.write_text("old content", encoding="utf-8")

    result = await write_file.handler(path="note.txt", content="new")

    assert result.is_error is False
    assert existing.read_text(encoding="utf-8") == "new"


async def test_relative_escape_returns_error_and_writes_nothing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_file = make_write_file(workspace)

    result = await write_file.handler(path="../evil.txt", content="pwned")

    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert "escapes workspace" in result.content
    assert not (tmp_path / "evil.txt").exists()


async def test_absolute_path_outside_root_returns_error(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    write_file = make_write_file(workspace)

    target = outside / "evil.txt"
    result = await write_file.handler(path=str(target), content="pwned")

    assert result.is_error is True
    assert "escapes workspace" in result.content
    assert not target.exists()


async def test_escape_does_not_raise_out_of_handler(tmp_path: Path) -> None:
    """A traversal must surface as an error result, never a raised exception."""
    write_file = make_write_file(tmp_path)

    # Would raise if the handler let _PathEscape propagate.
    result = await write_file.handler(path="../../../etc/evil.txt", content="x")

    assert result.is_error is True


async def test_writing_where_directory_exists_returns_error(tmp_path: Path) -> None:
    write_file = make_write_file(tmp_path)
    (tmp_path / "adir").mkdir()

    result = await write_file.handler(path="adir", content="data")

    assert result.is_error is True
    assert "directory" in result.content
    assert (tmp_path / "adir").is_dir()


async def test_accepts_str_root(tmp_path: Path) -> None:
    write_file = make_write_file(str(tmp_path))

    result = await write_file.handler(path="x.txt", content="ok")

    assert result.is_error is False
    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == "ok"
