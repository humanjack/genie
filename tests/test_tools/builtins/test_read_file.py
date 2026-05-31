"""Tests for the ``read_file`` built-in tool factory."""

from __future__ import annotations

from pathlib import Path

from genie.tools.base import Tool
from genie.tools.builtins.read_file import make_read_file


def test_factory_returns_tool_with_object_schema_and_required_path(tmp_path: Path) -> None:
    rf = make_read_file(tmp_path)
    assert isinstance(rf, Tool)
    assert rf.name == "read_file"
    assert rf.tags == ["fs"]
    schema = rf.input_schema
    assert schema["type"] == "object"
    assert "path" in schema["properties"]
    assert schema["required"] == ["path"]


async def test_reads_full_file_content(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello\nworld\n", encoding="utf-8")
    rf = make_read_file(tmp_path)

    result = await rf.handler(path="f.txt")

    assert result.is_error is False
    assert result.content == "hello\nworld\n"


async def test_relative_path_resolved_under_root(tmp_path: Path) -> None:
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text("x = 1\n", encoding="utf-8")
    rf = make_read_file(tmp_path)

    result = await rf.handler(path="pkg/mod.py")

    assert result.is_error is False
    assert result.content == "x = 1\n"


async def test_line_slice_returns_requested_lines(tmp_path: Path) -> None:
    (tmp_path / "lines.txt").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    rf = make_read_file(tmp_path)

    result = await rf.handler(path="lines.txt", offset=1, limit=2)

    assert result.is_error is False
    assert result.content == "b\nc\n"


async def test_line_slice_offset_only_returns_tail(tmp_path: Path) -> None:
    (tmp_path / "lines.txt").write_text("a\nb\nc\n", encoding="utf-8")
    rf = make_read_file(tmp_path)

    result = await rf.handler(path="lines.txt", offset=1)

    assert result.is_error is False
    assert result.content == "b\nc\n"


async def test_line_slice_clamps_out_of_range(tmp_path: Path) -> None:
    (tmp_path / "lines.txt").write_text("a\nb\n", encoding="utf-8")
    rf = make_read_file(tmp_path)

    result = await rf.handler(path="lines.txt", offset=10, limit=5)

    assert result.is_error is False
    assert result.content == ""


async def test_missing_file_returns_error_result(tmp_path: Path) -> None:
    rf = make_read_file(tmp_path)

    result = await rf.handler(path="nope.txt")

    assert result.is_error is True
    assert "nope.txt" in result.content


async def test_directory_target_returns_error_result(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    rf = make_read_file(tmp_path)

    result = await rf.handler(path="adir")

    assert result.is_error is True
    assert "regular file" in result.content


async def test_parent_traversal_escape_returns_error_not_raise(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "secret").write_text("top secret\n", encoding="utf-8")
    rf = make_read_file(workspace)

    result = await rf.handler(path="../secret")

    assert result.is_error is True
    assert "escapes workspace" in result.content


async def test_absolute_path_escape_returns_error_not_raise(tmp_path: Path) -> None:
    rf = make_read_file(tmp_path)

    result = await rf.handler(path="/etc/hostname")

    assert result.is_error is True
    assert "escapes workspace" in result.content


async def test_invalid_utf8_is_replaced_not_raised(tmp_path: Path) -> None:
    (tmp_path / "bin.txt").write_bytes(b"ok\xff\xfe")
    rf = make_read_file(tmp_path)

    result = await rf.handler(path="bin.txt")

    assert result.is_error is False
    assert result.content.startswith("ok")
