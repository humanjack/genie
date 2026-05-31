"""Tests for :class:`ToolResult` constructors and SPEC §5.4 layer-1 truncation."""

from __future__ import annotations

from genie.tools.result import ToolResult


def test_text_sets_content_and_clears_error() -> None:
    result = ToolResult.text("hello")
    assert result.content == "hello"
    assert result.is_error is False
    assert result.metadata == {}


def test_text_captures_metadata() -> None:
    result = ToolResult.text("hello", bytes=5, path="/tmp/x")
    assert result.metadata == {"bytes": 5, "path": "/tmp/x"}


def test_error_sets_error_flag_and_content() -> None:
    result = ToolResult.error("boom")
    assert result.content == "boom"
    assert result.is_error is True
    assert result.metadata == {}


def test_error_captures_metadata() -> None:
    result = ToolResult.error("boom", code=2)
    assert result.is_error is True
    assert result.metadata == {"code": 2}


def test_default_construction() -> None:
    result = ToolResult(content="x")
    assert result.is_error is False
    assert result.metadata == {}


def test_metadata_is_not_shared_between_instances() -> None:
    a = ToolResult.text("a")
    b = ToolResult.text("b")
    a.metadata["k"] = 1
    assert b.metadata == {}


def test_truncate_under_cap_returns_same_object() -> None:
    result = ToolResult.text("short")
    assert result.truncate(100) is result


def test_truncate_at_exact_cap_is_unchanged() -> None:
    content = "x" * 50
    result = ToolResult.text(content)
    truncated = result.truncate(50)
    assert truncated is result
    assert truncated.content == content


def test_truncate_over_cap_inserts_marker_with_count() -> None:
    content = "x" * 1000
    result = ToolResult.text(content)
    truncated = result.truncate(100)
    assert truncated is not result
    elided = 1000 - 100
    assert f"[truncated {elided} chars]" in truncated.content


def test_truncate_keeps_head_and_tail() -> None:
    content = "HEAD" + ("m" * 1000) + "TAIL"
    result = ToolResult.text(content)
    truncated = result.truncate(200)
    assert truncated.content.startswith("HEAD")
    assert truncated.content.endswith("TAIL")


def test_truncate_total_length_bounded_by_cap_plus_marker() -> None:
    content = "y" * 5000
    max_chars = 256
    truncated = ToolResult.text(content).truncate(max_chars)
    elided = 5000 - max_chars
    marker = f"\n…[truncated {elided} chars]…\n"
    assert len(truncated.content) == max_chars + len(marker)


def test_truncate_preserves_error_flag_and_copies_metadata() -> None:
    content = "z" * 100
    result = ToolResult.error(content, code=7)
    truncated = result.truncate(10)
    assert truncated.is_error is True
    assert truncated.metadata == {"code": 7}
    truncated.metadata["code"] = 0
    assert result.metadata == {"code": 7}


def test_truncate_odd_budget_splits_head_and_tail() -> None:
    content = "a" * 100
    truncated = ToolResult.text(content).truncate(7)
    head, _, tail = truncated.content.partition("\n")
    assert head == "aaa"
    assert tail.endswith("aaaa")
