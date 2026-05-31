"""Tests for the :class:`Tool` model and the ``@tool`` decorator."""

from __future__ import annotations

import pytest

from genie.tools.base import Tool, tool
from genie.tools.result import ToolResult


def test_decorator_returns_tool_instance() -> None:
    @tool()
    async def sample(path: str) -> ToolResult:
        """Read a file."""
        return ToolResult.text(path)

    assert isinstance(sample, Tool)


def test_input_schema_is_object_with_properties_and_required() -> None:
    @tool()
    async def read_file(path: str, limit: int = 10) -> ToolResult:
        """Read a file."""
        return ToolResult.text(path)

    schema = read_file.input_schema
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"path", "limit"}
    assert schema["properties"]["path"]["type"] == "string"
    assert schema["properties"]["limit"]["type"] == "integer"
    assert schema["properties"]["limit"]["default"] == 10
    assert schema["required"] == ["path"]


def test_input_schema_has_no_top_level_title() -> None:
    @tool()
    async def read_file(path: str) -> ToolResult:
        """Read a file."""
        return ToolResult.text(path)

    assert "title" not in read_file.input_schema


def test_name_defaults_to_function_name() -> None:
    @tool()
    async def my_tool(x: str) -> ToolResult:
        """Docs."""
        return ToolResult.text(x)

    assert my_tool.name == "my_tool"


def test_explicit_name_overrides() -> None:
    @tool(name="renamed")
    async def my_tool(x: str) -> ToolResult:
        """Docs."""
        return ToolResult.text(x)

    assert my_tool.name == "renamed"


def test_description_defaults_to_docstring() -> None:
    @tool()
    async def my_tool(x: str) -> ToolResult:
        """First line of docs."""
        return ToolResult.text(x)

    assert my_tool.description == "First line of docs."


def test_explicit_description_overrides_docstring() -> None:
    @tool(description="explicit")
    async def my_tool(x: str) -> ToolResult:
        """ignored docstring."""
        return ToolResult.text(x)

    assert my_tool.description == "explicit"


def test_missing_description_raises() -> None:
    with pytest.raises(ValueError, match="description"):

        @tool()
        async def my_tool(x: str) -> ToolResult:
            return ToolResult.text(x)


def test_missing_annotation_raises_at_decoration() -> None:
    with pytest.raises(TypeError, match="annotation"):

        @tool()
        async def my_tool(x) -> ToolResult:  # type: ignore[no-untyped-def]
            """Docs."""
            return ToolResult.text(str(x))


def test_non_async_function_raises() -> None:
    with pytest.raises(TypeError, match="coroutine"):

        @tool()  # pyright: ignore[reportArgumentType]
        def my_tool(x: str) -> ToolResult:
            """Docs."""
            return ToolResult.text(x)


def test_flags_carried_onto_tool() -> None:
    @tool(
        sequential=True,
        dangerous=True,
        tags=["fs", "net"],
        max_result_chars=256,
    )
    async def my_tool(x: str) -> ToolResult:
        """Docs."""
        return ToolResult.text(x)

    assert my_tool.sequential is True
    assert my_tool.dangerous is True
    assert my_tool.tags == ["fs", "net"]
    assert my_tool.max_result_chars == 256


def test_default_flags() -> None:
    @tool()
    async def my_tool(x: str) -> ToolResult:
        """Docs."""
        return ToolResult.text(x)

    assert my_tool.sequential is False
    assert my_tool.dangerous is False
    assert my_tool.tags == []
    assert my_tool.max_result_chars == 8192


async def test_handler_returns_awaited_tool_result() -> None:
    @tool()
    async def my_tool(path: str) -> ToolResult:
        """Docs."""
        return ToolResult.text(f"read {path}")

    result = await my_tool.handler(path="a.txt")
    assert isinstance(result, ToolResult)
    assert result.content == "read a.txt"
    assert result.is_error is False


async def test_handler_coerces_bare_str_to_text_result() -> None:
    @tool()
    async def my_tool(x: str) -> str:
        """Docs."""
        return f"value={x}"

    result = await my_tool.handler(x="hi")
    assert isinstance(result, ToolResult)
    assert result.content == "value=hi"
    assert result.is_error is False


async def test_handler_passes_through_error_result() -> None:
    @tool()
    async def my_tool(x: str) -> ToolResult:
        """Docs."""
        return ToolResult.error("nope")

    result = await my_tool.handler(x="hi")
    assert result.is_error is True
    assert result.content == "nope"


def test_no_args_tool_has_empty_properties() -> None:
    @tool()
    async def ping() -> ToolResult:
        """Ping."""
        return ToolResult.text("pong")

    assert ping.input_schema["type"] == "object"
    assert ping.input_schema.get("properties", {}) == {}
    assert "required" not in ping.input_schema
