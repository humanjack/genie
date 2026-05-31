"""Tests for :class:`ToolRegistry`: registration, per-provider specs, dispatch."""

from __future__ import annotations

import pytest

from genie.tools.base import Tool, tool
from genie.tools.registry import ToolRegistry
from genie.tools.result import ToolResult


@tool(name="read_file", tags=["fs"])
async def read_file(path: str, limit: int = 10) -> ToolResult:
    """Read a file."""
    return ToolResult.text(f"read {path} limit={limit}")


@tool(name="echo")
async def echo(text: str) -> ToolResult:
    """Echo text back."""
    return ToolResult.text(text)


@tool(name="tiny", max_result_chars=10)
async def tiny(payload: str) -> ToolResult:
    """Return whatever it is given (used to exercise truncation)."""
    return ToolResult.text(payload)


@tool(name="boom")
async def boom(x: str) -> ToolResult:
    """Always raises, to exercise the error contract."""
    raise RuntimeError(f"kaboom: {x}")


def _registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(tools)
    return registry


def test_register_get_contains_len_names() -> None:
    registry = ToolRegistry()
    assert len(registry) == 0
    registry.register(read_file)
    registry.register(echo)

    assert len(registry) == 2
    assert registry.get("read_file") is read_file
    assert "read_file" in registry
    assert "echo" in registry
    assert "missing" not in registry
    # names() follows registration order.
    assert registry.names() == ["read_file", "echo"]


def test_register_all_registers_each() -> None:
    registry = ToolRegistry()
    registry.register_all([read_file, echo, tiny])
    assert registry.names() == ["read_file", "echo", "tiny"]


def test_duplicate_registration_raises_value_error() -> None:
    registry = ToolRegistry()
    registry.register(read_file)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(read_file)


def test_register_all_duplicate_keeps_prior_and_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register_all([read_file, echo, read_file])
    # The tools registered before the collision remain registered.
    assert registry.names() == ["read_file", "echo"]


def test_get_unknown_raises_keyerror_listing_known() -> None:
    registry = _registry(read_file)
    with pytest.raises(KeyError) as exc_info:
        registry.get("nope")
    message = str(exc_info.value)
    assert "nope" in message
    assert "read_file" in message


def test_get_unknown_on_empty_registry_mentions_none() -> None:
    registry = ToolRegistry()
    with pytest.raises(KeyError, match="<none>"):
        registry.get("nope")


def test_specs_for_anthropic_shape() -> None:
    registry = _registry(read_file)
    specs = registry.specs_for("anthropic")
    assert specs == [
        {
            "name": "read_file",
            "description": "Read a file.",
            "input_schema": read_file.input_schema,
        }
    ]


def test_specs_for_openai_wraps_function_with_parameters() -> None:
    registry = _registry(read_file)
    specs = registry.specs_for("openai")
    assert len(specs) == 1
    spec = specs[0]
    assert spec["type"] == "function"
    function = spec["function"]
    assert function["name"] == "read_file"
    assert function["description"] == "Read a file."
    # The crux of the OpenAI translation: parameters IS the input_schema.
    assert function["parameters"] == read_file.input_schema


def test_specs_for_fake_uses_anthropic_passthrough() -> None:
    registry = _registry(read_file)
    assert registry.specs_for("fake") == registry.specs_for("anthropic")


def test_specs_for_preserves_registration_order() -> None:
    registry = _registry(read_file, echo)
    names = [spec["name"] for spec in registry.specs_for("anthropic")]
    assert names == ["read_file", "echo"]


def test_specs_for_empty_registry_is_empty_list() -> None:
    assert ToolRegistry().specs_for("openai") == []


def test_specs_for_unknown_provider_raises_listing_supported() -> None:
    registry = _registry(read_file)
    with pytest.raises(ValueError) as exc_info:
        registry.specs_for("gemini")
    message = str(exc_info.value)
    assert "gemini" in message
    # Lists every supported provider so the caller can self-correct.
    for supported in ("anthropic", "openai", "fake"):
        assert supported in message


async def test_call_awaits_handler_and_returns_result() -> None:
    registry = _registry(read_file)
    result = await registry.call("read_file", {"path": "a.txt", "limit": 5})
    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert result.content == "read a.txt limit=5"


async def test_call_truncates_to_tool_max_result_chars() -> None:
    registry = _registry(tiny)
    long_payload = "x" * 500
    result = await registry.call("tiny", {"payload": long_payload})
    # tiny declares max_result_chars=10, so the content is truncated and the
    # head/tail marker appears in place of the elided middle.
    assert "[truncated" in result.content
    assert len(result.content) < len(long_payload)


async def test_call_does_not_truncate_short_result() -> None:
    registry = _registry(tiny)
    result = await registry.call("tiny", {"payload": "short"})
    assert result.content == "short"
    assert "[truncated" not in result.content


async def test_call_wraps_handler_exception_as_error_result() -> None:
    registry = _registry(boom)
    result = await registry.call("boom", {"x": "boom"})
    assert result.is_error is True
    assert "kaboom: boom" in result.content


async def test_call_unknown_tool_raises_keyerror() -> None:
    registry = _registry(read_file)
    with pytest.raises(KeyError, match="missing"):
        await registry.call("missing", {})
