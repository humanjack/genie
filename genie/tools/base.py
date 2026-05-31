"""Tool definition and the ``@tool`` decorator.

A :class:`Tool` is the registry's unit of currency: a name, a model-facing
description, a JSON-Schema description of its inputs, and an async handler that
returns a :class:`ToolResult`. The :func:`tool` decorator builds one from an
ordinary async function, deriving ``input_schema`` from the function signature
via pydantic so the schema and the implementation can never drift apart.

The design is *strict on input, lenient on output*: a parameter without a type
annotation is an error at decoration time (explicit over magical), but a
handler that returns a bare ``str`` is coerced to ``ToolResult.text`` for
convenience.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from genie.tools.result import ToolResult


class Tool(BaseModel):
    """A callable capability exposed to the model.

    Attributes:
        name: Stable identifier the model uses to invoke the tool.
        description: Natural-language description the model reads to decide
            when and how to call the tool.
        input_schema: JSON Schema (object type) describing the tool's
            arguments — the lowest common denominator across providers.
        handler: The async function that executes the call and returns a
            :class:`ToolResult`.
        sequential: When ``True``, a batch containing this tool runs serially
            rather than in parallel (Pi pattern; see SPEC §5.3).
        dangerous: When ``True``, the call is short-circuited to the approval
            hook before execution (SPEC §7.3).
        tags: Free-form labels for filtering, e.g. ``["fs", "net"]``.
        max_result_chars: Per-tool truncation budget applied to the result's
            content (SPEC §5.4 layer 1).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Awaitable[ToolResult]]
    sequential: bool = False
    dangerous: bool = False
    tags: list[str] = []
    max_result_chars: int = 8192


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    sequential: bool = False,
    dangerous: bool = False,
    tags: Sequence[str] = (),
    max_result_chars: int = 8192,
) -> Callable[[Callable[..., Awaitable[Any]]], Tool]:
    """Wrap an async function as a :class:`Tool`.

    The decorated function's signature is inspected to build ``input_schema``:
    each parameter contributes a field (its annotation as the type, its default
    as the field default, required when it has no default). Every parameter
    **must** carry a type annotation; the lowest common denominator schema is
    produced with :func:`pydantic.create_model` and its top-level ``title`` is
    stripped for cleanliness.

    Args:
        name: Tool name; defaults to the function's ``__name__``.
        description: Model-facing description; defaults to the function's
            docstring. One of the two must be present.
        sequential: Sets :attr:`Tool.sequential`.
        dangerous: Sets :attr:`Tool.dangerous`.
        tags: Sets :attr:`Tool.tags`.
        max_result_chars: Sets :attr:`Tool.max_result_chars`.

    Returns:
        A decorator turning an async function into a :class:`Tool`.

    Raises:
        TypeError: If the decorated function is not a coroutine function.
        TypeError: If any parameter lacks a type annotation.
        ValueError: If no description is available (neither argument nor
            docstring).
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> Tool:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                f"@tool requires an async function; {func.__name__!r} is not a coroutine function"
            )

        resolved_name = name or func.__name__
        resolved_description = description or inspect.getdoc(func)
        if not resolved_description:
            raise ValueError(
                f"@tool requires a description for {resolved_name!r}: pass description= "
                "or give the function a docstring"
            )

        input_schema = _schema_from_signature(func, resolved_name)

        @functools.wraps(func)
        async def handler(**kwargs: Any) -> ToolResult:
            result = await func(**kwargs)
            if isinstance(result, ToolResult):
                return result
            if isinstance(result, str):
                return ToolResult.text(result)
            raise TypeError(
                f"tool {resolved_name!r} handler must return ToolResult or str, "
                f"got {type(result).__name__}"
            )

        return Tool(
            name=resolved_name,
            description=resolved_description,
            input_schema=input_schema,
            handler=handler,
            sequential=sequential,
            dangerous=dangerous,
            tags=list(tags),
            max_result_chars=max_result_chars,
        )

    return decorator


def _schema_from_signature(func: Callable[..., Any], tool_name: str) -> dict:
    """Derive an object-type JSON Schema from ``func``'s parameters.

    Args:
        func: The function whose signature defines the tool's inputs.
        tool_name: Name used in error messages for missing annotations.

    Returns:
        A JSON Schema ``dict`` (object type) with the top-level ``title``
        removed.

    Raises:
        TypeError: If any parameter lacks a type annotation, uses ``*args`` /
            ``**kwargs``, or has an annotation that cannot be resolved.
    """
    signature = inspect.signature(func)
    # Resolve stringized annotations (PEP 563 / ``from __future__ import
    # annotations``) to real types so unions like ``int | None`` build a valid
    # schema instead of crashing create_model on an unresolved forward ref.
    try:
        hints = get_type_hints(func)
    except Exception as exc:
        # Surface any annotation-resolution failure as a clear decoration error.
        raise TypeError(f"@tool could not resolve type hints for {tool_name!r}: {exc}") from exc
    hints.pop("return", None)

    fields: dict[str, Any] = {}
    for param_name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(
                f"@tool does not support *args/**kwargs; parameter {param_name!r} on "
                f"{tool_name!r} must be an explicit named parameter"
            )
        if param_name not in hints:
            raise TypeError(
                f"@tool parameter {param_name!r} on {tool_name!r} lacks a type annotation; "
                "annotate every parameter explicitly"
            )
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (hints[param_name], default)

    args_model = create_model(f"{tool_name}_Args", **fields)
    schema = args_model.model_json_schema()
    schema.pop("title", None)
    return schema
