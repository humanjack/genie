"""Tool subsystem: the ``@tool`` decorator, the :class:`Tool` model, and :class:`ToolResult`."""

from __future__ import annotations

from genie.tools.base import Tool, tool
from genie.tools.registry import ToolRegistry
from genie.tools.result import ToolResult

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "tool",
]
