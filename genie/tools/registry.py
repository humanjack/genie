"""The tool registry: register tools, translate specs per provider, dispatch a call.

A :class:`ToolRegistry` is the loop's single source of truth for the tools a
model may call. It does three things and nothing more:

1. **Holds** tools by name (registration is the only mutation).
2. **Translates** the held tools into each provider's native tool shape via
   :meth:`ToolRegistry.specs_for` — the loop hands the result straight to
   ``provider.stream(..., tools=...)``.
3. **Dispatches** a single named call via :meth:`ToolRegistry.call`, awaiting
   the tool's handler and applying *layer 1* truncation (SPEC §5.4) to the
   result.

Provider translation lives in a small per-provider mapping (:data:`_TRANSLATORS`)
so adding a provider is one entry — the extension seam the SPEC asks for.

**Error contract.** :meth:`call` catches any ``Exception`` raised by a handler
and returns ``ToolResult.error(str(exc))`` rather than propagating. The loop's
dispatcher (SPEC §5.3) also wraps handler errors as tool-result errors, but the
registry owning this guarantee means one misbehaving tool can never crash a
parallel batch: every call resolves to a :class:`ToolResult`, never an
exception. Lookup failures in :meth:`get` (``KeyError``) and unknown providers
in :meth:`specs_for` (``ValueError``) are *programming* errors and do propagate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from genie.tools.base import Tool
from genie.tools.result import ToolResult


def _spec_anthropic(tool: Tool) -> dict:
    """Anthropic tool shape: ``{name, description, input_schema}`` (passthrough)."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _spec_openai(tool: Tool) -> dict:
    """OpenAI tool shape: a ``function`` wrapper whose ``parameters`` is the schema."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


# The extension seam: one entry per provider. ``fake`` reuses the Anthropic-style
# passthrough — it is the simplest shape and the FakeProvider does not inspect
# tool structure, so the lowest-common-denominator form is the right default.
_TRANSLATORS: dict[str, Callable[[Tool], dict]] = {
    "anthropic": _spec_anthropic,
    "openai": _spec_openai,
    "fake": _spec_anthropic,
}


class ToolRegistry:
    """A name-keyed collection of :class:`Tool` objects with per-provider specs.

    The registry is mutated only by registration; lookups, membership, length,
    spec translation, and dispatch are all read-only over the held tools.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add ``tool`` to the registry, keyed by its name.

        Args:
            tool: The tool to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
                Names are the model's handle for a tool, so a silent overwrite
                would let one tool shadow another; we fail loudly instead.
        """
        if tool.name in self._tools:
            raise ValueError(f"a tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def register_all(self, tools: Iterable[Tool]) -> None:
        """Register every tool in ``tools`` in order.

        A duplicate name anywhere in ``tools`` raises as in :meth:`register`;
        tools registered before the offending one remain registered.

        Args:
            tools: An iterable of tools to register.

        Raises:
            ValueError: If any tool's name collides with an existing one.
        """
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        """Return the registered tool named ``name``.

        Args:
            name: The tool name to look up.

        Returns:
            The matching :class:`Tool`.

        Raises:
            KeyError: If no tool with that name is registered. The message
                lists the known names to make a typo obvious.
        """
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(sorted(self._tools)) or "<none>"
            raise KeyError(f"no tool named {name!r}; registered tools: {known}") from None

    def names(self) -> list[str]:
        """Return the registered tool names in registration order."""
        return list(self._tools)

    def __contains__(self, name: object) -> bool:
        """Return whether a tool named ``name`` is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def specs_for(self, provider_name: str) -> list[dict]:
        """Translate every registered tool to ``provider_name``'s native shape.

        The result is ready to pass as the ``tools`` argument to that provider's
        :meth:`~genie.providers.base.ProviderClient.stream`. Order follows
        registration order.

        Supported providers and their shapes:

        - ``"anthropic"`` → ``{"name", "description", "input_schema"}`` (passthrough).
        - ``"openai"`` → ``{"type": "function", "function": {"name", "description",
          "parameters"}}`` where ``parameters`` is the tool's ``input_schema``.
        - ``"fake"`` → the Anthropic-style passthrough (simplest shape).

        Args:
            provider_name: The provider whose tool shape to produce.

        Returns:
            A list of provider-native tool ``dict`` objects, one per tool.

        Raises:
            ValueError: If ``provider_name`` is not a supported provider; the
                message lists the supported names.
        """
        try:
            translate = _TRANSLATORS[provider_name]
        except KeyError:
            supported = ", ".join(sorted(_TRANSLATORS))
            raise ValueError(
                f"unknown provider {provider_name!r}; supported providers: {supported}"
            ) from None
        return [translate(tool) for tool in self._tools.values()]

    async def call(self, name: str, args: dict) -> ToolResult:
        """Look up ``name``, run its handler with ``args``, and truncate the result.

        The handler is awaited as ``handler(**args)`` and its :class:`ToolResult`
        is passed through ``result.truncate(tool.max_result_chars)`` (SPEC §5.4
        layer 1) before returning.

        Per the registry's error contract (see module docstring), any
        ``Exception`` raised by the handler is caught and returned as
        ``ToolResult.error(str(exc))`` so a single failing tool cannot crash a
        batch. A missing tool is a *programming* error and still raises
        ``KeyError`` from :meth:`get`.

        Args:
            name: The name of the tool to invoke.
            args: Keyword arguments for the handler.

        Returns:
            The (possibly truncated) :class:`ToolResult`; an error result if the
            handler raised.

        Raises:
            KeyError: If no tool named ``name`` is registered.
        """
        tool = self.get(name)
        try:
            result = await tool.handler(**args)
        except Exception as exc:
            result = ToolResult.error(str(exc))
        return result.truncate(tool.max_result_chars)
