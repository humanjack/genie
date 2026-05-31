"""The tool registry: register tools, translate specs per provider, dispatch a call.

A :class:`ToolRegistry` is the loop's single source of truth for the tools a
model may call. It does three things and nothing more:

1. **Holds** tools by name (registration is the only mutation).
2. **Exposes** the held tools as provider-neutral specs via
   :meth:`ToolRegistry.specs_for` — the loop hands the result straight to
   ``provider.stream(..., tools=...)``.
3. **Dispatches** a single named call via :meth:`ToolRegistry.call`, awaiting
   the tool's handler and applying *layer 1* truncation (SPEC §5.4) to the
   result.

**Where translation lives.** Each provider adapter translates the neutral spec
into its own wire format *inside* ``stream`` (Anthropic passes the neutral shape
through unchanged; OpenAI wraps it as a ``function`` tool — see
``genie/providers/*_client.py``). So :meth:`specs_for` returns the
lowest-common-denominator ``{name, description, input_schema}`` shape for **every**
provider; pre-wrapping here would double-translate and corrupt the spec. The
per-provider mapping (:data:`_TRANSLATORS`) is kept as the seam for a future
provider that genuinely needs registry-side shaping.

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


def _spec_neutral(tool: Tool) -> dict:
    """The provider-neutral tool spec: ``{name, description, input_schema}``.

    This is the lowest common denominator the loop passes to ``stream`` for any
    provider; each adapter shapes it to its own wire format internally.
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


# The extension seam: one entry per provider. Every provider currently uses the
# neutral spec because the adapters own wire-format translation in ``stream``
# (pre-wrapping here would double-translate). A future provider needing
# registry-side shaping adds its own translator entry.
_TRANSLATORS: dict[str, Callable[[Tool], dict]] = {
    "anthropic": _spec_neutral,
    "openai": _spec_neutral,
    "fake": _spec_neutral,
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
        """Return the neutral tool specs to pass to ``provider_name``'s ``stream``.

        The result is ready to hand straight to that provider's
        :meth:`~genie.providers.base.ProviderClient.stream` as ``tools=``; the
        adapter shapes it to its own wire format internally (so the same neutral
        ``{name, description, input_schema}`` list is returned for every
        supported provider — see the module docstring on why pre-wrapping here
        would double-translate). Order follows registration order.

        Args:
            provider_name: The provider the specs are destined for. Validated
                against the supported set; currently does not change the shape.

        Returns:
            A list of neutral tool ``dict`` objects, one per tool.

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
