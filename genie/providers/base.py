"""Provider-agnostic LLM contract.

This module defines the single seam every model provider is reached through.
The contract deliberately exposes **no** provider-native types: messages are
plain dataclasses and tool definitions are JSON-Schema ``dict`` objects. Each
concrete :class:`ProviderClient` is responsible for translating to and from its
SDK's wire format, so the agent loop never learns who is on the other end.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import wraps
from typing import Any


def resolve_api_key(settings: object | None, provider_name: str, env_var: str) -> str:
    """Resolve ``provider_name``'s API key from settings or the environment.

    When ``settings`` exposes ``require_api_key`` (the
    :class:`~genie.config.Settings` contract) it is consulted with the live
    environment; otherwise ``env_var`` is read directly. Shared by the SDK
    adapters so credential resolution lives in one audited place.

    Raises:
        ValueError: If no key is found, naming the environment variable to set.
    """
    require = getattr(settings, "require_api_key", None)
    if require is not None:
        return require(provider_name, os.environ)
    key = os.environ.get(env_var)
    if not key:
        raise ValueError(
            f"Missing API key for provider {provider_name!r}: "
            f"set the {env_var} environment variable."
        )
    return key


@dataclass
class ChatMessage:
    """A single message in a conversation, normalized across providers.

    Attributes:
        role: One of ``"system"``, ``"user"``, ``"assistant"``, ``"tool"``.
        content: Either plain text or a list of provider-neutral content
            blocks (``list[dict]``).
        tool_calls: Tool-call requests emitted by an assistant turn, or
            ``None`` when the turn made no tool calls.
        tool_call_id: For ``role == "tool"`` results, the id of the call this
            message answers; ``None`` otherwise.
    """

    role: str
    content: str | list[dict]
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


@dataclass
class ChatChunk:
    """One streamed increment of a model response.

    A stream yields a sequence of these. Fields are independent: a chunk may
    carry text, a tool-call delta, a terminal ``finish_reason``, a ``usage``
    report, or any combination — unset fields are ``None``.

    Attributes:
        delta_text: Newly produced text, if any.
        tool_call_delta: One increment of a *single* tool call, addressed by an
            integer slot ``index`` so fragments of several parallel tool calls
            can be interleaved and reassembled. This mirrors both OpenAI
            (``ChoiceDeltaToolCall.index`` + streamed ``function.arguments``
            string) and Anthropic (``content_block`` index + ``input_json_delta``
            ``partial_json`` string). Shape::

                {
                    "index": int,                 # required — which tool-call slot
                    "id": str | None,             # set once, on the slot's first fragment
                    "name": str | None,           # set once, on the slot's first fragment
                    "arguments_delta": str | None # partial JSON to append for this slot
                }

            The consumer accumulates ``arguments_delta`` per ``index`` and
            ``json.loads`` the joined string when the turn finishes. Arguments
            are **never** delivered as a pre-parsed dict — that would not
            survive real streaming.
        finish_reason: Terminal reason for the turn (e.g. ``"stop"``,
            ``"tool_calls"``), set on the final chunk of a turn.
        usage: Token accounting for the turn with keys ``input_tokens``,
            ``output_tokens``, ``cache_read``, ``cache_write``. Delivered on the
            turn's terminal chunk.
    """

    delta_text: str | None = None
    tool_call_delta: dict | None = None
    finish_reason: str | None = None
    usage: dict | None = None


class ProviderClient(ABC):
    """Abstract base every LLM provider implementation must satisfy.

    Concrete subclasses set :attr:`name` and :attr:`model` and translate the
    provider-neutral arguments below into their SDK's calls. The loop depends
    only on this interface, which is what makes providers swappable with zero
    edits to the loop (see SPEC operating principle "Pluggable everywhere").
    """

    # Checked after construction: class attributes, properties, and attributes
    # assigned in __init__ are all supported, without requiring super().__init__.
    name: str
    model: str

    def _validate_identity(self) -> None:
        for field in ("name", "model"):
            value = getattr(self, field, None)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{type(self).__name__}.{field} must be a nonempty string")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if type(self).__init__ is ProviderClient.__init__:
            self._validate_identity()

    def __post_init__(self, *_init_vars: Any) -> None:
        """Validate dataclass providers after their generated initializer runs."""
        if type(self).__post_init__ is ProviderClient.__post_init__:
            self._validate_identity()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate completed initializers without imposing another metaclass."""
        super().__init_subclass__(**kwargs)

        def wrap_initializer(method_name: str, initializer: Any) -> Any:
            @wraps(initializer)
            def checked(self: ProviderClient, *args: Any, **kwargs: Any) -> None:
                initializer(self, *args, **kwargs)
                # Children may assign identity after super() returns. Only the
                # outermost initializer validates, including inherited methods.
                if getattr(type(self), method_name) is checked:
                    self._validate_identity()

            return checked

        for method_name in ("__init__", "__post_init__"):
            # Leave inherited constructors untouched: dataclass decorators need
            # to generate __init__ when the class does not declare one itself.
            if method_name in cls.__dict__:
                setattr(cls, method_name, wrap_initializer(method_name, cls.__dict__[method_name]))

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system: str | None = None,
        cache_breakpoints: list[int] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Stream a model response as a sequence of :class:`ChatChunk`.

        Args:
            messages: Conversation history as provider-neutral messages.
            tools: Tool definitions as JSON-Schema ``dict`` objects; the
                implementation translates these to its provider's tool shape.
            max_tokens: Upper bound on generated tokens.
            temperature: Sampling temperature.
            system: Optional system prompt, surfaced however the provider
                expects (top-level param or leading message).
            cache_breakpoints: Indexes into ``messages`` at which to request
                prompt caching, when the provider supports it; otherwise a
                no-op.

        Yields:
            :class:`ChatChunk` increments until the turn terminates.
        """
        raise NotImplementedError
        # Make this an async generator for type checkers; never reached.
        yield ChatChunk()  # pragma: no cover

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        """Estimate the token count for ``messages`` (chars // 4, minimum 1).

        A deterministic, offline heuristic — not a real tokenization — shared
        by providers unless overridden. This synchronous API stays offline;
        use :meth:`count_tokens_async` for SDK-backed counting when supported.
        """
        chars = sum(len(str(m.content)) for m in messages)
        return max(1, chars // 4)

    async def count_tokens_async(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> int:
        """Count input tokens, using the provider SDK when available.

        The default calls the offline :meth:`count_tokens` estimator and adds
        rough estimates for tool definitions and the system prompt. Fake and
        OpenAI providers currently use this fallback. Providers with a counting
        endpoint override this method and may require credentials and network
        access; their errors propagate to the caller. Counts exclude generated
        output and need not match the eventual response usage exactly.
        """
        extra_chars = len(system or "") + (len(str(tools)) if tools else 0)
        return self.count_tokens(messages) + extra_chars // 4
