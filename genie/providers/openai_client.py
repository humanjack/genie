"""OpenAI-backed :class:`ProviderClient` (SPEC §3.3).

Concrete adapter that translates the provider-neutral contract in
:mod:`genie.providers.base` to and from the OpenAI Python SDK's wire format.
Only the **chat completions** API is implemented in this PR; the SDK's newer
``responses`` API is a deliberately deferred branch (see :meth:`stream`).

The OpenAI streaming surface maps almost one-to-one onto the contract's
index-addressed :class:`ChatChunk` tool-call shape: ``ChoiceDeltaToolCall``
already carries ``.index``, ``.id`` (set once, on the opening fragment) and a
``.function.arguments`` *string fragment*. The only quirk handled here is that
the ``finish_reason`` chunk and the usage-bearing chunk are typically
**separate** events — the usage chunk arrives last with ``choices == []`` — so
usage is emitted on its own terminal :class:`ChatChunk` when it lands.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from genie.providers.base import ChatChunk, ChatMessage, ProviderClient

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def _translate_tools(tools: list[dict]) -> list[dict] | None:
    """Wrap neutral JSON-Schema tool dicts as OpenAI ``function`` tools.

    Each input tool is ``{"name", "description", "input_schema"}``; OpenAI wants
    ``{"type": "function", "function": {"name", "description", "parameters"}}``
    where ``parameters`` is the JSON Schema. Returns ``None`` for an empty list
    so the SDK call can omit the ``tools`` argument entirely.
    """
    if not tools:
        return None
    wrapped: list[dict] = []
    for tool in tools:
        function: dict[str, Any] = {"name": tool["name"]}
        if tool.get("description") is not None:
            function["description"] = tool["description"]
        function["parameters"] = tool.get("input_schema") or {}
        wrapped.append({"type": "function", "function": function})
    return wrapped


def _translate_tool_call(call: dict) -> dict:
    """Map one neutral tool-call to OpenAI's native function-call shape.

    The neutral shape (what the loop appends to history) is
    ``{"id", "name", "arguments": <dict>}``; OpenAI requires
    ``{"id", "type": "function", "function": {"name", "arguments": <JSON string>}}``
    with the arguments JSON-*encoded*. Already-native calls (carrying a
    ``"function"`` key) are passed through unchanged for robustness.
    """
    if "function" in call:
        return call
    arguments = call.get("arguments", {})
    arguments_str = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": call.get("id"),
        "type": "function",
        "function": {"name": call.get("name"), "arguments": arguments_str},
    }


def _translate_messages(messages: list[ChatMessage], system: str | None) -> list[dict]:
    """Translate neutral messages to OpenAI chat-completions message dicts.

    ``system`` (which chat completions has no top-level slot for) becomes a
    leading ``{"role": "system"}`` message. Assistant ``tool_calls`` are mapped
    from the neutral shape to OpenAI's native function-call array (see
    :func:`_translate_tool_call`), and ``role == "tool"`` results carry their
    ``tool_call_id`` so the model can correlate the answer to the request.
    """
    out: list[dict] = []
    if system is not None:
        out.append({"role": "system", "content": system})
    for msg in messages:
        if msg.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                }
            )
            continue
        entry: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            entry["tool_calls"] = [_translate_tool_call(c) for c in msg.tool_calls]
        out.append(entry)
    return out


class OpenAIClient(ProviderClient):
    """Provider client backed by the OpenAI SDK's chat completions API.

    The OpenAI client is built lazily on the first :meth:`stream` call so no API
    key is required at construction time; an explicit ``client=`` may be injected
    for tests. The ``api`` mode is read from ``settings.provider.openai.api`` when
    settings are supplied, defaulting to ``"chat_completions"`` otherwise; the
    ``"responses"`` mode is deferred (raises :class:`NotImplementedError`).
    """

    name = "openai"

    def __init__(
        self,
        model: str,
        settings: object | None = None,
        *,
        client: Any | None = None,
        api: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Build an OpenAI client.

        Args:
            model: The OpenAI model identifier (e.g. ``"gpt-4o-mini"``).
            settings: Optional ``Settings`` object used to resolve the API key
                and the ``api`` mode; no key is required at construction time.
            client: An injected ``AsyncOpenAI`` (or duck-compatible) instance,
                used by tests to avoid the network. When ``None`` a real client
                is built lazily on first :meth:`stream` use.
            api: Explicit API mode override (``"chat_completions"`` or
                ``"responses"``); falls back to settings, then
                ``"chat_completions"``.
            **kwargs: Reserved for forward-compatibility; ignored.
        """
        self.model = model
        self._settings = settings
        self._client: Any | None = client
        self._api = api or self._resolve_api(settings)

    @staticmethod
    def _resolve_api(settings: object | None) -> str:
        """Return the configured API mode, defaulting to ``"chat_completions"``.

        Reads ``settings.provider.openai.api`` when a settings object exposing
        that path is supplied; otherwise defaults to ``"chat_completions"``.
        """
        if settings is None:
            return "chat_completions"
        provider = getattr(settings, "provider", None)
        openai_cfg = getattr(provider, "openai", None)
        return getattr(openai_cfg, "api", None) or "chat_completions"

    def _ensure_client(self) -> AsyncOpenAI:
        """Return the OpenAI client, building it lazily on first use.

        The API key is resolved via ``settings.require_api_key("openai", env)``
        when settings are present, else from the ``OPENAI_API_KEY`` environment
        variable. Building is deferred so construction never needs a key.
        """
        if self._client is None:
            from openai import AsyncOpenAI

            require_api_key = getattr(self._settings, "require_api_key", None)
            if require_api_key is not None:
                api_key = require_api_key("openai", os.environ)
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "Missing API key for provider 'openai': "
                        "set the OPENAI_API_KEY environment variable."
                    )
            self._client = AsyncOpenAI(api_key=api_key)
        return self._client

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
        """Stream an OpenAI chat-completions response as :class:`ChatChunk`.

        Translates ``messages``/``tools``/``system`` to OpenAI's shapes and
        async-iterates the SDK stream, mapping each event to the neutral
        contract. ``cache_breakpoints`` is a no-op: OpenAI prompt caching is
        server-managed and not requestable per-message.

        Raises:
            NotImplementedError: If the ``responses`` API mode is configured
                (``provider.openai.api = "responses"``); only
                ``"chat_completions"`` is implemented in this PR.
        """
        if self._api == "responses":
            raise NotImplementedError(
                "OpenAI 'responses' API mode is not implemented yet; set "
                "provider.openai.api = 'chat_completions' (the supported mode)."
            )
        async for chunk in self._stream_chat_completions(
            messages,
            tools,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        ):
            yield chunk

    async def _stream_chat_completions(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> AsyncIterator[ChatChunk]:
        """Drive the chat-completions streaming endpoint and map its events."""
        client = self._ensure_client()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": _translate_messages(messages, system),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        wrapped_tools = _translate_tools(tools)
        if wrapped_tools is not None:
            request["tools"] = wrapped_tools

        stream = await client.chat.completions.create(**request)
        async for chunk in stream:
            for mapped in _map_chunk(chunk):
                yield mapped

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        """Estimate the token count for ``messages`` (chars // 4, min 1).

        This is a deterministic local heuristic, not a precise tokenization:
        precise counting via ``tiktoken`` is deferred to avoid the dependency
        (issue #47). The estimate is intentionally cheap and offline.
        """
        chars = sum(len(str(m.content)) for m in messages)
        return max(1, chars // 4)


def _map_chunk(chunk: Any) -> list[ChatChunk]:
    """Map one OpenAI ``ChatCompletionChunk`` to zero or more :class:`ChatChunk`.

    A single OpenAI chunk may carry text, tool-call deltas, and/or a
    ``finish_reason`` on its first choice. The usage-bearing final chunk arrives
    separately with ``choices == []`` and its ``usage`` is surfaced on its own
    terminal :class:`ChatChunk`.
    """
    out: list[ChatChunk] = []
    choices = getattr(chunk, "choices", None) or []
    if choices:
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            content = getattr(delta, "content", None)
            if content:
                out.append(ChatChunk(delta_text=content))
            for tool_call in getattr(delta, "tool_calls", None) or []:
                out.append(ChatChunk(tool_call_delta=_map_tool_call_delta(tool_call)))
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            out.append(ChatChunk(finish_reason=_map_finish_reason(finish_reason)))

    usage = getattr(chunk, "usage", None)
    if usage is not None:
        out.append(
            ChatChunk(
                usage={
                    "input_tokens": getattr(usage, "prompt_tokens", 0),
                    "output_tokens": getattr(usage, "completion_tokens", 0),
                }
            )
        )
    return out


def _map_tool_call_delta(tool_call: Any) -> dict:
    """Map a ``ChoiceDeltaToolCall`` to the contract's index-addressed dict.

    ``id`` and ``name`` are included only when present (they appear once, on the
    slot's opening fragment); ``arguments`` is passed through verbatim as the
    partial-JSON ``arguments_delta`` string. The integer ``index`` selects the
    tool-call slot so parallel calls reassemble independently.
    """
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", None) if function is not None else None
    arguments = getattr(function, "arguments", None) if function is not None else None
    delta: dict[str, Any] = {"index": tool_call.index, "arguments_delta": arguments}
    if tool_call.id is not None:
        delta["id"] = tool_call.id
    if name is not None:
        delta["name"] = name
    return delta


def _map_finish_reason(finish_reason: str) -> str:
    """Return the contract finish reason for an OpenAI ``finish_reason``.

    OpenAI's values (``"stop"``, ``"tool_calls"``, ``"length"``,
    ``"content_filter"``) already match the contract's vocabulary, so this is a
    pass-through kept as a seam for any future remapping.
    """
    return finish_reason
