"""OpenAI-backed :class:`ProviderClient` (SPEC §3.3).

Concrete adapter that translates the provider-neutral contract in
:mod:`genie.providers.base` to and from the OpenAI Python SDK's wire format.
Both Chat Completions and Responses are supported. Chat Completions remains
the default for compatibility; Responses opts into stored server-side state.

The OpenAI streaming surface maps almost one-to-one onto the contract's
index-addressed :class:`ChatChunk` tool-call shape: ``ChoiceDeltaToolCall``
already carries ``.index``, ``.id`` (set once, on the opening fragment) and a
``.function.arguments`` *string fragment*. The only quirk handled here is that
the ``finish_reason`` chunk and the usage-bearing chunk are typically
**separate** events — the usage chunk arrives last with ``choices == []`` — so
usage is emitted on its own terminal :class:`ChatChunk` when it lands.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from genie.providers.base import ChatChunk, ChatMessage, ProviderClient, resolve_api_key

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


def _response_content(content: str | list[dict]) -> str | list[dict]:
    """Convert shared text and Chat-style image blocks to Responses content."""
    if isinstance(content, str):
        return content
    blocks = deepcopy(content)
    for block in blocks:
        if block.get("type") == "text":
            block["type"] = "input_text"
        elif block.get("type") == "image_url":
            image = block["image_url"]
            block.update(type="input_image", image_url=image["url"])
            if "detail" in image:
                block["detail"] = image["detail"]
    return blocks


def _translate_response_messages(
    messages: list[ChatMessage], *, use_provider_data: bool = True
) -> list[dict]:
    """Translate history to Responses input, canonicalizing function arguments."""
    out: list[dict] = []
    for message in messages:
        if use_provider_data and message.role == "assistant":
            saved = (message.provider_data or {}).get("openai.responses", {})
            if saved.get("fingerprint") == _response_fingerprint(message):
                out.extend(deepcopy(saved["output"]))
                continue
        if message.role == "tool":
            out.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": _response_content(message.content),
                }
            )
            continue
        if message.content or not message.tool_calls:
            out.append({"role": message.role, "content": _response_content(message.content)})
        for call in message.tool_calls or []:
            native = _translate_tool_call(call)
            function = native["function"]
            arguments = function.get("arguments", "{}")
            # Preserve malformed arguments for provider validation.
            with suppress(ValueError, TypeError):
                arguments = json.dumps(json.loads(arguments), sort_keys=True)
            out.append(
                {
                    "type": "function_call",
                    "call_id": native["id"],
                    "name": function["name"],
                    "arguments": arguments,
                }
            )
    return out


def _response_fingerprint(message: ChatMessage) -> str:
    """Bind opaque output to the editable, provider-neutral assistant message."""
    neutral = _translate_response_messages([message], use_provider_data=False)
    encoded = json.dumps(neutral, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _response_provider_data(response: Any, assistant: ChatMessage) -> dict | None:
    """Keep completed native items for full-history replay, including encrypted reasoning.

    SDK objects never leave this adapter. Plaintext reasoning is neither
    requested nor retained; the provider's opaque encrypted content is enough
    to continue. Preserve output order, tool item IDs, and assistant phase.
    """
    output = []
    for item in getattr(response, "output", None) or []:
        native = item.model_dump(mode="json", exclude_none=True)
        if native["type"] == "reasoning":
            native = {
                key: value
                for key, value in native.items()
                if key in ("type", "id", "encrypted_content", "status")
            }
            native["summary"] = []
        output.append(native)
    if not output:
        return None
    return {"openai.responses": {"output": output, "fingerprint": _response_fingerprint(assistant)}}


class OpenAIClient(ProviderClient):
    """Provider client backed by the OpenAI SDK's configurable APIs.

    The OpenAI client is built lazily on the first :meth:`stream` call so no API
    key is required at construction time; an explicit ``client=`` may be injected
    for tests. The ``api`` mode is read from ``settings.provider.openai.api`` when
    settings are supplied, defaulting to ``"chat_completions"`` otherwise; the
    ``"responses"`` mode reuses successful server-side responses when history matches.
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
        self._response_id: str | None = None
        self._response_history: list[dict] = []
        self._response_active = False

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

        Building is deferred so construction never needs a key; see
        :func:`~genie.providers.base.resolve_api_key` for how the key resolves.
        """
        if self._client is None:
            from openai import AsyncOpenAI

            key = resolve_api_key(self._settings, "openai", "OPENAI_API_KEY")
            self._client = AsyncOpenAI(api_key=key)
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
    ) -> AsyncGenerator[ChatChunk, None]:
        """Stream the selected API; cache breakpoints are server-managed no-ops."""
        method = (
            self._stream_responses if self._api == "responses" else self._stream_chat_completions
        )
        async with aclosing(
            method(messages, tools, max_tokens=max_tokens, temperature=temperature, system=system)
        ) as chunks:
            async for chunk in chunks:
                yield chunk

    async def _stream_responses(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> AsyncGenerator[ChatChunk, None]:
        """Reuse state only for an exact continuation of a completed turn.

        Full input is retained as a value snapshot, so context pruning, history
        edits, retries, and new sessions cannot inherit unrelated server state.
        Failed, interrupted, and incompletely consumed turns never advance it.
        A provider instance handles one Responses stream at a time.
        """
        if self._response_active:
            raise RuntimeError("Concurrent Responses streams require separate provider clients")
        self._response_active = True
        try:
            history = _translate_response_messages(messages)
            request: dict[str, Any] = {
                "model": self.model,
                "input": history,
                "max_output_tokens": max_tokens,
                "stream": True,
                "store": True,
                "include": ["reasoning.encrypted_content"],
            }
            # GPT-5 and o-series reasoning defaults reject sampling parameters.
            # We expose no reasoning-effort override, so leave their defaults
            # to the server; conventional models retain the requested sampling.
            if not re.match(r"^(gpt-5(?:[.-]|$)|o\d+(?:[.-]|$))", self.model):
                request["temperature"] = temperature
            if system is not None:
                request["instructions"] = system
            previous = self._response_history
            if self._response_id and previous and history[: len(previous)] == previous:
                request["previous_response_id"] = self._response_id
                request["input"] = history[len(previous) :]
            wrapped = _translate_tools(tools)
            if wrapped:
                # Responses defaults to strict schemas. Keep the neutral tool
                # contract's optional parameters and permissive schemas intact.
                request["tools"] = [
                    {"type": "function", **tool["function"], "strict": False} for tool in wrapped
                ]
            stream = await self._ensure_client().responses.create(**request)
            text: list[str] = []
            calls: dict[int, dict] = {}
            try:
                async for event in stream:
                    if event.type in ("response.output_text.delta", "response.refusal.delta"):
                        text.append(event.delta)
                        yield ChatChunk(delta_text=event.delta)
                    elif (
                        event.type == "response.output_item.added"
                        and event.item.type == "function_call"
                    ):
                        item = event.item
                        calls[event.output_index] = {
                            "id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments,
                        }
                        yield ChatChunk(
                            tool_call_delta={
                                "index": event.output_index,
                                "id": item.call_id,
                                "name": item.name,
                                "arguments_delta": item.arguments,
                            }
                        )
                    elif event.type == "response.function_call_arguments.delta":
                        calls[event.output_index]["arguments"] += event.delta
                        yield ChatChunk(
                            tool_call_delta={
                                "index": event.output_index,
                                "arguments_delta": event.delta,
                            }
                        )
                    elif event.type == "response.completed":
                        usage = event.response.usage
                        accounting = (
                            None
                            if usage is None
                            else {
                                "input_tokens": usage.input_tokens,
                                "output_tokens": usage.output_tokens,
                                "cache_read": getattr(
                                    usage.input_tokens_details, "cached_tokens", 0
                                ),
                                "cache_write": 0,
                            }
                        )
                        assistant = ChatMessage(
                            "assistant", "".join(text), list(calls.values()) or None
                        )
                        assistant.provider_data = _response_provider_data(event.response, assistant)
                        yield ChatChunk(
                            finish_reason="tool_calls" if calls else "stop",
                            usage=accounting,
                            provider_data=assistant.provider_data,
                        )
                        self._response_history = deepcopy(
                            history + _translate_response_messages([assistant])
                        )
                        self._response_id = event.response.id
                        return
                    elif event.type in ("response.failed", "response.incomplete", "error"):
                        # Do not let the loop execute partial tools on a failed
                        # or truncated response (it otherwise ignores reasons).
                        detail = getattr(event, "message", None)
                        response = getattr(event, "response", None)
                        if response is not None:
                            detail = getattr(response, "error", None) or getattr(
                                response, "incomplete_details", None
                            )
                        raise RuntimeError(f"OpenAI {event.type}: {detail}")
                raise RuntimeError("OpenAI Responses stream ended without response.completed")
            finally:
                await stream.close()
        finally:
            self._response_active = False

    async def _stream_chat_completions(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> AsyncGenerator[ChatChunk, None]:
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
        # OpenAI's finish reasons ("stop", "tool_calls", "length",
        # "content_filter") already match the contract vocabulary verbatim.
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            out.append(ChatChunk(finish_reason=finish_reason))

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
