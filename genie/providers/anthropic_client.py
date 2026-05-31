"""Anthropic-backed :class:`ProviderClient` implementation.

This adapter translates the provider-neutral contract in
:mod:`genie.providers.base` into Anthropic's Messages API and maps the SDK's
raw streaming events back onto :class:`ChatChunk`. The agent loop therefore
never sees an Anthropic-native type; it consumes the same index-addressed
tool-call delta shape that :class:`~genie.providers.fake.FakeProvider` emits, so
the real and fake providers are interchangeable.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from genie.providers.base import ChatChunk, ChatMessage, ProviderClient

# Anthropic stop_reason -> contract finish_reason. Unmapped reasons pass through.
_STOP_REASON_MAP = {"end_turn": "stop", "tool_use": "tool_calls"}


class AnthropicClient(ProviderClient):
    """Stream completions from Anthropic's Messages API.

    The SDK client is created lazily on the first :meth:`stream` call (or
    injected via ``client=`` for tests), so constructing this object never
    requires an API key — only actually streaming does. Credentials resolve via
    ``settings.require_api_key("anthropic", os.environ)`` when a settings object
    is supplied, otherwise from the ``ANTHROPIC_API_KEY`` environment variable.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str,
        settings: object | None = None,
        *,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the client without contacting Anthropic.

        Args:
            model: The Anthropic model id (e.g. ``"claude-sonnet-4-6"``).
            settings: Optional settings object exposing ``require_api_key``; when
                given it is consulted for the API key at first use.
            client: An optional pre-built ``AsyncAnthropic`` (or compatible)
                client. When supplied it is used as-is and no key is required —
                this is the injection seam used by unit tests.
            **kwargs: Ignored; accepted so the factory can forward extra options.
        """
        self.model = model
        self._settings = settings
        self._client = client

    def _resolve_api_key(self) -> str:
        """Return the API key, raising a clear error if it is missing."""
        if self._settings is not None and hasattr(self._settings, "require_api_key"):
            return self._settings.require_api_key("anthropic", os.environ)  # type: ignore[attr-defined]
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Missing API key for provider 'anthropic': "
                "set the ANTHROPIC_API_KEY environment variable."
            )
        return key

    def _get_client(self) -> Any:
        """Return the SDK client, building it lazily on first use."""
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self._resolve_api_key())
        return self._client

    @staticmethod
    def _translate_message(message: ChatMessage) -> dict[str, Any]:
        """Translate one :class:`ChatMessage` into an Anthropic message dict.

        Assistant ``tool_calls`` become ``tool_use`` content blocks and
        ``role == "tool"`` results become a ``tool_result`` block keyed by
        ``tool_call_id``. Plain text/content is passed through untouched.
        """
        if message.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }
                ],
            }
        if message.role == "assistant" and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if isinstance(message.content, str):
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
            else:
                blocks.extend(message.content)
            for call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id"),
                        "name": call.get("name"),
                        "input": call.get("arguments", {}),
                    }
                )
            return {"role": "assistant", "content": blocks}
        return {"role": message.role, "content": message.content}

    @staticmethod
    def _apply_cache_breakpoint(message: dict[str, Any]) -> None:
        """Tag ``message``'s last content block with an ephemeral cache control.

        Anthropic attaches ``cache_control`` to a content block, so string
        content is first promoted to a single text block.
        """
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
            message["content"] = content
        if content:
            content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}

    def _build_params(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        system: str | None,
        cache_breakpoints: list[int] | None,
    ) -> dict[str, Any]:
        """Assemble the keyword arguments for ``client.messages.stream``.

        Factored out so message/param translation can be unit-tested without a
        live stream.
        """
        translated = [self._translate_message(m) for m in messages]
        for idx in cache_breakpoints or []:
            if 0 <= idx < len(translated):
                self._apply_cache_breakpoint(translated[idx])
        params: dict[str, Any] = {
            "model": self.model,
            "messages": translated,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            params["tools"] = tools
        if system is not None:
            params["system"] = system
        return params

    @staticmethod
    def _usage_dict(message_usage: Any, output_tokens: int | None) -> dict[str, int]:
        """Build the contract usage dict from Anthropic usage objects.

        ``message_usage`` is the ``Usage`` from ``message_start`` (carries input
        and cache tokens); ``output_tokens`` comes from the terminal
        ``message_delta``.
        """
        usage: dict[str, int] = {}
        if message_usage is not None:
            usage["input_tokens"] = getattr(message_usage, "input_tokens", 0) or 0
            cache_read = getattr(message_usage, "cache_read_input_tokens", None)
            if cache_read is not None:
                usage["cache_read"] = cache_read
            cache_write = getattr(message_usage, "cache_creation_input_tokens", None)
            if cache_write is not None:
                usage["cache_write"] = cache_write
        usage["output_tokens"] = output_tokens or 0
        return usage

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
        """Stream an Anthropic completion as provider-neutral chunks.

        Maps the SDK's raw events onto :class:`ChatChunk`: ``text_delta`` ->
        ``delta_text``; a ``tool_use`` ``content_block_start`` opens a tool-call
        slot keyed by the block ``index``; ``input_json_delta`` appends
        ``partial_json`` to that slot's ``arguments_delta``; the terminal
        ``message_delta`` yields ``finish_reason`` and ``usage``.
        """
        client = self._get_client()
        params = self._build_params(
            messages,
            tools,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            cache_breakpoints=cache_breakpoints,
        )

        start_usage: Any = None
        async with client.messages.stream(**params) as stream:
            async for event in stream:
                # Attributes are read via getattr because the SDK's stream yields
                # a union of raw-event types; the field set varies per ``type``.
                event_type = getattr(event, "type", None)
                if event_type == "message_start":
                    start_usage = getattr(getattr(event, "message", None), "usage", None)
                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", None) == "tool_use":
                        yield ChatChunk(
                            tool_call_delta={
                                "index": getattr(event, "index", None),
                                "id": getattr(block, "id", None),
                                "name": getattr(block, "name", None),
                                "arguments_delta": "",
                            }
                        )
                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        yield ChatChunk(delta_text=getattr(delta, "text", ""))
                    elif delta_type == "input_json_delta":
                        yield ChatChunk(
                            tool_call_delta={
                                "index": getattr(event, "index", None),
                                "arguments_delta": getattr(delta, "partial_json", ""),
                            }
                        )
                elif event_type == "message_delta":
                    raw_reason = getattr(getattr(event, "delta", None), "stop_reason", None)
                    finish_reason = (
                        _STOP_REASON_MAP.get(raw_reason, raw_reason)
                        if raw_reason is not None
                        else None
                    )
                    output_tokens = getattr(getattr(event, "usage", None), "output_tokens", None)
                    yield ChatChunk(
                        finish_reason=finish_reason,
                        usage=self._usage_dict(start_usage, output_tokens),
                    )

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        """Return a deterministic local estimate of the prompt's token count.

        Uses a cheap ``chars // 4`` heuristic (minimum 1) to avoid a network
        round-trip. This is an *estimate*, not Anthropic's exact tokenizer count;
        a precise async ``messages.count_tokens`` call is deferred (issue #47).
        """
        chars = sum(len(str(m.content)) for m in messages)
        return max(1, chars // 4)
