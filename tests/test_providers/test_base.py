"""Tests for the provider contract dataclasses and ABC."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from genie.providers.base import ChatChunk, ChatMessage, ProviderClient


def test_provider_client_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ProviderClient()  # type: ignore[abstract]


def test_minimal_concrete_subclass_works() -> None:
    class Minimal(ProviderClient):
        name = "minimal"
        model = "m-1"

        async def stream(
            self,
            messages,
            tools,
            *,
            max_tokens=4096,
            temperature=0.0,
            system=None,
            cache_breakpoints=None,
        ) -> AsyncIterator[ChatChunk]:
            yield ChatChunk(delta_text="hi")

        def count_tokens(self, messages) -> int:
            return len(messages)

    client = Minimal()
    assert client.name == "minimal"
    assert client.model == "m-1"
    assert client.count_tokens([ChatMessage(role="user", content="x")]) == 1


def test_chat_message_defaults() -> None:
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


def test_chat_message_accepts_content_blocks() -> None:
    blocks = [{"type": "text", "text": "hi"}]
    msg = ChatMessage(role="assistant", content=blocks, tool_calls=[{"id": "c1"}])
    assert msg.content == blocks
    assert msg.tool_calls == [{"id": "c1"}]


def test_chat_chunk_defaults() -> None:
    chunk = ChatChunk()
    assert chunk.delta_text is None
    assert chunk.tool_call_delta is None
    assert chunk.finish_reason is None
    assert chunk.usage is None


def test_chat_chunk_fields_set() -> None:
    usage = {"input_tokens": 1, "output_tokens": 2, "cache_read": 0, "cache_write": 0}
    chunk = ChatChunk(delta_text="x", finish_reason="stop", usage=usage)
    assert chunk.delta_text == "x"
    assert chunk.finish_reason == "stop"
    assert chunk.usage == usage


@pytest.mark.parametrize("field", ["name", "model"])
@pytest.mark.parametrize("value", [None, "", "   ", 123])
def test_provider_rejects_invalid_identity(field, value) -> None:
    from genie.providers.fake import FakeProvider

    class Invalid(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            setattr(self, field, value)

    with pytest.raises(TypeError, match=field):
        Invalid()


@pytest.mark.parametrize("field", ["name", "model"])
def test_provider_rejects_missing_identity(field) -> None:
    class Missing(ProviderClient):
        async def stream(self, *args, **kwargs) -> AsyncIterator[ChatChunk]:
            yield ChatChunk()

    setattr(Missing, "model" if field == "name" else "name", "provided")
    with pytest.raises(TypeError, match=field):
        Missing()


async def test_async_count_preserves_custom_sync_estimator() -> None:
    from genie.providers.fake import FakeProvider

    class Custom(FakeProvider):
        def count_tokens(self, messages) -> int:
            return 42

    assert await Custom().count_tokens_async([]) == 42


async def test_async_estimate_includes_prompt_and_tools() -> None:
    from genie.providers.fake import FakeProvider

    provider = FakeProvider()
    tools = [{"name": "read", "input_schema": {"type": "object"}}]
    system = "A long system prompt"
    assert await provider.count_tokens_async([], tools=tools, system=system) == (
        provider.count_tokens([]) + (len(str(tools)) + len(system)) // 4
    )
    assert await provider.count_tokens_async([], tools=[], system="") == 1


def test_provider_accepts_property_identity_and_inheritance() -> None:
    class Properties(ProviderClient):
        @property
        def name(self) -> str:
            return "property-provider"

        @property
        def model(self) -> str:
            return "property-model"

        async def stream(self, *args, **kwargs) -> AsyncIterator[ChatChunk]:
            yield ChatChunk()

    class Inherited(Properties):
        pass

    provider = Inherited()
    assert provider.name == "property-provider"
    assert provider.model == "property-model"


def test_provider_can_use_framework_abc_metaclass() -> None:
    from abc import ABCMeta

    from genie.providers.fake import FakeProvider

    class FrameworkMeta(ABCMeta):
        pass

    class FrameworkBase(metaclass=FrameworkMeta):
        pass

    class CustomProvider(FakeProvider, FrameworkBase):
        pass

    provider = CustomProvider(model="framework-model")
    assert isinstance(provider, FrameworkBase)
    assert provider.name == "fake"
    assert provider.model == "framework-model"
    with pytest.raises(TypeError, match="model"):
        CustomProvider(model="")


def test_identity_validation_waits_for_concrete_constructor() -> None:
    class Parent(ProviderClient):
        def __init__(self) -> None:
            self.name = "parent"

        async def stream(self, *args, **kwargs) -> AsyncIterator[ChatChunk]:
            yield ChatChunk()

    class Child(Parent):
        def __init__(self, model: str) -> None:
            super().__init__()
            self.model = model

    class Inherited(Child):
        pass

    assert Child("late-model").model == "late-model"
    assert Inherited("inherited-model").model == "inherited-model"
    with pytest.raises(TypeError, match="model"):
        Parent()
    with pytest.raises(TypeError, match="model"):
        Inherited("")


def test_dataclass_provider_keeps_generated_constructor_and_validates_identity() -> None:
    from dataclasses import dataclass

    @dataclass
    class DataProvider(ProviderClient):
        name = "data"
        model: str

        async def stream(self, *args, **kwargs) -> AsyncIterator[ChatChunk]:
            yield ChatChunk()

    assert DataProvider("data-model").model == "data-model"
    with pytest.raises(TypeError, match="model"):
        DataProvider("")


def test_dataclass_provider_can_finalize_identity_in_post_init() -> None:
    from dataclasses import dataclass

    from genie.providers.fake import FakeProvider

    @dataclass
    class DataProvider(FakeProvider):
        model: str

        def __post_init__(self) -> None:
            self.model = self.model.strip()
            super().__post_init__()
            self.model = self.model or "fallback-model"

    assert DataProvider("").model == "fallback-model"

    @dataclass
    class Inherited(DataProvider):
        pass

    assert Inherited("  data-model  ").model == "data-model"


def test_provider_preserves_cooperative_framework_constructor() -> None:
    from abc import ABCMeta

    class FrameworkMeta(ABCMeta):
        pass

    class FrameworkBase(metaclass=FrameworkMeta):
        def __init__(self, model: str) -> None:
            self.model = model

    class CustomProvider(ProviderClient, FrameworkBase):
        name = "framework"

        async def stream(self, *args, **kwargs) -> AsyncIterator[ChatChunk]:
            yield ChatChunk()

    assert CustomProvider("framework-model").model == "framework-model"
    with pytest.raises(TypeError, match="model"):
        CustomProvider("")
