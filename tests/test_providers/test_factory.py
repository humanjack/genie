"""Tests for provider_factory — the startup wiring and lazy-import seam."""

from __future__ import annotations

import pytest

from genie.providers.factory import provider_factory
from genie.providers.fake import FakeProvider


def test_fake_spec_returns_fake_provider_with_model() -> None:
    provider = provider_factory("fake:fake-1")
    assert isinstance(provider, FakeProvider)
    assert provider.model == "fake-1"
    assert provider.name == "fake"


def test_fake_spec_custom_model() -> None:
    provider = provider_factory("fake:custom-model")
    assert provider.model == "custom-model"


def test_model_may_contain_colon() -> None:
    provider = provider_factory("fake:a:b:c")
    assert isinstance(provider, FakeProvider)
    assert provider.model == "a:b:c"


def test_malformed_spec_no_colon_raises() -> None:
    with pytest.raises(ValueError, match="provider:model"):
        provider_factory("anthropic")


def test_unknown_provider_lists_supported() -> None:
    with pytest.raises(ValueError) as exc:
        provider_factory("mystery:x")
    msg = str(exc.value)
    assert "mystery" in msg
    assert "anthropic" in msg
    assert "openai" in msg
    assert "fake" in msg


def test_anthropic_factory_builds_client() -> None:
    from genie.providers.anthropic_client import AnthropicClient

    provider = provider_factory("anthropic:claude-sonnet-4-6", client=object())
    assert isinstance(provider, AnthropicClient)
    assert provider.model == "claude-sonnet-4-6"


def test_openai_factory_builds_client() -> None:
    from genie.providers.openai_client import OpenAIClient

    provider = provider_factory("openai:gpt-5", client=object())
    assert isinstance(provider, OpenAIClient)
    assert provider.model == "gpt-5"


def test_settings_ignored_by_fake() -> None:
    provider = provider_factory("fake:fake-1", settings=object())
    assert isinstance(provider, FakeProvider)
