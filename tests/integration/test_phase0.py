"""Phase 0 integration tests — the composed surface, not unit-mocked internals.

These exercise the real ``config → factory → provider → CLI stream`` wiring
together (only the network is faked) and prove the central Phase 0 promise:
the CLI/loop depend on the :class:`ProviderClient` abstraction alone, so any
implementation drops in with zero edits to the caller.

Live smoke tests against the real Anthropic/OpenAI APIs are gated behind
``RUN_LIVE_API`` and skipped by default.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from genie import cli
from genie.cli import main, run_chat_once
from genie.config import load_config
from genie.providers.anthropic_client import AnthropicClient
from genie.providers.base import ChatChunk, ChatMessage, ProviderClient
from genie.providers.factory import provider_factory
from genie.providers.fake import FakeProvider
from genie.providers.openai_client import OpenAIClient


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


# --- config file → factory wiring (real components, no mocks) ----------------


def test_config_file_drives_provider_selection(tmp_path: Path) -> None:
    """A provider.default in a real TOML flows through load_config into the factory."""
    cfg = _write_config(tmp_path, '[provider]\ndefault = "fake:from-config"\n')
    settings = load_config(cfg, env={})

    provider = provider_factory(settings.provider.default, settings=settings)

    assert isinstance(provider, FakeProvider)
    assert provider.model == "from-config"


def test_env_override_drives_provider_selection(tmp_path: Path) -> None:
    """GENIE_PROVIDER_DEFAULT beats the TOML value end-to-end into the factory."""
    cfg = _write_config(tmp_path, '[provider]\ndefault = "fake:from-config"\n')
    settings = load_config(cfg, env={"GENIE_PROVIDER_DEFAULT": "fake:from-env"})

    provider = provider_factory(settings.provider.default, settings=settings)

    assert provider.model == "from-env"


def test_factory_builds_real_adapters_without_network() -> None:
    """The real factory wires the real SDK adapters (client injected; no network)."""
    anthropic = provider_factory("anthropic:claude-sonnet-4-6", client=object())
    openai = provider_factory("openai:gpt-4o-mini", client=object())

    assert isinstance(anthropic, AnthropicClient)
    assert isinstance(openai, OpenAIClient)
    assert anthropic.model == "claude-sonnet-4-6"
    assert openai.model == "gpt-4o-mini"


# --- full chat-once path: config + logger + CLI + provider -------------------


def test_chat_once_streams_full_reply_through_cli(capsys, monkeypatch) -> None:
    """`genie chat-once` streams a scripted FakeProvider reply to stdout, rc 0."""
    monkeypatch.setattr(
        cli,
        "provider_factory",
        lambda *a, **k: FakeProvider.from_text("the answer is 42", chunks=4),
    )

    rc = main(["chat-once", "what is the answer?", "--model", "fake:fake-1"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "the answer is 42" in captured.out


# --- replaceability: a second ProviderClient impl drops in unchanged ---------


class EchoProvider(ProviderClient):
    """A minimal alternative ProviderClient: streams the prompt back, char by char.

    Deliberately unrelated to FakeProvider — its existence proves the CLI core
    depends only on the :class:`ProviderClient` ABC, not on any concrete type.
    """

    name = "echo"

    def __init__(self, *, model: str = "echo-1") -> None:
        self.model = model

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
        last = str(messages[-1].content) if messages else ""
        for ch in last:
            yield ChatChunk(delta_text=ch)
        yield ChatChunk(finish_reason="stop")

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        return max(1, sum(len(str(m.content)) for m in messages) // 4)


async def test_replaceability_run_chat_once_against_two_impls(capsys) -> None:
    """run_chat_once behaves identically for FakeProvider and a second impl — no edits."""
    fake_result = await run_chat_once(FakeProvider.from_text("ping"), "ignored")
    capsys.readouterr()  # drain

    echo_result = await run_chat_once(EchoProvider(), "ping")
    capsys.readouterr()

    # Different providers, same loop core, same contract → same observable text.
    assert fake_result == "ping"
    assert echo_result == "ping"


def test_replaceability_echo_provider_through_cli(capsys, monkeypatch) -> None:
    """The alternative impl also drives the full CLI path with zero CLI changes."""
    monkeypatch.setattr(cli, "provider_factory", lambda *a, **k: EchoProvider())

    rc = main(["chat-once", "hello", "--model", "echo:echo-1"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "hello" in captured.out


# --- live smoke (gated) ------------------------------------------------------


@pytest.mark.skipif(not os.getenv("RUN_LIVE_API"), reason="set RUN_LIVE_API for live smoke")
async def test_live_anthropic_end_to_end() -> None:  # pragma: no cover - network-gated
    """Real Anthropic call through the full contract when RUN_LIVE_API is set."""
    provider = provider_factory("anthropic:claude-haiku-4-5-20251001")
    out = [
        c
        async for c in provider.stream(
            [ChatMessage(role="user", content="Reply with the single word: pong.")],
            [],
            max_tokens=16,
        )
    ]
    text = "".join(c.delta_text for c in out if c.delta_text is not None)
    assert text.strip()
    assert any(c.finish_reason for c in out)


@pytest.mark.skipif(not os.getenv("RUN_LIVE_API"), reason="set RUN_LIVE_API for live smoke")
async def test_live_openai_end_to_end() -> None:  # pragma: no cover - network-gated
    """Real OpenAI call through the full contract when RUN_LIVE_API is set."""
    provider = provider_factory("openai:gpt-4o-mini")
    out = [
        c
        async for c in provider.stream(
            [ChatMessage(role="user", content="Reply with the single word: pong.")],
            [],
            max_tokens=16,
        )
    ]
    text = "".join(c.delta_text for c in out if c.delta_text is not None)
    assert text.strip()
    assert any(c.finish_reason for c in out)
