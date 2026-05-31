"""Tests for the ``genie`` CLI and the ``run_chat_once`` streaming seam.

Every test runs offline: the happy paths drive a scripted
:class:`~genie.providers.fake.FakeProvider` directly into :func:`run_chat_once`,
and the full ``main`` path monkeypatches ``genie.cli.provider_factory`` so no
real provider (and no network) is ever reached. A single live test is gated
behind ``RUN_LIVE_API``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

from genie import cli
from genie.cli import main, run_chat_once
from genie.providers.base import ChatChunk, ChatMessage, ProviderClient
from genie.providers.fake import FakeProvider


async def test_run_chat_once_returns_and_prints(capsys):
    """Streams the scripted text, returns it whole, and prints it to stdout."""
    provider = FakeProvider.from_text("hello world")

    result = await run_chat_once(provider, "hi")

    assert result == "hello world"
    captured = capsys.readouterr()
    assert "hello world" in captured.out


async def test_run_chat_once_emits_usage_to_stderr(capsys):
    """A terminal chunk with usage surfaces a summary on stderr, not stdout."""
    provider = FakeProvider.from_text("hi", usage={"input_tokens": 3, "output_tokens": 2})

    result = await run_chat_once(provider, "hi")

    captured = capsys.readouterr()
    assert result == "hi"
    assert "hi" in captured.out
    assert "in=3" in captured.err
    assert "out=2" in captured.err
    assert "usage" not in captured.out


async def test_run_chat_once_no_usage_keeps_stderr_quiet(capsys):
    """Without usage on the terminal chunk, nothing is written to stderr."""
    provider = FakeProvider.from_text("plain")

    await run_chat_once(provider, "hi")

    captured = capsys.readouterr()
    assert captured.err == ""


async def test_run_chat_once_passes_system_and_max_tokens(capsys):
    """``system`` and ``max_tokens`` reach the provider's stream call."""
    provider = FakeProvider.from_text("ok")

    await run_chat_once(provider, "hi", system="be terse", max_tokens=128)

    assert provider.calls[0]["system"] == "be terse"
    assert provider.calls[0]["max_tokens"] == 128
    assert provider.calls[0]["tools"] == []
    messages = provider.calls[0]["messages"]
    assert messages == [ChatMessage(role="user", content="hi")]


def test_main_chat_once_streams_via_monkeypatched_factory(capsys, monkeypatch):
    """The full ``main`` path builds a provider via the factory and streams it."""
    monkeypatch.setattr(
        cli, "provider_factory", lambda *a, **k: FakeProvider.from_text("scripted reply")
    )

    rc = main(["chat-once", "hi", "--model", "fake:fake-1"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "scripted reply" in captured.out


def test_main_chat_once_forwards_system_and_max_tokens(capsys, monkeypatch):
    """``--system`` and ``--max-tokens`` flow through main into the provider."""
    provider = FakeProvider.from_text("done")
    monkeypatch.setattr(cli, "provider_factory", lambda *a, **k: provider)

    rc = main(["chat-once", "hello", "--system", "sys text", "--max-tokens", "256"])

    assert rc == 0
    assert provider.calls[0]["system"] == "sys text"
    assert provider.calls[0]["max_tokens"] == 256


def test_main_chat_once_malformed_model_spec(capsys):
    """A spec with no colon is reported as a clean one-line error, rc != 0."""
    rc = main(["chat-once", "hi", "--model", "bogus"])

    captured = capsys.readouterr()
    assert rc != 0
    assert "genie:" in captured.err
    assert "bogus" in captured.err
    assert "Traceback" not in captured.err


def test_main_chat_once_missing_api_key(capsys, monkeypatch):
    """A provider whose stream raises ValueError yields a clean error, rc != 0."""

    class _MissingKeyProvider(ProviderClient):
        name = "fake"
        model = "fake-1"

        async def stream(self, messages, tools, **kwargs) -> AsyncIterator[ChatChunk]:
            raise ValueError("Missing API key for provider 'anthropic': set ANTHROPIC_API_KEY")
            yield ChatChunk()  # pragma: no cover

        def count_tokens(self, messages) -> int:  # pragma: no cover
            return 0

    monkeypatch.setattr(cli, "provider_factory", lambda *a, **k: _MissingKeyProvider())

    rc = main(["chat-once", "hi", "--model", "anthropic:claude-sonnet-4-6"])

    captured = capsys.readouterr()
    assert rc != 0
    assert "ANTHROPIC_API_KEY" in captured.err
    assert "Traceback" not in captured.err


def test_main_help_returns_zero(capsys):
    """No args / --help preserves the Phase 0 smoke behavior."""
    assert main([]) == 0
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "genie" in captured.out
    assert "chat-once" in captured.out


def test_main_unknown_command_returns_two(capsys):
    """An unknown command returns 2 with 'unknown' on stderr (smoke contract)."""
    rc = main(["nope"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "unknown" in captured.err


def test_main_code_is_placeholder(capsys):
    """The ``code`` subcommand stays a Phase 1 placeholder, returning 0."""
    rc = main(["code"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Phase 1" in captured.err


@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_API"),
    reason="live API test; set RUN_LIVE_API to enable",
)
def test_live_chat_once():  # pragma: no cover - network-gated
    """End-to-end smoke against a real provider when RUN_LIVE_API is set."""
    rc = main(["chat-once", "Say the word OK and nothing else."])
    assert rc == 0
