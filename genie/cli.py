"""CLI entrypoint. Each subcommand lands in its own phase.

Phase 0 ships ``genie chat-once``: load config, build a provider via the
factory, and stream a single reply to stdout. The streaming core is factored
into :func:`run_chat_once` — a provider-in, text-out seam that tests drive with
a :class:`~genie.providers.fake.FakeProvider`, so the happy path needs no
network and no factory.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from genie.config import load_config
from genie.providers.base import ChatMessage, ProviderClient
from genie.providers.factory import provider_factory
from genie.utils.logger import configure_logging, get_logger

_HELP = (
    "genie — coming online. Available subcommands land per-phase.\n"
    "  chat-once <prompt>  (Phase 0)\n"
    "  code [path]         (Phase 1)"
)


async def run_chat_once(
    provider: ProviderClient,
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """Stream one reply from ``provider``, printing it live, and return the text.

    Builds a single user-message conversation, calls :meth:`ProviderClient.stream`
    with no tools, and writes each ``delta_text`` to stdout as it arrives —
    flushing after every chunk so streaming is visible — then emits a trailing
    newline. If the terminal chunk carries ``usage``, a concise dim summary is
    written to stderr, keeping stdout clean prompt output. Returns the full
    concatenated reply.
    """
    messages = [ChatMessage(role="user", content=prompt)]
    parts: list[str] = []
    usage: dict | None = None
    async for chunk in provider.stream(messages, [], system=system, max_tokens=max_tokens):
        if chunk.delta_text:
            parts.append(chunk.delta_text)
            sys.stdout.write(chunk.delta_text)
            sys.stdout.flush()
        if chunk.usage is not None:
            usage = chunk.usage
    sys.stdout.write("\n")
    sys.stdout.flush()
    if usage is not None:
        _print_usage(usage)
    return "".join(parts)


def _print_usage(usage: dict) -> None:
    """Write a minimal, dim token-usage summary to stderr."""
    in_tokens = usage.get("input_tokens", 0)
    out_tokens = usage.get("output_tokens", 0)
    line = f"[usage] in={in_tokens} out={out_tokens}"
    try:
        from rich.console import Console

        Console(stderr=True).print(line, style="dim")
    except Exception:
        print(line, file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the ``chat-once`` and ``code`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="genie",
        description="A from-scratch, Python-native AI coding agent.",
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser("chat-once", help="Stream a single reply for one prompt.")
    chat.add_argument("prompt", help="The prompt to send to the model.")
    chat.add_argument("--model", help="Provider spec 'provider:model' (default: config).")
    chat.add_argument("--system", help="Optional system prompt.")
    chat.add_argument(
        "--max-tokens", type=int, default=4096, help="Max tokens to generate (default: 4096)."
    )
    chat.add_argument("--json-logs", action="store_true", help="Emit logs as JSON.")

    code = subparsers.add_parser("code", help="Interactive coding agent (Phase 1).")
    code.add_argument("path", nargs="?", help="Project path.")

    return parser


def _cmd_chat_once(args: argparse.Namespace) -> int:
    """Run the ``chat-once`` subcommand: build a provider and stream a reply."""
    log = get_logger("genie.cli")
    try:
        settings = load_config()
        spec = args.model or settings.provider.default
        provider = provider_factory(spec, settings=settings)
        asyncio.run(
            run_chat_once(
                provider,
                args.prompt,
                system=args.system,
                max_tokens=args.max_tokens,
            )
        )
    except KeyboardInterrupt:
        # User aborted mid-stream; exit quietly without a traceback (130 = SIGINT).
        print("\ngenie: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        # Boundary catch-all: config/spec errors (ValueError/RuntimeError) AND
        # provider-SDK runtime errors (auth, network, rate-limit — subclasses of
        # the SDKs' APIError, i.e. plain Exception) must surface as one clean
        # line, never a traceback. The error is logged with its type for triage.
        log.error("chat_once_failed", error=str(exc), error_type=type(exc).__name__)
        print(f"genie: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch a subcommand; return a process exit code.

    With no arguments or ``-h``/``--help``, prints per-phase help and returns 0.
    An unknown command returns 2. ``chat-once`` streams one reply (see
    :func:`run_chat_once`); ``code`` is a Phase 1 placeholder.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(_HELP)
        return 0

    command = argv[0]
    if command not in {"chat-once", "code"}:
        print(f"genie: unknown command '{command}'", file=sys.stderr)
        return 2

    parser = _build_parser()
    args = parser.parse_args(argv)

    if command == "chat-once":
        configure_logging(json_output=args.json_logs)
        return _cmd_chat_once(args)

    print("genie code: coming in Phase 1.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
