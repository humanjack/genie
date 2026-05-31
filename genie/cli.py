"""CLI entrypoint. Each subcommand lands in its own phase.

Phase 0 ships ``genie chat-once``: load config, build a provider via the
factory, and stream a single reply to stdout. The streaming core is factored
into :func:`run_chat_once` — a provider-in, text-out seam that tests drive with
a :class:`~genie.providers.fake.FakeProvider`, so the happy path needs no
network and no factory.

Phase 1 ships ``genie code [path]``: the interactive coding REPL (SPEC US-2).
The wiring (registry, sandbox, hooks, session, prompt) and the REPL core live in
:mod:`genie.agent`; this module only parses arguments, builds the provider, and
supplies the stdin reader. The reader is factored into :func:`_read_stdin_line`
so a test can feed a scripted line + EOF without a terminal.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from genie.agent import ToolCallDisplay, build_registry, load_system_prompt, run_code_session
from genie.config import load_config
from genie.hooks.manager import HookManager
from genie.providers.base import ChatMessage, ProviderClient
from genie.providers.factory import provider_factory
from genie.session.session import Session
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
    code.add_argument("path", nargs="?", help="Project path (default: current directory).")
    code.add_argument("--model", help="Provider spec 'provider:model' (default: config).")
    code.add_argument("--json-logs", action="store_true", help="Emit logs as JSON.")

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
    :func:`run_chat_once`); ``code`` opens the interactive coding REPL (see
    :func:`_cmd_code`).
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

    configure_logging(json_output=args.json_logs)
    if command == "chat-once":
        return _cmd_chat_once(args)
    return _cmd_code(args)


def _stdin_reader() -> str | None:
    """Read one line from stdin, returning None at EOF (the default REPL input)."""
    try:
        return input("genie> ")
    except EOFError:
        return None


def _cmd_code(args: argparse.Namespace) -> int:
    """Run the ``code`` subcommand: open an interactive coding session in a repo.

    Wires a provider, the four built-in tools (confined to ``path``), a
    tool-call display hook, and a fresh session, then drives the REPL. Provider
    and tool errors surface as one clean stderr line (no traceback); the input
    source is :data:`_READ_INPUT` so tests can inject a scripted reader.
    """
    log = get_logger("genie.cli")
    try:
        settings = load_config()
        spec = args.model or settings.provider.default
        provider = provider_factory(spec, settings=settings)
        root = Path(args.path or ".").resolve()
        registry = build_registry(root)
        hooks = HookManager()
        hooks.register(ToolCallDisplay())
        session = Session.create(
            root / ".genie" / "sessions",
            id=uuid4().hex,
            model=spec,
            working_dir=str(root),
        )
        asyncio.run(
            run_code_session(
                provider,
                registry,
                hooks,
                session,
                system=load_system_prompt(),
                read_input=_READ_INPUT,
                max_iterations=settings.loop.max_iterations,
            )
        )
    except KeyboardInterrupt:
        print("\ngenie: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        log.error("code_session_failed", error=str(exc), error_type=type(exc).__name__)
        print(f"genie: {exc}", file=sys.stderr)
        return 1
    return 0


# The REPL input source; overridable in tests (a scripted reader) so the code
# session can be driven without a real terminal.
_READ_INPUT = _stdin_reader


if __name__ == "__main__":
    raise SystemExit(main())
