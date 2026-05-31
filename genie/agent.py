"""The coding agent: wiring + the interactive REPL core (SPEC US-2).

Where :mod:`genie.loop` owns the provider-agnostic ReAct cycle, this module is
the *assembly* that turns the merged seams into a usable coding session:

- :func:`build_registry` binds a workspace root to a sandbox and the four
  built-in tools (``read_file``, ``write_file``, ``edit_file``, ``bash``).
- :func:`load_system_prompt` loads the coding-agent persona from
  ``prompts/coding_system.md`` (with a built-in fallback so an installed wheel
  that does not ship the prompt still runs).
- :class:`ToolCallDisplay` is the observability hook SPEC US-2 calls for —
  "shows every tool call" — printing a line before each call and a terse
  success/failure preview after it. It never blocks.
- :func:`run_code_session` is the REPL: read a line, run a turn, stream the
  reply, repeat. Input is injected as a ``read_input`` callable so the real CLI
  can wire stdin while tests drive a scripted iterator with no terminal.

Nothing here reaches for the clock, a uuid, the network, or a concrete provider:
identity and time are the caller's to supply (see :mod:`genie.cli`), and the
provider arrives ready-built. That keeps the whole module drivable offline by a
``FakeProvider``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from genie.hooks.manager import HookManager, HookOutcome
from genie.loop import ToolCall, run_turn
from genie.providers.base import ChatMessage, ProviderClient
from genie.sandbox.local_subprocess import LocalSubprocessBackend
from genie.session.session import Session
from genie.tools.builtins.bash import make_bash
from genie.tools.builtins.edit_file import make_edit_file
from genie.tools.builtins.read_file import make_read_file
from genie.tools.builtins.write_file import make_write_file
from genie.tools.registry import ToolRegistry
from genie.tools.result import ToolResult

_PROMPT_RELPATH = "prompts/coding_system.md"
_ARGS_PREVIEW_CHARS = 80
_RESULT_PREVIEW_CHARS = 120

_FALLBACK_SYSTEM_PROMPT = (
    "You are genie, a coding agent operating in a repository via tools "
    "(read_file, write_file, edit_file, bash). Read files before you edit them, "
    "make the smallest change that solves the task, run tests to verify your "
    "work, and be concise."
)

# Exit commands recognised at the REPL prompt.
_EXIT_COMMANDS = frozenset({"/exit", "/quit"})


def build_registry(root: str | Path, *, bash_timeout: float = 30.0) -> ToolRegistry:
    """Build a registry of the four built-in tools bound to ``root``.

    Constructs a :class:`~genie.sandbox.local_subprocess.LocalSubprocessBackend`
    confined to ``root`` and registers ``read_file``, ``write_file``,
    ``edit_file`` (all bound to the same workspace root) and ``bash`` (bound to
    the sandbox). Registration order is the order above.

    Args:
        root: The workspace root every tool is confined to.
        bash_timeout: Per-command timeout (seconds) for the ``bash`` tool.

    Returns:
        A populated :class:`~genie.tools.registry.ToolRegistry`.
    """
    sandbox = LocalSubprocessBackend(root)
    registry = ToolRegistry()
    registry.register(make_read_file(root))
    registry.register(make_write_file(root))
    registry.register(make_edit_file(root))
    registry.register(make_bash(sandbox, timeout=bash_timeout))
    return registry


def load_system_prompt() -> str:
    """Return the coding-agent system prompt.

    Reads ``prompts/coding_system.md`` relative to the repository root (this
    file's parent's parent). The prompt lives outside the importable package, so
    a wheel that does not ship it falls back to a short built-in string rather
    than failing — the session still has a usable persona either way.

    Returns:
        The prompt text, stripped of trailing whitespace, or the built-in
        fallback when the file is absent or unreadable.
    """
    prompt_path = Path(__file__).resolve().parent.parent / _PROMPT_RELPATH
    try:
        text = prompt_path.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_SYSTEM_PROMPT
    return text.strip() or _FALLBACK_SYSTEM_PROMPT


def _truncate(text: str, limit: int) -> str:
    """Collapse ``text`` to a single line and clip it to ``limit`` chars."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


def _format_args(args: dict) -> str:
    """Render a tool call's args as a compact ``k=v, …`` preview."""
    return _truncate(", ".join(f"{k}={v!r}" for k, v in args.items()), _ARGS_PREVIEW_CHARS)


class ToolCallDisplay:
    """A :class:`~genie.hooks.manager.Hook` that prints every tool call (SPEC US-2).

    Subscribes to ``before_tool_call`` and ``after_tool_call``. Before a call it
    prints ``→ name(args)``; after it prints ``✓``/``✗`` with a short preview of
    the result. It is purely observational: it always returns ``None`` so it can
    never block a call (gating is the approval hook's job, SPEC §7.3).

    Output goes through an injected ``write`` callable (default :func:`print`),
    so tests can capture it and the CLI can route it wherever it likes.

    Attributes:
        name: Stable hook identifier for logs/diagnostics.
        events: The two tool-lifecycle events this hook observes.
    """

    name = "tool_call_display"

    def __init__(self, write: Callable[[str], None] = print) -> None:
        """Build the display hook.

        Args:
            write: Sink for each rendered line; defaults to :func:`print`.
        """
        # Instance attribute (not a class default): satisfies the Hook protocol's
        # instance-level ``events`` and avoids a mutable class-attribute default.
        self.events = ["before_tool_call", "after_tool_call"]
        self._write = write

    async def __call__(self, event: str, **payload: object) -> HookOutcome | None:
        """Render the call (``before``) or its result (``after``); never block."""
        call = payload.get("call")
        if event == "before_tool_call" and isinstance(call, ToolCall):
            self._write(f"→ {call.name}({_format_args(call.args)})")
        elif event == "after_tool_call" and isinstance(call, ToolCall):
            result = payload.get("result")
            self._write(self._format_result(call, result))
        return None

    @staticmethod
    def _format_result(call: ToolCall, result: object) -> str:
        """Render the after-call line: a status glyph plus a result preview."""
        if not isinstance(result, ToolResult):
            return f"✓ {call.name}"
        mark = "✗" if result.is_error else "✓"
        preview = _truncate(result.content, _RESULT_PREVIEW_CHARS)
        return f"{mark} {call.name}: {preview}" if preview else f"{mark} {call.name}"


async def run_code_session(
    provider: ProviderClient,
    registry: ToolRegistry,
    hooks: HookManager,
    session: Session,
    *,
    system: str | None,
    read_input: Callable[[], str | None],
    write: Callable[[str], None] = print,
    max_iterations: int = 50,
) -> None:
    """Run the interactive coding REPL until the user ends the session.

    Each iteration reads one line via ``read_input``. ``None`` (EOF) or a line
    in ``{"/exit", "/quit"}`` ends the loop. Otherwise the line is appended as a
    ``user`` message and one :func:`~genie.loop.run_turn` is driven, streaming
    the assistant's text to stdout as it arrives (so the reply is visible live);
    a trailing newline is written after the turn. Tool calls within the turn are
    rendered by any :class:`ToolCallDisplay` registered on ``hooks``.

    A ``KeyboardInterrupt`` (Ctrl-C) during a turn is caught and reported as a
    one-line notice, and the loop continues to the next prompt — interrupting a
    runaway turn returns control to the user rather than killing the session.

    Args:
        provider: The ready-built model client to stream turns from.
        registry: The tools the model may call this session.
        hooks: The hook chain every lifecycle event is announced through.
        session: The conversation state; user/assistant/tool messages are
            appended here as the session proceeds.
        system: System prompt passed to the model each turn (may be ``None``).
        read_input: Returns the next user line, or ``None`` at EOF/quit.
        write: Sink for line-oriented notices (prompt-adjacent output, the
            trailing newline, interrupt notices); defaults to :func:`print`.
        max_iterations: Per-turn safety budget forwarded to
            :func:`~genie.loop.run_turn`.
    """

    def _on_text_delta(fragment: str) -> None:
        sys.stdout.write(fragment)
        sys.stdout.flush()

    while True:
        line = read_input()
        if line is None or line.strip() in _EXIT_COMMANDS:
            break
        if not line.strip():
            continue

        session.append(ChatMessage(role="user", content=line))
        try:
            await run_turn(
                session,
                provider,
                registry,
                hooks,
                system=system,
                max_iterations=max_iterations,
                on_text_delta=_on_text_delta,
            )
        except KeyboardInterrupt:
            # Ctrl-C during a turn aborts that turn, not the session: surface a
            # notice and return to the prompt so the user keeps control.
            write("\n[interrupted — press Ctrl-D or type /exit to quit]")
            continue
        # Terminate the streamed reply with a newline so the next prompt is clean.
        write("")
