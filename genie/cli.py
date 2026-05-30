"""CLI entrypoint. Each subcommand lands in its own phase."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Stub. Real subcommands land in PRs 0.7 and 1.11."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("genie — coming online. Available subcommands land per-phase.")
        print("  chat-once <prompt>  (Phase 0)")
        print("  code [path]         (Phase 1)")
        return 0
    print(f"genie: unknown command '{argv[0]}'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
