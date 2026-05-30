# genie

A from-scratch, Python-native AI coding agent. Minimal, secure, pluggable — and powerful enough to dogfood from Phase 1.

See [docs/SPEC.md](docs/SPEC.md) for the full specification and [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) for the rolling PR plan.

## Status

Phase 0 (skeleton + provider abstraction) — in progress.

## Quickstart (once Phase 0 lands)

```bash
make install
export ANTHROPIC_API_KEY=...
.venv/bin/genie chat-once "Hello" --model anthropic:claude-sonnet-4-6
```

## Engineering rules

1. Every subsystem reached only through its abstract base.
2. Every disk / network / shell call gated by the hook chain.
3. Every component ships with ≥ 70% test coverage and at least two implementations (real + fake).
4. PRs are scoped to one component; reviewed with `/code-review` before merge.

See [SPEC.md §Operating principles](docs/SPEC.md#operating-principles--load-bearing--all-subsequent-design-defers-to-these).
