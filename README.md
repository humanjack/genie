# genie

A from-scratch, Python-native AI coding agent. Minimal, secure, pluggable — and powerful enough to dogfood from Phase 1.

Documentation:

- [docs/SPEC.md](docs/SPEC.md) — the full specification (all seven phases).
- [docs/DESIGN.md](docs/DESIGN.md) — architecture & design rationale for what's shipped.
- [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) — file-by-file implementation detail.
- [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) — the rolling PR plan.

## Status

Phase 0 (skeleton + provider abstraction) and Phase 1 (ReAct loop + four tools) complete —
tag `v0.1.0-phase1`. 374 tests, 99% coverage. Phase 2 (safety hooks) is next.

## Quickstart

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
