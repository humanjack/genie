# genie — Development Plan (Phase 0 + Phase 1)

> Concrete sequencing of PRs, ownership of work units, parallelism strategy, and exit criteria. Mirrors and operationalizes Part III of [SPEC.md](./SPEC.md).

## Scope of this plan

Phase 0 (skeleton + provider abstraction) and Phase 1 (minimal ReAct loop with the four Pi tools). Phases 2–7 are out of scope until the Phase 1 tag is cut.

## Component → PR map

Each row is one PR. Status legend: ⬜ not started, 🟡 in progress, ✅ merged.

### Phase 0 — Skeleton & provider abstraction

| # | PR slug | Branch | Touches | Depends on | Parallel? |
|---|---------|--------|---------|------------|-----------|
| 0.1 | `chore/scaffold` | `phase0/scaffold` | `pyproject.toml`, `.gitignore`, `Makefile`, CI config, package skeleton | — | no (must land first) |
| 0.2 | `feat/config` | `phase0/config` | `genie/config.py` + tests | 0.1 | yes |
| 0.3 | `feat/logger` | `phase0/logger` | `genie/utils/logger.py` + tests | 0.1 | yes |
| 0.4 | `feat/provider-base` | `phase0/provider-base` | `genie/providers/base.py`, `factory.py`, `fake.py` + tests | 0.1 | yes |
| 0.5 | `feat/provider-anthropic` | `phase0/provider-anthropic` | `genie/providers/anthropic_client.py` + tests | 0.4 | yes (with 0.6) |
| 0.6 | `feat/provider-openai` | `phase0/provider-openai` | `genie/providers/openai_client.py` + tests | 0.4 | yes (with 0.5) |
| 0.7 | `feat/cli-chat-once` | `phase0/cli-chat-once` | `genie/cli.py` w/ `chat-once` subcommand + tests | 0.2, 0.4 | no |
| 0.8 | `test/phase0-integration` | `phase0/integration` | `tests/integration/test_phase0.py` | all above | no (final) |

### Phase 1 — ReAct loop + 4 tools

| # | PR slug | Branch | Touches | Depends on | Parallel? |
|---|---------|--------|---------|------------|-----------|
| 1.1 | `feat/tool-base` | `phase1/tool-base` | `genie/tools/base.py` (Tool, @tool, ToolResult) + tests | 0.* | no (gates the rest) |
| 1.2 | `feat/tool-registry` | `phase1/tool-registry` | `genie/tools/registry.py` + tests | 1.1 | no |
| 1.3 | `feat/hooks-skeleton` | `phase1/hooks-skeleton` | `genie/hooks/manager.py` (events + registration; no built-ins yet) + tests | 1.1 | yes (parallel with 1.2) |
| 1.4 | `feat/sandbox-local` | `phase1/sandbox-local` | `genie/sandbox/base.py`, `local_subprocess.py`, `recording.py` (fake) + tests | 0.* | yes |
| 1.5 | `feat/session-transcript` | `phase1/session-transcript` | `genie/session/transcript.py`, `session.py` + tests | 0.* | yes |
| 1.6 | `feat/tool-read-file` | `phase1/tool-read-file` | `genie/tools/builtins/read_file.py` + tests | 1.2 | yes |
| 1.7 | `feat/tool-write-file` | `phase1/tool-write-file` | `genie/tools/builtins/write_file.py` + tests | 1.2, 1.4 | yes |
| 1.8 | `feat/tool-edit-file` | `phase1/tool-edit-file` | `genie/tools/builtins/edit_file.py` + tests | 1.2 | yes |
| 1.9 | `feat/tool-bash` | `phase1/tool-bash` | `genie/tools/builtins/bash.py` + tests | 1.2, 1.4 | yes |
| 1.10 | `feat/loop` | `phase1/loop` | `genie/loop.py` + tests | 1.1, 1.2, 1.3, 1.5 | no |
| 1.11 | `feat/cli-code` | `phase1/cli-code` | `genie/cli.py` `code` subcommand (REPL) + tests | 1.10, all tools | no |
| 1.12 | `test/phase1-e2e` | `phase1/e2e` | `tests/e2e/test_user_stories.py` | all above | no (final) |

## Parallelism strategy

We use the `Workflow` orchestration to spawn parallel implementation agents for independent PRs in the same wave. Wave boundaries:

- **Wave A (Phase 0 foundation):** PR 0.1 alone, then 0.2, 0.3, 0.4 in parallel.
- **Wave B (Phase 0 providers):** 0.5, 0.6 in parallel.
- **Wave C (Phase 0 finish):** 0.7 then 0.8.
- **Wave D (Phase 1 base):** 1.1, then 1.2 + 1.3 in parallel, then 1.4 + 1.5 in parallel.
- **Wave E (Phase 1 tools):** 1.6, 1.7, 1.8, 1.9 in parallel.
- **Wave F (Phase 1 loop & e2e):** 1.10, 1.11, 1.12 sequentially.

Each parallel agent works on its own git branch in a temporary worktree (`isolation: "worktree"`).

## Per-PR checklist

```
[ ] Branch created from latest main
[ ] Component implemented behind its abstract base
[ ] At least one alternative impl OR a fake/stub for tests
[ ] pytest passes locally
[ ] Coverage for the touched package ≥ 70%
[ ] Pre-commit hooks (ruff, pyright) clean
[ ] PR opened with description (what / why / tested / deferred)
[ ] /code-review (skill) run; findings resolved or filed as issues
[ ] CI green
[ ] Squash-merge to main
```

## Integration / E2E test plan

After Phase 1, the e2e suite scripts these scenarios as a real user:

| Scenario | Setup | Success |
|----------|-------|---------|
| S1 | seeded buggy `parse_args` repo; FakeProvider with recorded stream that calls read_file → edit_file → bash | tests pass after edit; transcript contains all 3 tool calls; no errors |
| S2 | empty repo; user prompt "create hello.py that prints 'hi'" | file created via write_file; bash run prints 'hi' |
| S3 | repo with mock dangerous prompt; FakeProvider tries `rm -rf /tmp/genie-test` via bash | hook denies; session continues with denial result returned to model |
| S4 | repeat S1 against real Anthropic API (gated `RUN_LIVE_API=1`) | same success criteria; cost recorded in ledger |
| S5 | resume a saved JSONL transcript and re-run last turn | output matches golden file (modulo timestamps) |

S1–S3 must run on every CI build. S4 runs nightly. S5 runs on PRs that touch `session/`.

## Done definition

Phase 1 is done when:

1. All 20 PRs (0.1–0.8, 1.1–1.12) are squash-merged.
2. Phase 0 integration tests pass.
3. Phase 1 e2e tests S1, S2, S3, S5 pass on `main`.
4. Loop SLOC ≤ 300 (measured with `wc -l genie/loop.py`).
5. Tag `v0.1.0-phase1` cut from the head of `main`.
