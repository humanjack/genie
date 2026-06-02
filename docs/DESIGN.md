# genie — Design Document (v0.1.0-phase1)

> The architecture and design rationale for what is **actually built and shipped** at tag
> `v0.1.0-phase1`. Where [SPEC.md](./SPEC.md) is the aspirational design across seven phases,
> this document describes the realized system: Phase 0 (skeleton + provider abstraction) and
> Phase 1 (the ReAct loop + four tools). For file-by-file specifics see
> [IMPLEMENTATION.md](./IMPLEMENTATION.md).

**Status:** Phase 0 + Phase 1 complete. 374 tests, 99% line coverage, `loop.py` = 293 lines.
**Audience:** anyone reading the code to understand or extend it.

---

## 1. What got built

A small, legible coding agent you can drive against a real repository: it streams a model
turn, dispatches the tool calls the model asks for (read a file, write one, edit one, run a
shell command), feeds the results back, and repeats until the model stops — persisting every
message to a replayable JSONL transcript and gating every tool call through a single hook
chokepoint.

Two entrypoints exist today:

- **`genie chat-once "<prompt>"`** — Phase 0. Stream one reply, no tools. Proves the provider
  abstraction end-to-end.
- **`genie code [path]`** — Phase 1. The interactive coding REPL: chat with the agent in a
  repo and watch every tool call.

The north-star from the spec is met: genie closes a real one-file bug end-to-end (read → edit
→ run test → report), and every decision is recoverable by reading the transcript.

### Deliberately *not* built yet (Phase 2+)

Approval/policy hooks, cost ledger, iteration-budget-as-hook, `AGENTS.md` memory, compaction,
tree sessions/`fork`, `grep`/`glob`/web tools, skills, RAG, sub-agents, MCP, the eval harness,
and the Docker sandbox. The seams for all of them exist (see §6); the implementations are
future phases. This document does not describe them.

---

## 2. Operating principles (and how the code honors them)

The five principles from SPEC §Operating-principles are the load-bearing constraints. Every
design decision below defers to them. Concretely, as realized:

| Principle | How it shows up in the shipped code |
|---|---|
| **Minimal first** | `loop.py` is 293 lines. No feature exists that isn't on the Phase 1 exit criteria. The hook system is *pure mechanism* — zero built-in policy hooks ship yet. |
| **Secure by default** | Every shell command goes through `SandboxBackend.exec` (cwd-confined, env-curated). Every file path goes through `_workspace.confine()`. Every tool call is announced to the hook chain at a `before_tool_call` chokepoint that fails *closed*. |
| **Pluggable everywhere** | Provider, sandbox, tool, and hook are each reached only through an abstract contract. Swapping `AnthropicClient` → `OpenAIClient`, or `LocalSubprocessBackend` → `RecordingBackend`, requires **zero edits** to `loop.py`. |
| **Replaceability is testable** | Every subsystem ships ≥2 implementations from day one: `FakeProvider` + two real adapters; `RecordingBackend` + `LocalSubprocessBackend`. The fakes aren't test scaffolding bolted on later — they're proof the abstraction holds, and they drive the entire e2e suite offline. |
| **Powerful enough to dogfood** | The e2e suite (`tests/e2e/test_user_stories.py`) drives the *real* loop, *real* tools, *real* sandbox, and *real* session against a seeded buggy repo and the fix lands on disk. |

---

## 3. Realized architecture

The Phase 0 + 1 system. Solid boxes are built; the dashed context-pipeline box is a
single method seam (`Session.materialize_messages`) awaiting Phase 3.

```
                 ┌──────────────────────────────────────────────┐
                 │                  Entrypoints                   │
                 │  genie chat-once  (P0)   │   genie code  (P1)  │
                 │            cli.py        │      cli.py         │
                 └───────────────┬──────────┴──────────┬─────────┘
                                 │                      │
                       run_chat_once()          run_code_session()
                       (cli.py, no tools)         (agent.py, REPL)
                                 │                      │
                                 │                      ▼
                                 │            ┌──────────────────┐
                                 │            │   run_turn()     │   the ReAct loop
                                 │            │    loop.py       │   (293 lines)
                                 │            └────────┬─────────┘
                                 │                     │
              ┌──────────────────┼─────────────────────┼────────────────────┐
              ▼                  ▼                     ▼                    ▼
        ┌───────────┐     ┌────────────┐       ┌──────────────┐     ┌──────────────┐
        │ Provider  │     │   Tool     │       │   Hook       │     │  Session +   │
        │  client   │     │ registry + │       │  manager     │     │  transcript  │
        │ (stream)  │     │ dispatch   │       │ (chokepoint) │     │  (JSONL)     │
        └─────┬─────┘     └──────┬─────┘       └──────────────┘     └──────────────┘
              │                  │                                  
   ┌──────────┼─────────┐        │ builtins                        
   ▼          ▼         ▼         ▼                                 
┌──────┐ ┌─────────┐ ┌──────┐  ┌─────────────────────────────┐    ┌──────────────┐
│ Fake │ │Anthropic│ │OpenAI│  │ read/write/edit_file · bash │───▶│   Sandbox    │
└──────┘ └─────────┘ └──────┘  └─────────────────────────────┘    │  backend     │
                                          │                       │ (exec)       │
                                  _workspace.confine()            └──────┬───────┘
                                  (path containment)                     │
                                                              ┌──────────┴─────────┐
                                                              ▼                    ▼
                                                       ┌─────────────┐     ┌─────────────┐
                                                       │ LocalSubproc│     │  Recording  │
                                                       └─────────────┘     └─────────────┘

  cross-cutting:  config.py (typed settings)   ·   utils/logger.py (structlog + redaction)
```

The loop holds **no policy and no provider knowledge**. It composes four contracts —
`ProviderClient`, `ToolRegistry`, `HookManager`, `Session` — and is unaware of who is on the
other end of any of them. That is the whole point: the same `run_turn` runs against a live
SDK adapter and a scripted `FakeProvider` with zero edits.

---

## 4. The eight subsystems — built vs. deferred

SPEC §1 names eight cross-cutting subsystems. Status at this tag:

| # | Subsystem | Status | Where |
|---|---|---|---|
| 1 | Provider client wrapper | ✅ Built | `providers/` (base, fake, anthropic, openai, factory) |
| 2 | Tool registry & dispatcher | ✅ Built | `tools/` (base, registry, result, builtins/) + `loop.dispatch_tool_calls` |
| 3 | Sandbox layer | ✅ Built (local) | `sandbox/` (base, local_subprocess, recording) |
| 4 | Hook / middleware system | ✅ Mechanism built; policy hooks deferred | `hooks/manager.py` |
| 5 | Context pipeline | 🟡 Minimal seam only | `Session.materialize_messages()` |
| 6 | Session / transcript store | ✅ Built (flat; tree deferred) | `session/` (session, transcript) |
| 7 | Skill / extension loader | ⬜ Deferred (Phase 4) | — |
| 8 | Eval harness | ⬜ Deferred (Phase 6) | — |

---

## 5. Load-bearing contracts

These are the small number of interfaces and shapes everything else is built on. Get these
right and the system composes; violate one and things break in subtle, cross-component ways
(several did during development — see IMPLEMENTATION §11). Each is documented in the code at
its definition site.

### 5.1 The provider-neutral contract

`genie/providers/base.py` defines the *only* seam every model is reached through. It exposes
**no provider-native types**:

```python
@dataclass
class ChatMessage:                  # role, content, tool_calls, tool_call_id
@dataclass
class ChatChunk:                    # delta_text, tool_call_delta, finish_reason, usage
class ProviderClient(ABC):
    name: str; model: str
    async def stream(messages, tools, *, max_tokens, temperature, system, cache_breakpoints) -> AsyncIterator[ChatChunk]
    def count_tokens(messages) -> int
```

Messages are plain dataclasses; tool definitions are JSON-Schema `dict`s. **Each adapter
translates to and from its SDK's wire format inside `stream()`.** The loop never learns
which provider it is talking to — this is what makes providers swappable with zero loop
edits.

### 5.2 Index-addressed streaming tool-call deltas

The subtlest contract, and the one that earns the "our own wrapper" decision (SPEC §3.4). A
streamed tool call is **never** delivered as a pre-parsed dict — that would not survive real
streaming. Instead each `ChatChunk.tool_call_delta` is a fragment addressed by an integer
slot `index`:

```python
{
    "index": int,                  # required — which tool-call slot
    "id":   str | None,            # set once, on the slot's first fragment
    "name": str | None,            # set once, on the slot's first fragment
    "arguments_delta": str | None  # partial-JSON to append for this slot
}
```

This shape is the lowest common denominator of both wire protocols:

- **OpenAI** `ChoiceDeltaToolCall` already carries `.index`, `.id` (once), and a streamed
  `.function.arguments` string fragment.
- **Anthropic** uses a `content_block` `index`, a `tool_use` block-start (carrying `id`/`name`),
  and `input_json_delta.partial_json` string fragments.

The consumer (`loop._collect_assistant_turn`) accumulates `arguments_delta` per `index`, and
`json.loads` the joined string when the turn finishes. The index is what lets **parallel**
tool calls interleave on the wire and still reassemble independently.

### 5.3 Neutral tool specs in, neutral tool-calls out — translation lives in the adapter

The registry and loop speak a **neutral shape on both directions**:

- **Tool specs** the loop passes to `stream(..., tools=...)`: `{name, description, input_schema}`
  (`ToolRegistry.specs_for` returns exactly this for *every* provider).
- **Assistant `tool_calls`** the loop appends to history: `{id, name, arguments: <dict>}`.

Each adapter translates these to its own wire format *inside* `stream()` — Anthropic passes
specs through and builds `tool_use` blocks; OpenAI wraps specs as `function` tools and encodes
`arguments` as a JSON string. **Pre-translating in the registry or loop double-translates and
corrupts the spec.** This bit us once (the OpenAI multi-turn `tool_calls` translation, fixed in
PR #64) and is now the explicit rule. The registry keeps a `_TRANSLATORS` table as the seam for
a future provider that genuinely needs registry-side shaping, but today all three entries map
to the neutral spec.

### 5.4 The hook chokepoint

`genie/hooks/manager.py` is **pure mechanism** — the "secure by default" principle made
concrete. The loop never touches the world directly; before it calls the model or a tool, it
announces the event to the `HookManager` and lets registered hooks observe, mutate the payload,
or veto.

Eight lifecycle events (intentionally matching Claude Code's hook schema for free
interoperability): `session_start`, `session_end`, `before_model_call`, `after_model_call`,
`before_tool_call`, `after_tool_call`, `model_error`, `tool_error`.

Decided here so the loop and future policy hooks can rely on it:

- **Registration order is run order.**
- **First block wins.** The first hook returning `block=True` short-circuits the cascade; later
  hooks for that event don't run.
- **`before_*` fails closed.** The loop uses `run_or_raise` at `before_model_call` /
  `before_tool_call`, which raises `BlockedError` on a veto. A *raising* hook (a bug, not a
  clean block) also propagates — so a broken policy hook stops the action rather than silently
  letting it through.
- **`after_*` observes.** Observational events use `run`, where a block has nothing to stop.
- **Payload mutation merges** (`dict.update`, last writer wins) into a running payload later
  hooks see.

US-3 ("fail safe") rides on this: a denied tool call becomes a `blocked: …` tool-result fed
back to the model, so the session continues rather than crashing. The e2e suite proves it with
a `_DenyDangerousBash` hook that blocks `rm -rf` and verifies the sentinel file survives.

### 5.5 The sandbox security contract

`genie/sandbox/base.py` mandates two guarantees every backend MUST honor:

1. **Working-directory confinement.** `cwd` defaults to the backend's root and must resolve to
   the root or a subpath; a `..` traversal, absolute path, or symlink escape raises
   `SandboxError` *before any process is spawned*. Symlinks are resolved before the check.
2. **Environment curation.** The child does not inherit the parent environment. The backend
   starts from a curated allowlist (`PATH`, `HOME`, `LANG`, `LC_ALL`, `TERM`) and overlays only
   what the caller explicitly passes — so `AWS_*` / `GH_TOKEN` never leak by default.

**Honest scope:** cwd confinement controls only the command's *starting directory*. It does not
sandbox the filesystem or network — `cd / && cat /etc/passwd` still works. Real FS/network
isolation is the deferred Docker backend's job (SPEC §6.3); per-command allow/deny is the
deferred bash-AST guard's (§6.2). For v1 the (Phase 2) approval hook is what gates risk; this
backend is the cwd/env/timeout primitive beneath it. The code says so at the top of
`local_subprocess.py` — read before trusting it.

### 5.6 Workspace path confinement

The file tools (`read/write/edit_file`) share **one** audited containment helper,
`genie/tools/builtins/_workspace.confine(root, path)`. It resolves `(root / path)` — expanding
symlinks and `..` *before* the `is_relative_to(root)` check — so symlink-out, traversal-out,
and absolute-elsewhere paths are all rejected with a `WorkspaceEscape`. The error names only
the supplied `path`, never the absolute host root, to avoid leaking host paths to the model.
Centralizing this means the symlink guarantee is locked by one test suite instead of three
diverging copies (this was extracted after review found drift between the tools).

### 5.7 Session determinism + crash-durable transcript

`genie/session/` keeps **identity and time injectable**: `id` and `started_at` are passed in,
never generated with `uuid4`/`datetime.now` inside the module, so construction and replay are
byte-for-byte deterministic and testable. (The CLI is where a real `uuid4().hex` is minted.)

The transcript is append-only JSONL, **flushed after every line**, so an interrupted run still
leaves every committed message on disk — the property `resume` depends on. `read_records`
*skips and logs* a malformed line (e.g. a torn final write from a crash) rather than raising,
so one interrupted append can never make the prior history unreadable.

### 5.8 `ToolResult` — the single tool return shape

Every tool returns a `ToolResult(content, is_error, metadata)` (`tools/result.py`). A handler
may return a bare `str` (coerced to `ToolResult.text`) but nothing else. `is_error` carries the
failure flag *without* raising — the loop feeds error content back to the model so it can react
(errors are never retried). `truncate(max_chars)` implements SPEC §5.4 layer 1: head+tail
slices joined by a `…[truncated N chars]…` marker. Layers 2 (spill-to-disk) and 3 (per-turn
budget) are deferred to Phase 3.

---

## 6. Control flow: anatomy of one `run_turn`

The ReAct cycle, exactly as `loop.run_turn` runs it:

```
  ┌─▶ 1. messages = session.materialize_messages()
  │      tools    = registry.specs_for(provider.name)        # neutral specs
  │
  │   2. hooks.run_or_raise("before_model_call", …)          # veto ⇒ stop("model_call_blocked")
  │
  │   3. _collect_assistant_turn(provider.stream(…)):         # stream + reassemble
  │        • concat delta_text     → assistant content (+ on_text_delta for live REPL)
  │        • accumulate tool_call_delta by index
  │        • json.loads each slot's args (parse failure ⇒ ToolCall.parse_error, not a raise)
  │        • capture usage from terminal chunk
  │
  │   4. session.append(assistant_msg, usage)                 # persist (in-memory + JSONL)
  │      hooks.run("after_model_call", …)
  │
  │   5. if no tool_calls:  return stop("model_stopped")      # ← the normal exit
  │
  │   6. dispatch_tool_calls(calls, …):                       # Pi rule: any sequential ⇒ serial,
  │        for each call:                                     #          else asyncio.gather
  │          • parse_error?  → ToolResult.error
  │          • unknown tool? → ToolResult.error
  │          • before_tool_call veto? → "blocked: …" error   # US-3 fail-safe
  │          • else registry.call(name, args) → after_tool_call
  │
  │   7. append each result as role="tool" keyed to its call id
  │
  └── 8. iterations++; if ≥ max_iterations: return stop("max_iterations")   # Phase-1 stopgap
```

The loop is **intentionally tolerant of model misbehavior**: malformed tool arguments,
hallucinated tool names, and hook denials each become a tool-result the model reads and reacts
to, never an exception that aborts the run. The only hard stops are a vetoed
`before_model_call` and the `max_iterations` budget.

`TurnResult(stopped, last_message, iterations, stop_reason)` reports the outcome, where
`stop_reason ∈ {"model_stopped", "max_iterations", "model_call_blocked"}`.

> **Note on `max_iterations`.** This is a Phase-1 stopgap. SPEC §4 makes the turn budget an
> `iteration_budget` *hook* in Phase 2; at that point the counter is removed in favor of the
> hook owning the policy. It is documented as such in the code so it isn't mistaken for a
> permanent design choice.

---

## 7. The security model

Defense as it exists today, layered from the model outward:

1. **Tool surface is fixed and small.** The model can only call the four registered tools.
   There is no dynamic tool loading, no `eval`, no arbitrary import.
2. **Every tool call passes the hook chokepoint** (`before_tool_call`), which fails closed.
   Phase 2 plugs `approval`/`policy` hooks in here with zero loop changes.
3. **File tools are workspace-confined** (`confine()`), symlink-safe.
4. **Shell commands are cwd-confined and env-curated** (`SandboxBackend`), with a timeout
   (process-group kill, rc=124) and per-stream output caps.
5. **Secrets don't leak into logs.** The structlog chain redacts secret-named keys recursively,
   matching on word components so `api_key`/`x-api-key`/`auth_token` are masked while
   `input_tokens` (cost telemetry) is preserved.
6. **Secrets don't leak into the child env** unless explicitly passed (allowlist projection).

What is **out of scope at v1** and must not be mistaken for protection: filesystem/network
isolation (Docker, Phase 6), per-command bash parsing (AST guard, Phase 4), and the approval
gate itself (Phase 2). The sandbox is the *primitive*; the gate is future work.

---

## 8. The pluggability model

"Replaceability is testable" means every abstract contract ships with at least two concrete
implementations *now*, and the fake/stub is a first-class citizen used to drive real tests —
not an afterthought.

| Contract (ABC / Protocol) | Real implementation(s) | Fake / alternate | Proves |
|---|---|---|---|
| `ProviderClient` | `AnthropicClient`, `OpenAIClient` | `FakeProvider` (scripted streams) | Loop is provider-agnostic; e2e runs offline |
| `SandboxBackend` | `LocalSubprocessBackend` | `RecordingBackend` (scripted results) | Tools shell out with no real process |
| `Hook` (Protocol) | `ToolCallDisplay` | `_DenyDangerousBash` (e2e), `_CallableHook` | Chokepoint works for observe + veto |
| `Tool` | the four builtins (via factories) | any `@tool`-decorated coroutine | New capability = one decorated function |

The wiring point is a single factory call at startup: `provider_factory("provider:model")`
selects the provider, `build_registry(root)` selects the sandbox + tools. Adding a provider is
a one-line edit to `factory._REGISTRY`; adding a tool is one `@tool` function + one
`registry.register`.

---

## 9. Error philosophy

Three categories, handled differently and deliberately:

- **Model misbehavior** (bad JSON args, unknown tool, a tool that fails) → returned to the model
  as a `ToolResult`/error so it can self-correct. Never retried by the loop, never raised.
  (Pattern from all six reference agents in the spec's lineage.)
- **Policy decisions** (a hook blocks) → a clean `BlockedError` caught by the loop and surfaced
  to the model as a `blocked: …` tool-result. The session continues.
- **Programming errors** (unknown provider name, missing tool in `registry.get`, corrupt
  `meta.json`) → raise (`ValueError`/`KeyError`/`SessionError`). These are bugs or operator
  errors, not model behavior, and should fail loudly.

At the CLI boundary, *everything* unhandled collapses to one clean stderr line (no traceback),
logged with its type for triage; `KeyboardInterrupt` exits 130, other errors exit 1, unknown
command exits 2.

---

## 10. Observability

Every component logs through `utils/logger.py` — a thin structlog wrapper with three
guarantees: per-session correlation (`bind_session` stamps `session_id`), recursive secret
redaction (§7), and a pluggable final renderer (human console by default, JSON via
`--json-logs`). `configure_logging` is idempotent — structlog *replaces* the processor chain,
so repeated calls never stack processors.

The replayable transcript is the other half of observability: the north-star's "explain every
decision by reading the trace" is satisfied by `transcript.jsonl` + `Session.resume`, proven by
e2e scenario S5 (resume replays byte-for-byte).

---

## 11. Design decisions worth calling out

- **Our own provider wrapper, not LiteLLM.** We wanted to *own* the translation surface —
  specifically the tool-call delta normalization (§5.2), which is where providers differ most
  and where the loop's correctness lives.
- **Anchor-based `edit_file`, not patch/diff.** A unified diff is brittle against line drift; a
  fuzzy apply can land a hunk in the wrong place. Instead `edit_file` requires an `old` snippet
  that occurs **exactly once** — an ambiguous anchor is refused rather than silently rewriting
  several places. The exactly-once rule is the core safety property.
- **Factories for stateful tools.** The workspace root and sandbox aren't model arguments, so
  `make_read_file(root)`, `make_bash(sandbox)`, etc. bind them at construction and expose only
  the model-facing parameters. This keeps the `@tool`-derived schema clean (no `root` leaking
  into the model's view) and the confinement boundary non-negotiable.
- **`@tool` derives schema from the signature.** The JSON Schema is built from the function's
  typed parameters via pydantic, so schema and implementation can never drift. Every parameter
  *must* be annotated (explicit over magical); a bare `str` return is coerced, anything else
  raises.
- **Async-first.** Tool-call parallelism and streaming both want it; both provider SDKs support
  it natively.
- **`rich` + plain stdin for the REPL**, not `prompt_toolkit` — fastest path to ship Phase 1
  (SPEC open-question #3). The input source is injectable so tests drive it with no terminal.

---

## 12. Traceability

The system was built as 22 single-component PRs across six parallelism waves (A–F), each with a
GitHub issue, each adversarially reviewed before merge, under umbrella epic #1. The PR → component
map and the wave/parallelism strategy are in [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md);
the per-PR landing order and open follow-ups are in [IMPLEMENTATION.md](./IMPLEMENTATION.md) §12.

**Open follow-ups (Phase 2):** #46 (config `extra='forbid'` + wrap `TOMLDecodeError`),
#47 (provider async `count_tokens`), #49 (OpenAI Responses API mode).
