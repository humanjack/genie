# genie — Implementation Detail (v0.1.0-phase1)

> The file-by-file, code-level companion to [DESIGN.md](./DESIGN.md). Where DESIGN explains
> *what* exists and *why*, this document explains *how* it is built: public API surfaces, key
> functions, exact behaviors, invariants, and the gotchas that bite. Everything here describes
> the code shipped at tag `v0.1.0-phase1` and was verified against the source, not from memory.

**How to read:** §2–§3 are setup. §4 is the per-subsystem walkthrough (the bulk). §5 is a
consolidated invariants/gotchas cheat-sheet. §6 is testing + metrics. §7 is extension recipes.
§8 is PR traceability and deferred work.

---

## 1. Snapshot

| Metric | Value |
|---|---|
| Source lines (`genie/`, incl. docstrings) | 4,160 across 32 modules |
| `loop.py` | **293 lines** (SPEC target ≤300) |
| Tests | **374 passing**, 6 skipped (live-API, gated by `RUN_LIVE_API`) |
| Coverage | **99%** total; every package ≥90% (lowest: `edit_file.py` 90%, `cli.py` 92%, `agent.py` 95%) |
| CI matrix | Python 3.11 + 3.12; ruff check + `ruff format --check` + pyright + pytest `--cov-fail-under=70` |
| Runtime deps | `anthropic`, `openai`, `pydantic`, `pydantic-settings`, `httpx`, `rich`, `structlog` |

---

## 2. Repository layout

```
genie/
├── __init__.py
├── cli.py                  228  entrypoint: chat-once (P0) + code (P1) subcommands
├── agent.py                229  coding-agent wiring (build_registry) + REPL core
├── loop.py                 293  the ReAct loop — run_turn, dispatch_tool_calls
├── config.py               230  typed settings: defaults < TOML < env
├── providers/
│   ├── base.py             125  ChatMessage, ChatChunk, ProviderClient (ABC)
│   ├── fake.py             203  FakeProvider — scripted streams (the test seam)
│   ├── anthropic_client.py 261  Anthropic Messages API adapter
│   ├── openai_client.py    312  OpenAI chat-completions adapter
│   └── factory.py           94  provider_factory("provider:model")
├── tools/
│   ├── base.py             182  Tool model + @tool decorator + schema derivation
│   ├── registry.py         195  ToolRegistry: register, specs_for, call
│   ├── result.py            95  ToolResult + truncate (SPEC §5.4 layer 1)
│   └── builtins/
│       ├── _workspace.py     43  confine() — the one audited path guard
│       ├── read_file.py      67  make_read_file(root)
│       ├── write_file.py     53  make_write_file(root)        [dangerous]
│       ├── edit_file.py     125  make_edit_file(root)         [dangerous]
│       └── bash.py           85  make_bash(sandbox, timeout)  [dangerous]
├── hooks/
│   └── manager.py          247  HookManager, Hook, HookOutcome, BlockedError, EVENTS
├── sandbox/
│   ├── base.py             123  SandboxBackend (ABC), ExecResult, SandboxError
│   ├── local_subprocess.py 206  LocalSubprocessBackend
│   └── recording.py         96  RecordingBackend (the test seam)
├── session/
│   ├── session.py          274  Session.create/resume, meta.json, append
│   └── transcript.py       136  Transcript — append-only JSONL r/w
└── utils/
    └── logger.py           167  structlog config + secret redaction

tests/                            374 tests mirroring the package tree
├── e2e/test_user_stories.py       S1–S5 — drive the whole stack as a user
├── integration/test_phase0.py     provider abstraction end-to-end
└── test_*/                         one suite per module
```

---

## 3. Build, test, and CI

```bash
make install      # python3.11 -m venv .venv ; pip install -e ".[dev]"
make lint         # ruff check + ruff format --check
make typecheck    # pyright (pinned to .venv — see gotcha below)
make test         # pytest
make cov          # pytest --cov=genie --cov-fail-under=70
make all          # lint + typecheck + cov  (the pre-merge gate)
```

**Tooling config** (`pyproject.toml`):
- **pytest**: `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed), `--strict-markers`.
- **coverage**: `source = ["genie"]`; excludes `pragma: no cover`, `if TYPE_CHECKING:`,
  `raise NotImplementedError`, `...`.
- **ruff**: line-length 100; selects `E, F, I, W, UP, B, SIM, RUF`; ignores `E501`.
- **pyright**: `venvPath="."`, `venv=".venv"`, `pythonVersion="3.11"`, basic mode.

**Environment gotchas** (not obvious from the repo):
- **pyright picks the wrong interpreter** if not pinned. Bare `pyright` selects system Python;
  the pyproject pins `venvPath`/`venv` and CI builds a `.venv` so the right interpreter and
  installed deps are found.
- **Don't add `# noqa` for unselected rules.** ruff's `RUF100` flags *unused* noqa directives,
  so a `# noqa: BLE001` (BLE isn't selected) turns CI red. Fix the code, don't suppress.
- **Run ruff with visible output.** A `ruff check … && echo OK` pattern hides the failure
  behind the `&&`; a silent lint failure once shipped a red CI. Let it print.

---

## 4. Per-subsystem implementation

### 4.1 Providers (`genie/providers/`)

**`base.py` — the contract.** Two dataclasses and one ABC (see DESIGN §5.1–5.2 for the
contract rationale). `ProviderClient.stream` is declared as an `async def` with a trailing
unreachable `yield ChatChunk()` so type-checkers infer `AsyncIterator` correctly; the body
`raise NotImplementedError`. Subclasses set class attributes `name` and `model`.

**`fake.py` — `FakeProvider`.** The replaceability proof and the workhorse of every test.
- Construct with `turns` as either a **list of turns** (`list[list[ChatChunk]]`, multi-turn) or
  a **flat `list[ChatChunk]`** (single turn). `_normalize` distinguishes them by sniffing
  `isinstance(turns[0], ChatChunk)`.
- Each `stream()` replays the next turn and advances a cursor; over-driving raises `IndexError`
  with a clear message (never hangs).
- Records every call's `messages/tools/system/...` into `.calls` so tests assert what the loop
  sent.
- Three builders produce **real wire-shaped** fixtures: `from_text(text, chunks=, usage=)`
  (splits into `delta_text` pieces + terminal `stop`); `with_tool_call(name, args, call_id=)`;
  `with_tool_calls([(name,args)], call_ids=)` (assigns distinct slot `index`es — the parallel-
  dispatch fixture). `_tool_call_fragments` splits args into partial-JSON pieces (id/name on the
  first fragment only) — exercising the loop's reassembly exactly as a live provider would.

**`anthropic_client.py` — `AnthropicClient`.**
- **Lazy client.** The SDK client is built on first `stream()` (or injected via `client=` for
  tests), so constructing the object never needs a key. Key resolves via
  `settings.require_api_key("anthropic", os.environ)` or falls back to `$ANTHROPIC_API_KEY`.
- **`_translate_message`**: a `role="tool"` message → a `{role:user, content:[{type:tool_result,
  tool_use_id, content}]}`; an assistant message with `tool_calls` → text block(s) +
  `{type:tool_use, id, name, input}` blocks; everything else passes through.
- **`_apply_cache_breakpoint`**: promotes string content to a text block and tags the last block
  with `cache_control: {type: ephemeral}`.
- **Stop-reason map**: `end_turn → stop`, `tool_use → tool_calls`; unmapped reasons pass through.
- **`stream` event mapping**: `message_start` captures input/cache usage; a `tool_use`
  `content_block_start` opens a slot keyed by block `index`; `text_delta → delta_text`;
  `input_json_delta.partial_json → arguments_delta`; the terminal `message_delta` yields
  `finish_reason` + assembled `usage`.
- `count_tokens` is a `chars // 4` estimate (min 1) — deliberately offline; precise async
  counting is deferred (issue #47).

**`openai_client.py` — `OpenAIClient`.** Chat-completions only.
- **`responses` mode raises `NotImplementedError`** (deferred, issue #49). Mode resolves from
  `settings.provider.openai.api`, default `"chat_completions"`.
- **`_translate_tools`**: neutral `{name, description, input_schema}` → `{type:function,
  function:{name, description, parameters}}`; returns `None` for an empty list so the SDK call
  omits `tools`.
- **`_translate_tool_call`** (the PR #64 fix): neutral `{id, name, arguments:<dict>}` →
  `{id, type:function, function:{name, arguments:<JSON string>}}`. Already-native calls (carrying
  a `function` key) pass through unchanged. **This is the translation that must live in the
  adapter, not the loop** (DESIGN §5.3).
- **`_translate_messages`**: `system` → leading `{role:system}` message (chat-completions has no
  top-level system slot); `role="tool"` carries `tool_call_id`.
- **The separate-usage quirk**: `_map_chunk` returns *zero or more* `ChatChunk` per OpenAI
  chunk. The `finish_reason` chunk and the usage chunk are typically distinct events — usage
  arrives last with `choices == []`. Requested via `stream_options={"include_usage": True}`.

**`factory.py` — `provider_factory(spec, *, settings=, **kwargs)`.**
- Splits `spec` on the **first** colon (`"anthropic:claude-sonnet-4-6"` → `("anthropic",
  "claude-sonnet-4-6")`); the model may itself contain colons.
- `_REGISTRY` maps name → loader. Real adapters are imported **lazily inside their loaders**, so
  importing the factory never imports the SDKs and a missing adapter module degrades to a clear
  `RuntimeError` only when that provider is requested.
- No colon → `ValueError`; unknown provider → `ValueError` listing supported names.
- **Adding a provider is one line** in `_REGISTRY`.

### 4.2 Tools (`genie/tools/`)

**`base.py` — `Tool` + `@tool`.**
- `Tool` is a pydantic model: `name, description, input_schema, handler, sequential=False,
  dangerous=False, tags=[], max_result_chars=8192` (`arbitrary_types_allowed=True` for the
  handler callable).
- `@tool(*, name=, description=, sequential=, dangerous=, tags=, max_result_chars=)` wraps an
  async function:
  - Rejects a non-coroutine function (`TypeError`).
  - Description defaults to the docstring; absent both → `ValueError`.
  - `_schema_from_signature` derives `input_schema` via **`get_type_hints(func)`** (resolves
    PEP-563 stringized annotations, so `int | None` builds a valid schema instead of crashing on
    a forward-ref) → `pydantic.create_model` → `model_json_schema()` with the top-level `title`
    stripped. Every parameter **must** be annotated; `*args`/`**kwargs` are rejected; unresolved
    hints raise a clear decoration error.
  - The wrapped handler coerces a `str` return to `ToolResult.text`; a non-`ToolResult`/`str`
    return raises `TypeError`. (Strict on input, lenient on output.)

**`result.py` — `ToolResult`.** Dataclass `content, is_error=False, metadata={}`. Factories
`.text(content, **metadata)` and `.error(message, **metadata)`. `truncate(max_chars)`:
clamps `max_chars = max(0, max_chars)`; returns `self` if under cap; else `head = content[:n//2]`
+ `\n…[truncated {elided} chars]…\n` + `tail = content[-(n-n//2):]`, carrying `is_error` and a
shallow `metadata` copy.

**`registry.py` — `ToolRegistry`.**
- `register(tool)` — duplicate name raises `ValueError` (no silent shadowing); `register_all`,
  `get` (`KeyError` listing known names), `names`, `__contains__`, `__len__`.
- `specs_for(provider_name)` returns the **neutral** spec list for every provider via the
  `_TRANSLATORS` table (all three entries = `_spec_neutral`; see DESIGN §5.3 for why
  pre-wrapping here would double-translate). Unknown provider → `ValueError`.
- **`call(name, args)`** — the error contract: `await tool.handler(**args)`, catch **any
  `Exception`** → `ToolResult.error(str(exc))`, then `result.truncate(tool.max_result_chars)`.
  So one misbehaving tool can never crash a parallel batch. A missing *tool* (vs. a failing one)
  is a programming error and still raises `KeyError` from `get`.

**`builtins/` — the four tools, via factories.** Each `make_*(root | sandbox)` binds the
non-model state and returns a configured `Tool`:
- **`read_file`** (`tags=["fs"]`, `max_result_chars=16000`): `confine` → exists/is-file checks →
  `read_text(errors="replace")`; optional `offset`/`limit` line slice (0-based, clamped). Escape,
  missing, and non-regular-file each return an error result.
- **`write_file`** (`dangerous`): `confine` → refuse a directory → `mkdir(parents=True)` →
  `write_text` → `"wrote N bytes to {path}"`.
- **`edit_file`** (`dangerous`): anchor-based exactly-once replace. Empty `old` refused; file
  must exist; `text.count(old)` must be exactly 1 (0 → "anchor not found", >1 → "ambiguous …
  include more context"); on success returns a `@@ lines a-b @@` view of the changed region
  (clipped). The exactly-once rule is the safety property (DESIGN §11).
- **`bash`** (`tags=["shell"]`, `dangerous`): `sandbox.exec(command, timeout=)`. A **nonzero exit
  is NOT a tool error** — output is returned (prefixed `exit code: N`) for the model to react.
  Only `SandboxError` → `ToolResult.error`. `_format` adds `timed out after Ns` /
  `[output truncated]` notices.

**`_workspace.py` — `confine(root, path)`.** The single audited containment helper (DESIGN §5.6):
`(Path(root).resolve() / path).resolve()` then `is_relative_to` the resolved root; failure →
`WorkspaceEscape` naming only `path`. Resolve-before-check is what makes it symlink-safe.

### 4.3 Hooks (`genie/hooks/manager.py`)

- **`EVENTS`** — the 8-event frozenset (DESIGN §5.4).
- **`HookOutcome(block=False, block_reason=None, mutated_payload=None)`** with `proceed()` and
  `blocked(reason)` classmethods. A hook may also return `None` (implicit proceed).
- **`Hook`** — a `@runtime_checkable` Protocol: `name: str`, `events: list[str]`, async
  `__call__(event, **payload) -> HookOutcome | None`. Structural typing keeps it light (a class
  or any object with these members qualifies).
- **`BlockedError(reason)`** — carries `.reason`; raised by `run_or_raise`.
- **`HookManager`** (a dataclass holding `hooks: list`):
  - `register(hook)`; `on(event, fn)` wraps a bare callable in `_CallableHook` (validates the
    event name).
  - **`run(event, **payload)`** — iterate in registration order, skip hooks not subscribed to
    `event`, await each with the **running** (possibly-mutated) payload. First `block=True`
    returns immediately. Otherwise merge each `mutated_payload` (`dict.update`) into both the
    running payload and an accumulator, returning `proceed(merged or None)`. Unknown event →
    `ValueError`.
  - **`run_or_raise(event, **payload)`** — calls `run`, raises `BlockedError` on a block. The
    fail-closed variant for `before_*`.
  - **Documented caveat**: the payload copy is *shallow*. Replacing a key via `mutated_payload`
    is isolated, but mutating a *nested* object in place (e.g. `payload["call"].args`) affects
    the caller's object. Phase-2 hooks should mutate via the returned `mutated_payload`.

### 4.4 Sandbox (`genie/sandbox/`)

**`base.py`.** `SandboxError`; `ExecResult(returncode, stdout, stderr, truncated=False,
timed_out=False)` with an `.output` property (stdout then stderr, never pre-merged);
`SandboxBackend.exec(command, *, cwd=None, env=None, timeout=30.0)` ABC. The class docstring is
the **security contract** every backend must honor (DESIGN §5.5).

**`local_subprocess.py` — `LocalSubprocessBackend(root, *, env_allowlist=, max_output_bytes=8192)`.**
- Module constants: `DEFAULT_ENV_ALLOWLIST = (PATH, HOME, LANG, LC_ALL, TERM)`;
  `TIMEOUT_RETURNCODE = 124`; `DRAIN_TIMEOUT = 2.0`.
- `root` is `Path(root).resolve()` once at construction.
- **`_resolve_cwd`**: `None` → root; relative joined under root; `.resolve()` (expands symlinks);
  require `== root or is_relative_to(root)` else `SandboxError` *before spawning*.
- **`_build_env`**: project only allowlisted keys from `os.environ`, overlay caller `env`.
- **`_cap`**: per-stream truncate to `max_output_bytes`, `decode(errors="replace")`.
- **`exec`**: `asyncio.create_subprocess_shell(..., start_new_session=True)` (child leads its own
  process group); `asyncio.wait_for(proc.communicate(), timeout)`. On `TimeoutError` →
  `_terminate` and report `returncode=124, timed_out=True`. A `cwd` that passed confinement but
  doesn't exist → `SandboxError`.
- **`_terminate`**: `os.killpg(pid, SIGKILL)` (reaps grandchildren a bare `proc.kill()` would
  orphan) then a **bounded** post-kill drain (`wait_for(communicate(), DRAIN_TIMEOUT)`) so a
  session-escaped child holding the pipe (e.g. via `setsid`) can't block `exec()` past the
  timeout — it drops trailing output rather than hang. (This bounded drain was an
  adversarial-review fix.)

**`recording.py` — `RecordingBackend`.** Construct with a **list** (replayed in call order) or a
**dict** (keyed by exact command string). Records `command/cwd/env/timeout` of every call into
`.calls`. Over-driving raises `IndexError` (list exhausted) / `KeyError` (unmapped command) —
never a silent default.

### 4.5 Session & transcript (`genie/session/`)

**`transcript.py` — `Transcript(path)`.** Append-only JSONL over `ChatMessage`.
- `append(message, *, ts=None, usage=None)`: writes `{role, content, tool_calls, tool_call_id}`
  + optional `ts`/`usage` (omitted, not null, when absent); opens in append mode and **flushes**
  so the line survives an immediate crash.
- `read_records()`: skips blank lines; a malformed line is **skipped and logged**
  (`transcript_skipped_malformed_line` with `line_no`/`trailing`) rather than raising — a torn
  final write can't poison prior history. Missing file → `[]`.
- `read()`: `read_records` → `ChatMessage` objects, dropping `ts`/`usage`. Round-trips.

**`session.py` — `Session`.** `SessionError` for load failures.
- **`create(root, *, id, model, working_dir=None, parent_id=None, started_at=None)`**: makes
  `<root>/<id>/` with `meta.json` + empty `transcript.jsonl`. `working_dir` defaults to the
  session dir. **Nothing reads the clock or mints a uuid** — `id`/`started_at` are caller-supplied
  (determinism, DESIGN §5.7).
- **`resume(root, id)`**: reads `meta.json` (absent → `FileNotFoundError`; corrupt or
  missing-key → `SessionError`) and replays the transcript into `.messages`.
- `append(message, *, ts=, usage=)` keeps in-memory `.messages` and the on-disk transcript in
  lockstep. `materialize_messages()` returns a shallow copy — the Phase-3 context-pipeline seam.
- `meta.json` = `{id, parent_id, model, working_dir, started_at?}`.

### 4.6 The loop (`genie/loop.py`, 293 lines)

The centerpiece. Holds no policy and no provider knowledge.
- **`ToolCall(id, name, args={}, parse_error=None)`** and **`TurnResult(stopped, last_message,
  iterations, stop_reason)`** dataclasses.
- **`_collect_assistant_turn`**: consumes the stream, concatenating `delta_text` (forwarding each
  fragment to `on_text_delta` for the live REPL) and accumulating `tool_call_delta` by `index`
  via `_accumulate_tool_delta` (`id`/`name` captured once, `arguments_delta` appended).
  `_finalize_tool_call` parses each slot: empty/whitespace → `{}`; a `JSONDecodeError` is **not
  raised** — the call keeps `args={}` and records `parse_error` for the model to see and fix.
  Slots emit in first-seen order. Returns `(assistant_message, tool_calls, usage)`.
- **`dispatch_tool_calls`**: the **Pi rule** — if any call targets a `sequential` tool the whole
  batch runs serially; else `asyncio.gather`. Either way results preserve `calls` order
  (`zip(..., strict=True)`). `_call_one` resolves a `parse_error` / unknown tool / hook veto into
  an error result, never an exception.
- **`run_turn(session, provider, registry, hooks, *, system=None, max_tokens=4096,
  max_iterations=50, on_text_delta=None)`**: the cycle in DESIGN §6. `before_model_call` veto →
  `stop_reason="model_call_blocked"`; no tool calls → `"model_stopped"`; budget exhausted →
  `stopped=False, "max_iterations"`.

### 4.7 Agent wiring + REPL (`genie/agent.py`)

- **`build_registry(root, *, bash_timeout=30.0)`**: constructs a `LocalSubprocessBackend(root)`
  and registers `read_file, write_file, edit_file` (bound to `root`) + `bash` (bound to the
  sandbox), in that order.
- **`load_system_prompt()`**: reads `prompts/coding_system.md` relative to the repo root; on
  `OSError` (e.g. a wheel that doesn't ship it) returns a short built-in fallback string — the
  session always has a usable persona.
- **`ToolCallDisplay(write=print)`**: the US-2 observability hook. Subscribes to
  `before/after_tool_call`; prints `→ name(args)` then `✓`/`✗ name: <preview>`. Purely
  observational — always returns `None`, can never block. `events` is set as an **instance**
  attribute in `__init__` (satisfies the Protocol's instance-level `events` and dodges ruff
  RUF012's mutable-class-default rule). Args/result previews are flattened + clipped
  (80/120 chars).
- **`run_code_session(provider, registry, hooks, session, *, system, read_input, write=print,
  max_iterations=50)`**: the REPL. `read_input()` returns the next line or `None` (EOF/quit);
  `/exit`/`/quit` and blank-line handling; appends a `user` message; drives one `run_turn`
  streaming text to stdout live; a per-turn `KeyboardInterrupt` is caught (aborts the *turn*,
  not the session). Input is injected so tests drive it with no terminal.

### 4.8 CLI (`genie/cli.py`)

- **`main(argv=None)`**: no args / `-h` → per-phase help (exit 0); unknown command → exit 2;
  configures logging (`--json-logs`), dispatches.
- **`chat-once`** (`_cmd_chat_once` → `run_chat_once`): loads config, builds a provider via the
  factory, streams one reply to stdout (flushing per chunk), prints a dim `[usage]` line to
  stderr if present.
- **`code`** (`_cmd_code`): builds provider + `build_registry(root)` + a `HookManager` with
  `ToolCallDisplay`; creates a session under **`~/.genie/sessions/<uuid4().hex>`** (the one place
  a uuid is minted — kept out of the deterministic modules) with `working_dir = root`; runs the
  REPL via the overridable `_READ_INPUT` seam.
- **Boundary error handling**: `KeyboardInterrupt` → 130; any other `Exception` → one clean
  stderr line (no traceback), logged with `error_type`, exit 1.

### 4.9 Config (`genie/config.py`)

- Pydantic `Settings` aggregating `provider, loop, tools, sandbox, approval, memory, skills`
  sections — every SPEC §13 key has a typed default.
- **`load_config(path=None, *, env=None)`**: precedence **defaults < TOML < env**. Missing TOML
  → pure defaults (never an error). Only env override today: `GENIE_PROVIDER_DEFAULT`. `env`
  defaults to `os.environ` but is injectable for pure tests.
- **`provider_parts()`**: splits `provider.default` on the first `:`, **strips whitespace** each
  side, raises `ValueError` on a missing colon / empty side.
- **`resolve_api_key(name, env)`**: returns the key or `None` (empty string also → `None`);
  **`require_api_key(name, env)`**: raises `ValueError` naming the exact env var.
- OpenAI `api` defaults to `"chat_completions"` so a default-configured `openai:` run succeeds.
- TOML reading is isolated in `_read_toml` — the pluggable format seam.

### 4.10 Logger (`genie/utils/logger.py`)

- **`SECRET_KEYS`** frozenset + **`_is_secret_key`**: case-insensitive, matches on word
  *components* (split on `[-_.\s/]+`) and adjacent component pairs — so `api_key`, `x-api-key`,
  `auth_token`, `Authorization` are caught while `input_tokens`/`output_tokens` are **preserved**
  (they're cost telemetry, not secrets). This component-matching was a P0 review fix.
- **`_redact_secrets`** processor + **`_redact_value`**: redacts secret-named keys at the top
  level and **recursively** through nested dicts/lists → `"***"`.
- **`configure_logging(*, json_output=False, level="INFO")`**: validates the level (`ValueError`
  otherwise); builds the processor chain ending in `ConsoleRenderer` or `JSONRenderer` (the
  pluggable seam); `cache_logger_on_first_use=False`. **Idempotent** — structlog *replaces* the
  chain, so repeated calls never stack processors.
- `get_logger(name, **ctx)`, `bind_session(logger, session_id)`.
- **Known limit (documented)**: redaction is *key*-based; a secret embedded in free text or
  carried as a benign key's value isn't caught — pass secrets as values of well-named keys.

---

## 5. Invariants & gotchas (cheat-sheet)

The things that bite when you forget them — every one verified in the code:

1. **Translate tool-calls in the adapter's `stream()`, never in the loop/registry.** The loop
   emits neutral `{id, name, arguments:<dict>}`; pre-translating double-translates and breaks
   OpenAI multi-turn (the PR #64 bug).
2. **Streaming tool args are partial-JSON string fragments addressed by `index`** — never a
   pre-parsed dict. Reassemble by index, `json.loads` at turn end.
3. **A nonzero shell exit is not a tool error.** `bash` returns the output; only `SandboxError`
   is an error result. (So the model sees failing-test output and reacts.)
4. **`edit_file` anchors must be exactly-once.** 0 or >1 matches refuse without writing.
5. **`@tool` requires a type annotation on every parameter** and uses `get_type_hints` so
   `int | None` works under `from __future__ import annotations`. `*args`/`**kwargs` rejected.
6. **Workspace `confine()` and sandbox `_resolve_cwd` resolve *before* the containment check** —
   that's what makes them symlink-safe. Don't reorder.
7. **`before_*` hooks fail closed** (`run_or_raise`); a raising hook propagates. `after_*` use
   `run`.
8. **HookManager payload copy is shallow** — mutate via returned `mutated_payload`, not in place.
9. **Sessions don't mint time/ids**; the CLI does (`uuid4().hex`). Keep `datetime.now`/`uuid4`
   out of `session/`, `agent.py`, `loop.py` so replay stays deterministic.
10. **Transcript reads tolerate a torn final line** (skip + log). Don't "fix" this into a raise.
11. **Don't add `# noqa` for unselected ruff rules** (RUF100 → red CI). Pin pyright to `.venv`.
12. **`max_iterations` is a Phase-1 stopgap**, slated to become the `iteration_budget` hook.

---

## 6. Testing strategy & metrics

**Layered, mirroring the package tree** (`tests/test_<module>/...`):
- **Unit tests** for every public function, mocking external deps through the *abstract base*
  (inject a `RecordingBackend`, a `FakeProvider`, a fake SDK `client=`).
- **Replaceability tests** — each subsystem is exercised against both its real and fake
  implementation, proving the abstraction holds.
- **Loop tests** drive `run_turn` with scripted `FakeProvider` turns: tool-call reassembly,
  parallel vs. sequential dispatch, parse-error feedback, unknown tool, hook veto, the
  `max_iterations` budget, and the stop reasons.
- **Integration** (`tests/integration/test_phase0.py`) — the provider abstraction end-to-end.
- **E2E** (`tests/e2e/test_user_stories.py`) — S1–S5 drive the **whole real stack** (real loop,
  tools, sandbox, session) with a scripted `FakeProvider` standing in for the model's decisions,
  asserting **real outcomes**: a fix lands on disk (S1), a created file runs (S2), a denied
  `rm -rf precious.txt` leaves the sentinel intact (S3), a resumed session replays byte-for-byte
  (S5). S4 hits the live Anthropic API and is **skipped unless `RUN_LIVE_API`** is set.

**No internet in unit/CI tests** — all 6 networked tests are gated behind `RUN_LIVE_API`.

**Adversarial review was the real quality gate.** Each PR was reviewed by a subagent tasked to
*find real bugs / falsify the tests*. It caught cross-component contract bugs that the
fakes/unit tests missed — the OpenAI tool-call translation (#64), the parallel-tool-call missing
`index`, the PEP-563 `Optional` schema crash, the sandbox setsid-escape timeout hang, the
torn-line history loss, the nested-secret log leak, and a tautological S3 assertion
(`rm -rf .`, which `rm` refuses, replaced with `rm -rf precious.txt` and falsification-confirmed
to actually delete when unguarded). All fixed before merge.

**Coverage** (`make cov`, gate `--cov-fail-under=70`): 99% total, every package ≥90%. Lowest
files: `edit_file.py` 90%, `cli.py` 92%, `agent.py` 95% — uncovered lines are mostly the
prompt-file-missing fallback and rare OSError branches.

---

## 7. Extension recipes

The seams, concretely. None require touching `loop.py`.

**Add a provider.** Implement `ProviderClient` (translate in `stream()`, emit index-addressed
deltas), then add one line to `factory._REGISTRY`:
```python
_REGISTRY["mistral"] = _load_mistral   # lazy import inside the loader
```

**Add a tool.** Write an async function, decorate it, register it:
```python
@tool(name="grep", tags=["fs"])
async def grep(pattern: str, path: str) -> ToolResult:
    """Search files for a pattern."""        # docstring becomes the description
    ...
registry.register(grep)                        # or add to build_registry()
```
For stateful tools (a root, a client), follow the `make_*` factory pattern so the state never
leaks into the model-facing schema.

**Add a hook** (Phase-2 style). Implement the `Hook` protocol and register it; a `before_*`
hook returning `HookOutcome.blocked(reason)` vetoes the action (the loop feeds the reason back
to the model):
```python
class ApprovalHook:
    name = "approval"
    def __init__(self): self.events = ["before_tool_call"]
    async def __call__(self, event, **payload):
        call = payload.get("call")
        if getattr(call, "name", None) == "bash" and _is_dangerous(call.args["command"]):
            return HookOutcome.blocked("needs approval")
        return None
hooks.register(ApprovalHook())
```

**Add a sandbox backend.** Subclass `SandboxBackend`, honor the cwd-confinement + env-curation
contract, return an `ExecResult`. Wire it where `build_registry` constructs the backend.

---

## 8. Traceability & deferred work

**22 PRs, built in waves A–F, each its own issue under umbrella epic #1, each adversarially
reviewed before squash-merge.** Landing order:

| PR | Component | Issue/PR# |
|---|---|---|
| 0.1 / 0.1.1 | Scaffold (pyproject, CI, Makefile) + pyright venv pin | #40, #42 |
| 0.2 | Typed config loader | #43 |
| 0.3 | Structured logger + redaction | #44 |
| 0.4 | Provider abstraction: base + FakeProvider + factory | #45 |
| 0.5 / 0.6 | Anthropic / OpenAI clients (parallel) | #48, #50 |
| 0.7 / 0.8 | `chat-once` CLI / Phase-0 integration | #51, #52 |
| 1.1 | Tool base: `@tool`, `Tool`, `ToolResult` | #53 |
| 1.2 / 1.3 | Tool registry / Hook manager (parallel) | #56, #57 |
| 1.4 / 1.5 | Sandbox / Session+transcript (parallel) | #54, #55 |
| 1.6–1.9 | `read_file` (+`_workspace`) / `write_file` / `edit_file` / `bash` | #58, #59, #60, #61 |
| 1.10 | The ReAct loop | #62 |
| (fix) | OpenAI neutral→native tool_calls translation | #63 → #64 |
| 1.11 | `genie code` interactive REPL | #65 |
| 1.12 | Phase-1 e2e user-story suite (S1–S5) | #66 |

Test issues #29–#39 were closed with delivery notes (Phase-0 tests rode their impl PR's
`Closes`; Phase-1 test issues were closed manually as delivered-in-impl-PR). Tags:
`v0.1.0-phase0`, `v0.1.0-phase1`.

**Deferred follow-ups (open, Phase 2):**
- **#46** — config `extra='forbid'` + wrap `TOMLDecodeError` in a friendly error.
- **#47** — provider async `count_tokens` (replace the `chars//4` estimate) + name/model
  enforcement.
- **#49** — OpenAI Responses API mode (`provider.openai.api = "responses"`, currently
  `NotImplementedError`).

**Next phase (per SPEC §Phase 2):** the policy hooks the chokepoint was built for —
`approval`, `iteration_budget` (replacing `max_iterations`), `cost_ledger`, `policy` — plus
`--unsafe` and `genie status`. None require loop changes; that's the test of the design.
