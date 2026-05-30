# genie — Combined Specification

> A from-scratch, Python-native AI coding agent. v1 is a Codex/OpenCode-style coding CLI; v2+ generalizes to a Claude-Cowork-style multi-surface assistant. Built to be understood line-by-line, then extended.

**Authors:** humanjack (Jaeik)
**Status:** Draft v0.2 — 2026-05-29

## Operating principles (load-bearing — all subsequent design defers to these)

1. **Minimal first.** Ship the smallest thing that works. No speculative abstraction; no feature added "because we'll need it." If a feature isn't on the current phase's exit criteria, it does not get code.
2. **Secure by default.** Every action that touches disk, network, or shell goes through the hook chokepoint. Default-deny on the dangerous list. No `--unsafe` shortcuts in the core; only at the CLI surface and only when logged.
3. **Pluggable everywhere.** Every subsystem is reached only through its abstract base class. Concrete implementations are wired by a single factory call at startup. Swapping `AnthropicClient` for `LocalLlamaClient`, or `LocalSubprocessBackend` for `DockerBackend`, must require **zero edits** to `loop.py`, `cli.py`, or any other component.
4. **Replaceability is testable.** For each subsystem we ship at least two implementations from day one (e.g., `FakeProvider` + `AnthropicClient`; `LocalSubprocessBackend` + `RecordingBackend` for tests). If only one impl exists, the abstraction is unproven.
5. **Powerful enough to dogfood.** "Minimal" does not mean "toy." Phase 1 must close a real bug on this repo. If we can't dogfood it, we missed the bar.
**Source material:**
- *Build an AI Agent (From Scratch)* — 10 chapters (PDFs in `/books/`).
- `humanjack/ai-engineering/docs/agents/` — deep dives on Pi, OpenClaw, Hermes, OpenCode, Deep Agents, Codex CLI.

---

## Document Map

- [Part I — Product Specification](#part-i--product-specification)
- [Part II — Technical Specification](#part-ii--technical-specification)
- [Part III — Phased Development Plan](#part-iii--phased-development-plan)
- [Appendix A — Glossary](#appendix-a--glossary)
- [Appendix B — Design lineage (what we steal from whom)](#appendix-b--design-lineage)
- [Appendix C — Open questions](#appendix-c--open-questions)

---

# Part I — Product Specification

## 1. Vision & North Star

**One-sentence vision.** A small, legible coding agent that you fully own — every loop, every tool, every byte of context — and that you can keep extending until it does for you what Claude Code, Codex, and OpenClaw do for their users.

**North-star metric for v1.** *"Can it close one real GitHub issue end-to-end on this repo without me touching the keyboard, and can I explain every decision it made by reading the trace?"*

If both are true, the agent is real. Everything else is feature accretion.

**Design philosophy.** Borrowing one sentence from each lineage agent and synthesizing:

> *"Ship the minimum loop you can read in one sitting, gate every action behind a hook, and earn every new capability by writing the smallest extension that adds it."*

Concretely that means:
- Core loop is small enough to fit on one screen (target: < 300 SLOC).
- Every tool call goes through one chokepoint (before/after hook).
- Memory, planning, RAG, sub-agents, MCP — all live as composable middleware/extensions, never inside the loop.

## 2. Target user

**Audience: one** — myself. This isn't a product for strangers; it's a tool I'll use daily and modify weekly. That means:

- I can tolerate config files instead of GUIs.
- I prefer explicit over magical (no autosuggested middleware imports, no "smart defaults" I can't trace).
- I'm willing to write Python to add a new capability — but only once. After that the capability should be reachable via config.

## 3. Use cases & user stories

### 3.1 v1 — Coding-first (the only thing v1 must do well)

| # | Story | Acceptance |
|---|---|---|
| US-1 | *"As me, I want to point the agent at a repo and say 'fix the bug in `parse_args`' so it reads the file, edits it, runs the tests, and tells me what it did."* | Agent loads `AGENTS.md`, edits one file, runs one shell command, posts a diff + test output. |
| US-2 | *"As me, I want to drop into an interactive coding session where I can chat with the agent and see every tool call it makes."* | REPL with streaming tokens; every tool call printed with name + truncated args + result. |
| US-3 | *"As me, I want the agent to fail safe — it should never delete files, push to remote, or run `curl ... \| sh` without my approval."* | Approval hook fires on dangerous ops; default-deny on a configurable list. |
| US-4 | *"As me, I want my session to be replayable — show me a trace I can re-run or diff against another model."* | Each session writes a JSONL transcript with all messages, tool calls, and results. |
| US-5 | *"As me, I want to swap models mid-task (Anthropic ↔ OpenAI) without restarting."* | `/model <provider:name>` in REPL swaps client; session continues. |

### 3.2 v2 — Generalist (Cowork-style)

| # | Story | Acceptance |
|---|---|---|
| US-6 | *"As me, I want a chat surface I can talk to about anything — research, life admin, code — and it'll route the right tools."* | Chat mode with the same loop but a generalist system prompt + web search + file tools. |
| US-7 | *"As me, I want to spawn a 'researcher' sub-agent that searches the web and writes a brief while my main session keeps coding."* | Sub-agent registry + parallel execution; results return to parent thread. |
| US-8 | *"As me, I want to install community Skills (markdown + scripts) and have the agent discover them on launch."* | `~/.genie/skills/*/SKILL.md` discovered, registered, surfaced in system prompt header. |
| US-9 | *"As me, I want scheduled tasks: 'every morning at 7am, summarize my open PRs.'"* | Cron-driven runner spawns ephemeral session with pre-filled prompt. |

### 3.3 Non-goals (v1 and v2)

- **Hosting it for other people.** Single-user, single-machine. No auth, no rate limits, no billing.
- **A pretty GUI.** TUI/CLI only; web UI is explicitly out of v1 and v2.
- **Training models or fine-tuning.** Provider APIs only.
- **Reimplementing LangChain.** No retrievers/agents/chains abstractions — just direct SDK + my own primitives.

## 4. Feature catalog

A flat list. Each line tags which phase ships it.

**Core loop**
- ReAct-style think → tool → observe → think loop *(Phase 1)*
- Streaming token output *(Phase 1)*
- Tool calls dispatched in parallel by default; `sequential` flag per tool *(Phase 1, pattern from Pi)*
- Mid-session model swap *(Phase 2)*
- Context compaction on overflow *(Phase 3)*

**Tools**
- `read_file`, `write_file`, `edit_file`, `bash` (the Pi four) *(Phase 1)*
- `grep`, `glob` *(Phase 1.5)*
- `web_search`, `web_fetch` *(Phase 4)*
- `todo_write` / `todo_read` *(Phase 3)*
- `task` (spawn sub-agent) *(Phase 5)*
- MCP client (outbound) *(Phase 5)*
- MCP server (inbound) — defer to v2 *(Phase 7)*

**Safety / sandboxing**
- Before-tool-call hook with deny-list *(Phase 2)*
- Bash AST parsing for per-command permissions *(Phase 4, OpenCode-style)*
- Subprocess isolation: working dir scoped, no parent-dir access by default *(Phase 2)*
- Optional Docker/podman backend *(Phase 6)*

**Memory & context**
- `AGENTS.md` project memory *(Phase 2)*
- Hierarchical memory (project / user / session) *(Phase 3)*
- Vector RAG over arbitrary directory *(Phase 4)*
- Session JSONL with parent-id (tree sessions) *(Phase 3)*
- Tool-result truncation with spill to disk *(Phase 3)*

**Planning & reflection**
- `update_plan` tool — model writes/updates a structured plan *(Phase 3)*
- Reflection step before final answer (optional) *(Phase 4)*

**Skills / extensions**
- Markdown-frontmatter skill discovery (agentskills.io-compatible) *(Phase 4)*
- Python extension entry points: `register_tool`, `register_hook` *(Phase 4)*

**Sub-agents**
- `task` tool spawns child session with isolated context *(Phase 5)*
- Shared iteration budget across the tree *(Phase 5)*

**Evaluation**
- Trace-based eval harness — replay session, compare outputs *(Phase 6)*
- LLM-as-judge scoring for code-edit tasks *(Phase 6)*
- Mini SWE-bench-style runner over a curated task set *(Phase 6)*

**Multi-surface (v2)**
- Slack / Discord / Gmail adapters via a Gateway pattern *(Phase 7, OpenClaw-style)*
- Scheduled tasks (cron-driven sessions) *(Phase 7)*
- Web-hook listener for "agent on demand" *(Phase 7)*

## 5. Personas & primary workflows

There's only one persona — me — but my modes vary:

- **Coding mode.** `genie code <repo>` — agent in repo working dir, AGENTS.md auto-loaded, tools = code-edit pack.
- **Chat mode.** `genie chat` — generalist; tools = read/write in `~/notes`, web_search, web_fetch.
- **Research mode.** `genie research "<topic>"` — spawns researcher sub-agent, returns a `.md` brief.
- **Scheduled mode.** `genied` daemon runs cron entries, posts results to a chosen sink.

## 6. Success metrics

| Phase | Metric | Target |
|---|---|---|
| 1 | Round-trip a 1-file edit + bash test | < 90s for simple fixes |
| 1 | Loop SLOC (excluding tools/providers) | < 300 |
| 2 | Approval false-positive rate (dangerous ops gated correctly) | 100% on a 50-case suite |
| 3 | Session resume after crash | 100% on JSONL replay |
| 4 | RAG recall @ 10 on personal-notes corpus | ≥ 0.7 vs. ground truth |
| 5 | Sub-agent task ends with parent context unchanged | 100% (sub-agent context never leaks into parent transcript) |
| 6 | mini-eval pass rate vs. Claude-Code baseline | ≥ 70% on chosen task subset |
| 7 | End-to-end: scheduled "morning brief" delivered to Slack | Daily for one week without intervention |

---

# Part II — Technical Specification

## 1. Top-level architecture

```
                    ┌─────────────────────────────────────┐
                    │            Entrypoints              │
                    │ genie code | genie chat | genied │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │         Session orchestrator        │
                    │  • loads config + AGENTS.md         │
                    │  • opens JSONL transcript           │
                    │  • constructs the loop              │
                    └─────────────────┬───────────────────┘
                                      │
        ┌─────────────────────────────▼─────────────────────────────┐
        │                        Agent Loop                          │
        │  ┌───────────┐   ┌──────────────┐   ┌──────────────────┐  │
        │  │ provider  │──▶│ tool registry│──▶│ sandbox / runner │  │
        │  │ (wrapper) │   │  + dispatch  │   │  (subprocess/    │  │
        │  └───────────┘   └──────┬───────┘   │   docker)        │  │
        │       ▲                 │           └──────────────────┘  │
        │       │           ┌─────▼─────┐                            │
        │       │           │   hooks   │  before/after tool, model │
        │       │           │ middleware│  start/finish, error      │
        │       │           └─────┬─────┘                            │
        │       │                 │                                  │
        │       └────context─pipeline────┘                           │
        │       (compaction, caching, memory injection)              │
        └─────────────────────────────┬─────────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │     Persistence + Observability      │
                    │  JSONL transcripts • cost ledger •   │
                    │  rotating logs • SQLite index (P3)   │
                    └──────────────────────────────────────┘
```

Eight cross-cutting subsystems, each addressable independently:

1. **Provider client wrapper** (own class)
2. **Tool registry & dispatcher**
3. **Sandbox layer**
4. **Hook / middleware system**
5. **Context pipeline** (memory + compaction + caching)
6. **Session / transcript store**
7. **Skill / extension loader**
8. **Eval harness**

Each subsystem has a single public class and is testable in isolation. The loop itself is just plumbing.

## 2. Repository layout

```
genie/
├── README.md
├── SPEC.md                            # this file
├── pyproject.toml
├── books/                             # the 10 PDFs
├── genie/
│   ├── __init__.py
│   ├── cli.py                         # `genie` entrypoint (argparse)
│   ├── config.py                      # pydantic settings: load .env, ~/.genie/config.toml
│   ├── loop.py                        # the ~300 SLOC ReAct loop
│   ├── providers/
│   │   ├── base.py                    # ProviderClient abstract
│   │   ├── anthropic_client.py
│   │   ├── openai_client.py
│   │   └── factory.py                 # provider_factory("anthropic:claude-sonnet-4-5")
│   ├── tools/
│   │   ├── base.py                    # @tool decorator + Tool dataclass
│   │   ├── registry.py
│   │   ├── builtins/
│   │   │   ├── read_file.py
│   │   │   ├── write_file.py
│   │   │   ├── edit_file.py
│   │   │   ├── bash.py
│   │   │   ├── grep.py
│   │   │   ├── glob.py
│   │   │   ├── web_search.py
│   │   │   ├── web_fetch.py
│   │   │   ├── todo.py
│   │   │   └── task.py                # sub-agent (Phase 5)
│   │   └── result.py                  # ToolResult dataclass + truncation
│   ├── sandbox/
│   │   ├── base.py
│   │   ├── local_subprocess.py
│   │   ├── docker_backend.py          # Phase 6
│   │   └── bash_ast.py                # tree-sitter AST guard (Phase 4)
│   ├── hooks/
│   │   ├── manager.py                 # before/after tool + lifecycle
│   │   └── builtin/
│   │       ├── approval.py
│   │       └── policy.py
│   ├── context/
│   │   ├── memory.py                  # AGENTS.md loader, hierarchical
│   │   ├── compaction.py
│   │   ├── caching.py                 # provider-specific cache markers
│   │   └── rag.py                     # FAISS/Chroma wrapper (Phase 4)
│   ├── session/
│   │   ├── transcript.py              # JSONL writer/reader
│   │   ├── store.py                   # tree sessions (parent_id)
│   │   └── replay.py
│   ├── skills/
│   │   ├── loader.py                  # SKILL.md frontmatter
│   │   └── registry.py
│   ├── eval/
│   │   ├── runner.py
│   │   ├── judges.py
│   │   └── tasks/                     # task definitions for mini-eval
│   ├── adapters/                      # v2 — Gateway pattern
│   │   ├── slack_adapter.py
│   │   └── ...
│   └── utils/
│       ├── tokens.py                  # token counting per provider
│       ├── streaming.py
│       └── logger.py
├── tests/
│   ├── test_loop.py
│   ├── test_tools/
│   ├── test_sandbox/
│   ├── test_hooks/
│   ├── golden/                        # frozen transcripts for regression
│   └── eval_suite/
├── prompts/
│   ├── coding_system.md               # v1 system prompt
│   ├── chat_system.md                 # v2 system prompt
│   └── research_system.md
└── examples/
    ├── extension_custom_tool/
    └── extension_custom_hook/
```

## 3. Provider client wrapper

### 3.1 Contract

```python
# genie/providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

@dataclass
class ChatMessage:
    role: str                      # "system" | "user" | "assistant" | "tool"
    content: str | list[dict]      # text or content blocks
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

@dataclass
class ChatChunk:
    delta_text: str | None = None
    tool_call_delta: dict | None = None
    finish_reason: str | None = None
    usage: dict | None = None      # input_tokens, output_tokens, cache_read, cache_write

class ProviderClient(ABC):
    name: str                      # "anthropic" | "openai"
    model: str

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system: str | None = None,
        cache_breakpoints: list[int] | None = None,   # indexes into messages
    ) -> AsyncIterator[ChatChunk]: ...

    @abstractmethod
    def count_tokens(self, messages: list[ChatMessage]) -> int: ...
```

The contract intentionally **does not** expose provider-native schemas. Tool definitions are passed as JSON Schema dicts; the wrapper translates per provider.

### 3.2 Anthropic adapter

- Uses `anthropic.AsyncAnthropic`.
- Tool defs passed through as `tools=[{name, description, input_schema}]`.
- `system` is the top-level system param.
- Cache breakpoints become `cache_control: {"type": "ephemeral"}` on the message at that index.
- Streaming via `client.messages.stream(...)`; yields `ChatChunk`s mapped from `RawContentBlockDelta`, `ContentBlockStart`, `MessageStop`, etc.

### 3.3 OpenAI adapter

- Uses `openai.AsyncOpenAI` Responses API (preferred — server-side state) **or** Chat Completions; configurable per provider profile.
- Tool defs wrapped as `{"type":"function","function":{name,description,parameters}}` for Chat, or as Responses API tool shape.
- Cache markers are not user-controlled (server-managed); the breakpoint param becomes a no-op.

### 3.4 Why our own wrapper instead of LiteLLM/etc.

- We learn the exact translation surface (this is the entire reason to roll one).
- Tool-call delta normalization is provider-specific and we want to control how we surface it to the loop.
- Cost ledger and token counting want to live next to the call site — easier in our own class than monkeypatching.

## 4. Agent loop

```python
# genie/loop.py — target ~300 SLOC
async def run_turn(
    session: Session,
    provider: ProviderClient,
    tool_registry: ToolRegistry,
    hooks: HookManager,
) -> TurnResult:
    """One model call + zero-or-more tool calls until the model stops asking."""

    while True:
        # 1. context pipeline: assemble messages from session + memory
        messages = session.materialize_messages()
        tools = tool_registry.specs_for(provider.name)

        # 2. hook: about to call the model
        await hooks.run("before_model_call", session=session, messages=messages)

        # 3. stream model response, collecting tool calls + assistant text
        assistant_msg, tool_calls = await stream_one_assistant_turn(
            provider, messages, tools, session
        )
        session.append(assistant_msg)
        await hooks.run("after_model_call", session=session, message=assistant_msg)

        if not tool_calls:
            return TurnResult(stopped=True, last=assistant_msg)

        # 4. dispatch tool calls (parallel by default; serial if any tool requests it)
        results = await dispatch_tool_calls(
            tool_calls, tool_registry, hooks, session
        )
        for tc, r in zip(tool_calls, results):
            session.append(ToolMessage(tool_call_id=tc.id, content=r.content))
        # loop back: model gets to see results
```

Key decisions:

- **Async-first.** Python's `asyncio` because tool calls and streaming both benefit.
- **`dispatch_tool_calls` is one chokepoint.** Every tool call passes through `before_tool_call` → `after_tool_call` hooks. (Pattern from Pi/OpenClaw.)
- **No iteration limit baked in.** The `IterationBudget` is a hook (Hermes-style); the loop trusts the budget hook to raise.
- **No retry logic in the loop.** Retry / circuit-breaker is a hook around `before_model_call`.

## 5. Tool registry & dispatch

### 5.1 Tool definition

```python
# genie/tools/base.py
from typing import Callable, Awaitable
from pydantic import BaseModel

class Tool(BaseModel):
    name: str
    description: str
    input_schema: dict             # JSON Schema (the lowest common denominator)
    handler: Callable[..., Awaitable["ToolResult"]]
    sequential: bool = False       # if True, batch becomes serial (Pi pattern)
    dangerous: bool = False        # short-circuits to approval hook
    tags: list[str] = []           # for filtering: ["code", "fs", "net"]

def tool(*, name=None, description=None, sequential=False, dangerous=False, tags=()):
    """Decorator. Reads pydantic args model to derive input_schema."""
    def decorator(func): ...
    return decorator

# Example:
@tool(name="read_file", tags=["fs", "code"])
async def read_file(path: str) -> ToolResult: ...
```

### 5.2 Registry

```python
class ToolRegistry:
    def __init__(self): self._tools: dict[str, Tool] = {}
    def register(self, tool: Tool): ...
    def specs_for(self, provider_name: str) -> list[dict]:
        """Translate to provider-native shape."""
    async def call(self, name: str, args: dict, *, hooks, session) -> ToolResult: ...
```

Provider translation rules:
- **Anthropic:** `{name, description, input_schema}` — passed through.
- **OpenAI:** wrap as `{"type":"function","function":{name,description,parameters}}`.

### 5.3 Dispatch

```python
async def dispatch_tool_calls(calls, registry, hooks, session) -> list[ToolResult]:
    # Mirror Pi's rule: if any tool in this batch is sequential, run them serially.
    any_seq = any(registry.get(c.name).sequential for c in calls)
    if any_seq:
        return [await _call_one(c, registry, hooks, session) for c in calls]
    return await asyncio.gather(*[_call_one(c, registry, hooks, session) for c in calls])

async def _call_one(call, registry, hooks, session) -> ToolResult:
    await hooks.run("before_tool_call", call=call, session=session)
    try:
        result = await registry.call(call.name, call.args, hooks=hooks, session=session)
    except Exception as e:
        result = ToolResult.error(str(e))
    await hooks.run("after_tool_call", call=call, result=result, session=session)
    return result
```

### 5.4 Tool result post-processing

Three-layer defense (Hermes-style):

1. **Per-tool truncation.** Each tool declares `max_result_size_chars`; bash defaults to 8 KB.
2. **Spill to disk.** Anything over the cap is written to `~/.genie/spill/{session_id}/{tool_call_id}.txt` and the model sees head + tail + `[truncated; full output at ...]`.
3. **Per-turn aggregate budget.** Hook tracks total bytes per turn; raises before next dispatch if over.

`read_file` on a spill path always returns the full content — same mechanism Deep Agents uses (the spill path *is* a real file the model can read).

## 6. Sandbox layer

### 6.1 v1 — local subprocess

`LocalSubprocessBackend.exec(cmd, cwd, env, timeout)`:
- `cwd` is **always** the session's working directory or a subpath.
- `env` is a curated allowlist (PATH, HOME, LANG; no AWS_* / GH_TOKEN by default).
- `timeout` defaults to 30s, configurable per call.
- Output captured with line/byte caps; stderr merged.

### 6.2 v2 — bash AST guard (OpenCode-inspired)

Parse the command with `tree-sitter-bash`; for each command-node:
- Look up the command name in `~/.genie/bash_policy.toml` (allow / ask / deny).
- Honor an **arity dictionary** so `git checkout main && rm -rf /` is two commands, not a substring.
- Multi-line pipes / subshells walked recursively.

### 6.3 v3 — Docker backend (optional)

`DockerSandboxBackend.exec(...)`:
- Image: a minimal `python:3.12-slim` + `git` + repo bind-mounted.
- No network unless `tool.tags` includes `"net"`.
- Filesystem writes constrained to a tmpfs overlay.

### 6.4 What we explicitly defer

OS-syscall isolation (seccomp/landlock/seatbelt) — Codex's territory. We tip our hat and use Docker instead. Reasoning: building seatbelt profiles per macOS version is a project unto itself, and Docker gets us 80% of the isolation for a fraction of the engineering.

## 7. Hook / middleware system

### 7.1 Events

| Event | Payload | When |
|---|---|---|
| `session_start` | `Session` | before first turn |
| `session_end` | `Session, reason` | after loop exits |
| `before_model_call` | `messages, model, tools` | each turn |
| `after_model_call` | `assistant_message, usage` | each turn |
| `before_tool_call` | `call, session` | every tool call (incl. parallel) |
| `after_tool_call` | `call, result, session` | every tool call |
| `model_error` | `exception, retry_count` | provider raises |
| `tool_error` | `call, exception` | tool raises |

This is intentionally compatible with Anthropic's Claude Code hook schema (8 events) — same lift Codex made. Cheap interoperability.

### 7.2 Hook contract

```python
class Hook(Protocol):
    name: str
    events: list[str]
    async def __call__(self, event: str, **payload) -> HookOutcome: ...

class HookOutcome:
    block: bool = False            # short-circuit: don't actually call the tool/model
    block_reason: str | None = None
    mutated_payload: dict | None = None
```

Hooks fire in registration order; first `block=True` stops the cascade and propagates a `BlockedError` back to the loop, which formats it as a tool-result-equivalent for the model to read.

### 7.3 Built-in hooks (Phase 2)

- **`approval`** — `dangerous=True` tools prompt the user (TTY) or auto-deny in non-interactive mode.
- **`iteration_budget`** — caps total tool calls per session.
- **`cost_ledger`** — accumulates token usage into `~/.genie/ledger.jsonl`.
- **`policy`** — loads `~/.genie/policy.toml` rules: `bash` commands matching `rm -rf /` always blocked, etc.

## 8. Context pipeline

### 8.1 Layers (assembled top-down for the model call)

```
[system prompt]              # frozen, never mutated mid-session (Hermes pattern)
[memory header]              # AGENTS.md (project) + user-level memory + active skills
[skills disclosed]           # full bodies of skills loaded this turn
[session messages]           # the unbounded conversation thread, possibly compacted
[cache breakpoint]           # provider-specific (Anthropic markers / OpenAI no-op)
```

### 8.2 Memory model (Phase 3)

Three tiers:

- **Project memory**: `<repo>/AGENTS.md` — auto-loaded, read-only during session (writes via tool only).
- **User memory**: `~/.genie/MEMORY.md` — global facts (preferences, recurring projects).
- **Session memory**: in-transcript scratchpad; cleared at session end.

The agent has a `remember` tool that appends a tagged line to `MEMORY.md`. Mid-session memory writes go to **files**, not to the system prompt — so the cache stays warm (Hermes discipline).

### 8.3 Compaction

Trigger: assembled message tokens > `compaction_threshold` (default: 80% of model's context window).

Strategy:
1. **Tool-result eviction first.** Replace large tool messages with their spill-path reference (Deep Agents pattern).
2. **Old-turn summarization.** If still over budget, ask a cheap model (e.g., `claude-haiku`) to summarize messages 0..N-K into one assistant message; preserve last K turns verbatim.
3. **Cache-cold rebuild.** After step 2, the system prompt + first compacted summary become the new cache prefix.

### 8.4 RAG (Phase 4)

A *separate* tool the model can invoke — **not** a context-stuffing pre-step.

- Storage: SQLite + FAISS (or `chromadb`) — local, no external service.
- Indexing: `genie index <dir>` walks markdown / code, chunks by markdown header or ~500 tokens, embeds with `text-embedding-3-small` (OpenAI) or `voyage-3` (Anthropic-recommended).
- Retrieval: `rag_search(query, k=5, collection)` returns top-k with source paths.
- The model chooses to call it. We don't auto-stuff context.

## 9. Session / transcript

### 9.1 On-disk format

`~/.genie/sessions/<id>/`
- `meta.json` — id, parent_id, started_at, working_dir, model, branch
- `transcript.jsonl` — one line per message (role, content, tool_calls, ts, usage)
- `spill/` — large tool outputs

### 9.2 Tree sessions (Pi pattern)

- `parent_id` on every session.
- `/fork` from any point creates a new child with the parent's transcript up to the cursor.
- Useful for "what if I asked it differently from step 7" without re-running.

### 9.3 Replay

`genie replay <session_id> --upto <message_idx>` rebuilds the message list and re-runs from that point with the current code (regression check after refactors).

## 10. Skill / extension system (Phase 4)

### 10.1 Skill discovery

`~/.genie/skills/<name>/SKILL.md`:

```markdown
---
name: review-pr
description: Review a GitHub PR — checkout, diff, lint, test, comment
triggers: [pr-review, review-this-pr]
tools_required: [bash, read_file]
---

# Body

Read instructions...
```

On launch, the loader scans the skills dir, indexes name/description, and adds a one-line each into a **skills header** at the top of the messages. Full SKILL.md body is **not** in context until the model calls `load_skill(name)` — progressive disclosure (Claude Code / Pi / Codex convention).

### 10.2 Python extensions

```python
# ~/.genie/extensions/my_ext/extension.py
from genie.tools import tool
from genie.hooks import on

@tool(name="jira_search", tags=["net"])
async def jira_search(query: str) -> ToolResult: ...

@on("before_tool_call")
async def block_friday_deploys(call, session, **_): ...
```

Loader looks at `~/.genie/extensions/*/extension.py`, imports, lets the decorators self-register.

## 11. Sub-agents (Phase 5)

The `task` tool spawns a child session:

```python
@tool(name="task", description="Spawn a sub-agent with isolated context.")
async def task(*, prompt: str, persona: str = "default",
               tools_allowlist: list[str] | None = None,
               max_iterations: int = 10) -> ToolResult:
    child = await spawn_child_session(parent=current_session,
                                      prompt=prompt,
                                      persona=persona,
                                      tools=tools_allowlist,
                                      budget_share=...)
    transcript_summary = await run_to_completion(child)
    return ToolResult.text(transcript_summary)
```

Properties:
- **Isolated context.** The child's transcript never enters the parent's message list — only its final summary does.
- **Shared budget.** A single `IterationBudget` instance is threaded through the session tree.
- **Permissions.** Tools allowlist defaults to parent's minus `task` (no recursive spawning beyond a depth cap).

## 12. Evaluation harness (Phase 6)

### 12.1 Task definition

```python
# tests/eval_suite/fix_bug_in_parse.py
EVAL = EvalTask(
    name="fix_bug_in_parse",
    setup=lambda tmp: copy_fixture("buggy_parser", tmp),
    prompt="The function `parse_args` mis-handles negative numbers. Fix it and confirm tests pass.",
    success=lambda tmp: run_pytest(tmp) and grep("def parse_args", tmp),
)
```

### 12.2 Runner

`genie eval --suite=v1` runs each task:
1. Spin up a temp working dir.
2. Run the agent to completion.
3. Evaluate success function (returns bool + diagnostics).
4. Record: tokens used, wall time, tool calls, success.

### 12.3 LLM-as-judge

For tasks where success can't be a simple function:

```python
JUDGE_PROMPT = """
Original task: {task}
Agent's transcript: {transcript}
Expected behaviors: {rubric}
Score 1-5 and explain. Return JSON.
"""
```

Run on Sonnet against the agent's transcript; aggregate.

### 12.4 Mini-bench

Curate ~20 tasks from real GitHub issues we've solved historically. Re-run weekly. Track pass rate + cost over time.

## 13. Configuration

`~/.genie/config.toml`:

```toml
[provider]
default = "anthropic:claude-sonnet-4-6"

[provider.anthropic]
api_key_env = "ANTHROPIC_API_KEY"

[provider.openai]
api_key_env = "OPENAI_API_KEY"
api = "responses"                       # or "chat_completions"

[loop]
max_iterations = 50
compaction_threshold = 0.8

[tools.bash]
timeout_seconds = 30
max_output_bytes = 8192

[sandbox]
backend = "local_subprocess"            # or "docker"
working_dir_only = true

[approval]
mode = "ask"                            # ask | auto | deny
dangerous_patterns = ["rm -rf /", "git push", "curl .* | sh"]

[memory]
project_file = "AGENTS.md"
user_file = "~/.genie/MEMORY.md"

[skills]
dirs = ["~/.genie/skills"]
```

## 14. Cross-cutting concerns

### 14.1 Observability

- Every event flows through `structlog` with a per-session correlation id.
- Cost ledger: append-only JSONL — input/output/cache tokens, model, $ estimate.
- `genie status` reads ledger + recent sessions; prints last 7 days summary.

### 14.2 Errors

- All provider/network errors retried with exponential backoff (jitter), 3 attempts.
- Tool errors are **never** retried by the loop; they're returned to the model as tool-result errors so the model can react. (Pattern from all six reference agents.)
- Unhandled exceptions in the loop write a crash report to `~/.genie/crashes/{ts}.json` and exit with non-zero.

### 14.3 Token & cost accounting

`genie/utils/tokens.py` exposes:
- `count_messages(messages, model)` — provider-specific tokenizer (`anthropic.tokenize` / `tiktoken`).
- `estimate_cost(usage, model)` — table-driven `$/Mtok` values, kept current via `pricing.toml`.

### 14.4 Testing strategy

- **Unit tests** for every tool (mock fs / subprocess).
- **Loop tests** with a `FakeProvider` that emits scripted streams — proves the loop respects tool calls, hooks, errors, budgets.
- **Golden transcripts**: 5–10 frozen sessions in `tests/golden/` — replay must match (modulo timestamps/ids).
- **Integration tests** that hit Anthropic/OpenAI cheaply (Haiku/4o-mini) once per CI run.

## 15. Performance & cost targets

| Metric | Target |
|---|---|
| Cold start (CLI → first model token) | < 1s |
| Loop overhead per turn (excl. model latency) | < 50ms |
| Memory footprint (idle session) | < 200 MB |
| Tokens per simple bug-fix task | < 30k input, < 5k output |
| $ per task (Sonnet baseline) | < $0.20 |

---

# Part II.5 — Development workflow (process spec)

This section is process, not architecture, but it's load-bearing for how we ship.

## W.1 Component-as-PR rule

Every subsystem named in §1 ships as **one or more independent pull requests**. A PR contains:

- The component's abstract base class (if not already on `main`).
- At least one concrete implementation.
- Tests achieving **≥ 70% line coverage** for the component's package.
- Doc string on every public class/function; no comment narration of WHAT the code does.
- A short PR description: what changed, what's tested, what's deferred.

A PR must NOT bundle multiple subsystems. If a feature spans subsystems, it ships as a stack of stacked PRs, merged in order.

## W.2 PR lifecycle

```
branch from main → implement → unit tests pass locally → push → open PR
   → /code-review (skill) → resolve findings → re-review if needed
   → coverage ≥ 70% verified by CI → squash-merge to main
```

Branch naming: `phase{N}/<subsystem>-<short-slug>` (e.g., `phase1/tools-bash`).

## W.3 Test requirements per component

- **Unit tests** for every public function. Mock external deps (provider, fs, subprocess) using the abstract base.
- **Replaceability test.** Each subsystem ships with a stub/fake implementation, and tests prove the loop works against both `Real` and `Fake`. This is how we know the abstraction holds.
- **Coverage gate.** `pytest --cov=genie.<subsystem> --cov-fail-under=70` runs in CI per PR. Below 70%, CI red.
- **No internet in unit tests.** Provider integration tests are gated behind `RUN_LIVE_API=1` and run separately.

## W.4 Review gate

The `/code-review` skill runs on each PR before merge. Reviewer scope (in priority order):

1. **Pluggability** — does this honor the abstract base? Could a second impl drop in without touching callers?
2. **Security** — does any disk/network/shell call bypass the hook chain?
3. **Minimalism** — any code that doesn't serve the current phase's exit criteria?
4. **Tests** — coverage ≥ 70%, plus a fake/stub impl, plus a "swap impl" test.
5. **Correctness** — bugs, error handling at boundaries.

Any P0/P1 finding blocks merge. P2/P3 are tracked as follow-up issues.

## W.5 Parallelism rules

Components with **no shared interface change** can be developed in parallel branches off `main`. Concretely (Phase 1):

- `providers/` (Phase 0) blocks everything → must land first.
- After provider lands: `tools/base.py` + `tools/registry.py` must land before any individual tool.
- After registry lands: `read_file`, `write_file`, `edit_file`, `bash`, `session/transcript.py`, `sandbox/local_subprocess.py` can all branch in parallel.
- `loop.py` lands last, after all of its dependencies are merged.

## W.6 Integration / E2E gate

After every component in a phase has merged to `main`, an **integration test PR** lands that:

1. Runs realistic user scenarios end-to-end (not unit-mocked).
2. Uses `FakeProvider` with pre-recorded streams for determinism, then optionally a live `Anthropic` run gated by `RUN_LIVE_API=1`.
3. Covers each User Story for that phase as a scripted scenario.

Phase is only "done" when integration tests pass. Phase tag (`v0.1.0-phase1`) is cut from the merge commit.

---

# Part III — Phased Development Plan

Each phase has a **goal**, **deliverables**, **exit criteria**, **what's deferred**, and **rough effort**. Phases are sized to be shippable individually — each ends with a usable agent.

## Phase 0 — Skeleton (effort: ~1 week, 1 person)

### Goal
Repo scaffolded, dev loop running, can call both providers.

### Deliverables
- `pyproject.toml` with deps: `anthropic`, `openai`, `pydantic`, `httpx`, `rich`, `typer` (or argparse), `pytest`, `tree-sitter-bash`, `structlog`.
- `genie/providers/` — `base.py`, `anthropic_client.py`, `openai_client.py`, `factory.py`.
- `genie/cli.py` — single `genie chat-once "<prompt>"` command that streams a reply (no tools yet).
- Config loader (`config.py`) with env + TOML.
- CI: lint (ruff), typecheck (pyright), tests (pytest), runs on every push.

### Exit criteria
- `genie chat-once "Hello"` streams a response from both `anthropic:sonnet` and `openai:gpt-4o-mini` with a `--model` flag.
- `pytest` passes on a smoke test of each provider.
- `make lint typecheck test` is green.

### Deferred
Tools, loop, anything fancy. This phase only proves provider abstraction works.

### Risks
- API key handling — keep them out of git, surface clearly when missing.
- Provider streaming idiosyncrasies — write the `FakeProvider` *now* so subsequent phases can test without API calls.

---

## Phase 1 — Minimal ReAct loop + 4 tools (effort: ~2 weeks)

> Maps to book chapters 1–4: agents, LLMs, tool use, ReAct.

### Goal
Round-trip a coding task end-to-end: the agent reads files, edits them, runs bash, returns.

### Deliverables
- `genie/tools/base.py` — `@tool` decorator + `Tool` dataclass + `ToolResult`.
- `genie/tools/registry.py` — register + spec translation per provider.
- The Pi four: `read_file`, `write_file`, `edit_file`, `bash`.
- `genie/loop.py` — the ReAct loop (target ~300 SLOC including imports).
- `genie/session/transcript.py` — JSONL writer.
- `genie code [path]` CLI command opens REPL in given dir.
- Streaming display: assistant text streams, tool calls render as collapsible blocks (Rich).
- System prompt: `prompts/coding_system.md` (concise — borrows from Claude Code / Codex prompts, adapted).

### Exit criteria
- US-1 passes: "fix the bug in `parse_args`" on a seeded buggy repo, agent edits + runs tests + reports.
- US-2 passes: REPL is interactive, streams, shows every tool call.
- Loop SLOC < 300 (real measurement, not vibes).
- Golden transcript test runs to completion deterministically with `FakeProvider`.

### Deferred
- Approvals (next phase).
- Compaction (just hard-cap iterations for now).
- AGENTS.md (next phase).
- Sub-agents, RAG, skills.

### Risks
- **Tool-call streaming semantics differ by provider.** Build the test-stream fixtures from real API responses (record once, replay forever).
- **Edit tool granularity.** Decide upfront: `edit_file` is anchor-based (find/replace within file) — not patch-based. Patch-based is harder to get right with streaming.

### Validation
- Run on 3 real bugs in the genie repo itself (dogfooding from week one).
- Compare a simple session to a Claude-Code session on the same task — does ours arrive at the same edit?

---

## Phase 2 — Safety: hooks, approvals, sandbox (effort: ~1.5 weeks)

### Goal
Stop the agent from rm-rf'ing things. Make every action go through a chokepoint.

### Deliverables
- `genie/hooks/manager.py` — events listed in §7.1, registration API.
- Built-in hooks: `approval`, `iteration_budget`, `cost_ledger`, `policy`.
- `genie/sandbox/local_subprocess.py` — bash + write_file go through it; cwd-scoped.
- Policy file format: `~/.genie/policy.toml` with deny / ask / allow patterns.
- `--unsafe` flag to bypass for trusted batch runs (logs a warning).
- Cost ledger: every model call appends to `ledger.jsonl`.

### Exit criteria
- US-3: 50-case approval test suite — dangerous ops gated, safe ops pass. 100% correct.
- `genie status` prints last 7 days of token / cost usage.
- Session blocked by `iteration_budget` ends cleanly with budget-exhausted message.

### Deferred
- Bash AST parsing (Phase 4 — needs `tree-sitter-bash` plumbing).
- Docker sandbox (Phase 6).

### Risks
- **Approval UX in REPL.** Block on `input()` is fine for v1 but interrupts streaming — handle it.
- **Hook ordering.** Document explicit registration order; provide `--debug-hooks` flag to print the chain.

---

## Phase 3 — Memory, sessions, planning, compaction (effort: ~2 weeks)

> Maps to book chapters 6 (memory) + 7 (planning).

### Goal
Sessions are persistent, branchable, and don't OOM the context window. Agent can write/read a plan.

### Deliverables
- `genie/session/store.py` — tree sessions with `parent_id`.
- `/fork`, `/branches`, `/resume <id>` REPL commands.
- `genie/context/memory.py` — load `AGENTS.md` (project) + `~/.genie/MEMORY.md` (user); inject into messages.
- `remember` tool — appends to user memory (no system-prompt mutation — files only, Hermes-style).
- `todo_write`, `todo_read` tools.
- `genie/context/compaction.py` — eviction-first compaction; summarization fallback.
- `genie replay <id>` command.

### Exit criteria
- US-4: replay a saved session, agent's deterministic responses (with `FakeProvider`) match byte-for-byte.
- Session that hits compaction threshold compacts and keeps going; last K turns preserved.
- `AGENTS.md` in a repo is picked up by `genie code` automatically.

### Deferred
- RAG (next phase).
- Cross-session shared memory via vector store (Phase 4).

### Risks
- **Compaction loses important context.** Implement an "anchor" mechanism — explicitly marked messages survive compaction. Lift from OpenCode's two-tier strategy if naive eviction misses.
- **Plan tool feeling like UI clutter.** Render todos as a sticky panel in Rich, separate from the chat scroll.

---

## Phase 4 — Skills, RAG, web tools, bash AST guard (effort: ~2.5 weeks)

> Maps to book chapter 5 (RAG).

### Goal
Agent can discover community skills, search a knowledge base, hit the web, and refuses dangerous bash via real parsing.

### Deliverables
- `genie/skills/loader.py` + frontmatter parser.
- Progressive disclosure: SKILL.md headers in system, full body via `load_skill(name)` tool.
- `genie/context/rag.py` — FAISS-backed; `genie index <dir>` builds; `rag_search` tool exposes.
- `web_search` (Brave Search / Tavily / SerpAPI — choose one, configurable) and `web_fetch` tools.
- `genie/sandbox/bash_ast.py` — `tree-sitter-bash` parse → walk → check each command against policy.
- Two example skills shipped: `review-pr/` and `summarize-pdf/`.
- Two example extensions shipped: a `custom_tool/` and a `custom_hook/`.

### Exit criteria
- `genie index ~/notes` indexes a known directory; `rag_search` recall@10 ≥ 0.7 on a manually-built ground-truth set of 20 queries.
- Bash AST guard blocks `git checkout main && rm -rf /` while letting `git checkout main` through.
- Drop a third skill into `~/.genie/skills/` — agent discovers it on next launch.

### Deferred
- MCP client (next phase).
- Sub-agents (next phase).
- Embedding model swap UI (config-only for now).

### Risks
- **`tree-sitter-bash` grammar coverage.** Some bash gets misparsed; fall back to deny-by-default for unparseable input.
- **Embeddings cost.** Local model fallback (e.g., `bge-small-en` via `sentence-transformers`) for cheap reindexing.

---

## Phase 5 — MCP client + sub-agents (effort: ~2 weeks)

> Maps to book chapter 9 (multi-agent orchestration).

### Goal
Connect to the existing MCP ecosystem and spawn isolated child sessions for parallel work.

### Deliverables
- MCP client transport: stdio + SSE; configured via `mcp_servers.toml`.
- `genie mcp list` / `genie mcp test <server>` debugging commands.
- Tools auto-namespaced as `<server>_<tool>`; schemas translated.
- `task` tool: spawn child session, isolated context, shared budget.
- Sub-agent depth cap (default 2).
- Optional schema masking on MCP tools (Codex pattern): hide certain fields from the model via config.

### Exit criteria
- Connect to one real MCP server (e.g., GitHub MCP) and call a tool through the agent.
- US-7: "spawn a researcher" — sub-agent runs, writes a brief; parent context size unchanged.
- Recursive sub-agent attempt at depth > cap is rejected cleanly.

### Deferred
- MCP server (inbound) — Phase 7.
- Async / mailbox-style sub-agents (Codex first-class) — Phase 7 if needed.

### Risks
- **MCP server bugs.** Servers are uneven in quality; add aggressive logging on each call.
- **Context bleed.** Test rigorously that the sub-agent's tool calls never enter the parent transcript except as the final summary.

---

## Phase 6 — Evaluation harness (effort: ~1.5 weeks)

> Maps to book chapter 10 (evaluating agents).

### Goal
Know whether changes to the agent make it better or worse on real work.

### Deliverables
- `genie/eval/runner.py` — task loader, parallel execution, result aggregation.
- `genie/eval/judges.py` — LLM-as-judge wrapper with frozen judge prompt.
- 20 curated tasks in `tests/eval_suite/` derived from real issues.
- `genie eval --suite=v1` runs them all; prints success rate, cost, p50/p95 wall time.
- `genie eval --diff <run_a> <run_b>` compares two runs.
- Optional: Docker sandbox backend (`genie/sandbox/docker_backend.py`) for hermetic eval runs.

### Exit criteria
- ≥ 70% pass rate on the v1 suite vs. a Claude-Code baseline run on the same tasks.
- Regressions detectable: introduce a known-bad change, rerun, see the suite drop.
- Run takes < 30 minutes wall-clock on a workstation.

### Deferred
- Public leaderboard / shareable reports — not needed for a single-user tool.

### Risks
- **Task curation is the work.** Spend half the phase building real tasks, half on the runner.
- **Judge prompt drift.** Freeze it (version + hash); changes to the judge prompt re-baseline the suite.

---

## Phase 7 — Multi-surface (v2 begin) (effort: open-ended)

> The "Cowork-like" generalization. Optional and incremental.

### Goal
The agent isn't just a CLI; it's reachable from Slack, Gmail, scheduled crons, and webhooks. (OpenClaw's productization arc.)

### Deliverables
- **Gateway pattern**: an `Adapter` base class (`receive`, `send`, `auth`); concrete adapters for Slack, Gmail, Discord (one at a time).
- `genied` daemon runs adapters + cron + webhook listener.
- Channel-aware policy: each adapter has its own `policy.toml` slice (Slack DMs get fewer tools than the CLI).
- Cron entries spawn ephemeral sessions with templated prompts.
- MCP server (inbound) — expose genie's tools to other agents.

### Exit criteria
- US-9: morning brief lands in a chosen Slack channel daily for a week without intervention.
- A Slack mention triggers a session, agent replies in-thread.
- Sub-agent spawned from a Slack thread still streams progress back to the thread.

### Deferred
- Web UI (still explicitly out of scope).
- Multi-user.

### Risks
- **Auth surface area.** Slack OAuth, Gmail PubSub, etc. — each is a project. Cap one per sub-phase.
- **Policy explosion.** A 22-channel surface (OpenClaw's count) is not the goal here; pick the two you actually use.

---

## Phase ordering rationale

The order is **safety after capability, capability before optimization, eval after enough capability to evaluate**:

1. Get the loop working with tools (Phase 1) — without this nothing else matters.
2. Stop it from breaking things (Phase 2) — before you trust it with real repos.
3. Make it persistent and plannable (Phase 3) — before sessions become unmanageable.
4. Make it extensible (Phase 4) — before you write the 10th tool inline.
5. Make it composable (Phase 5) — sub-agents and MCP are force multipliers.
6. Measure it (Phase 6) — only meaningful with real capability to grade.
7. Productize (Phase 7) — only after the engine is stable.

If a phase reveals the prior phase was wrong, **rework, don't pile on**. A 300-line loop is cheap to rewrite; a 30,000-line one isn't.

---

# Appendix A — Glossary

- **AGENTS.md** — Markdown file in repo root that an agent auto-loads as project memory. Convention started by Claude Code; standard across Pi/OpenCode/Codex/Deep Agents.
- **MCP** — Model Context Protocol. Anthropic's standard for tool composition; servers expose tools, clients (agents) consume them.
- **ReAct loop** — Reason + Act loop: model thinks, calls a tool, sees the result, repeats.
- **Sub-agent** — A child session spawned by a tool call, with isolated context and shared budget.
- **Skill** — A markdown file (with YAML frontmatter) describing a reusable workflow the agent can invoke.
- **Hook** — A callback that fires on lifecycle events (before/after tool, before/after model, etc.).
- **Compaction** — Reducing the message list size (via eviction or summarization) when nearing the context window.
- **Tree session** — A session whose history forms a tree (every message has a `parent_id`), enabling forks.
- **Spill** — Writing oversized tool output to disk so the model sees only head + tail + a path.

# Appendix B — Design lineage

| Concept | Source | Why we steal it |
|---|---|---|
| 4-tool minimal core, extension hooks | Pi | Forces the loop to stay small. |
| Tree sessions (`parent_id` JSONL) | Pi | Cheap branching for "what if I asked differently." |
| Wrap every tool with one hook | OpenClaw | Codifies policy-as-middleware without a framework. |
| Hooks schema matching Claude Code (8 events) | Codex | Free interoperability with the Anthropic ecosystem. |
| Frozen system prompt + file-based mid-session writes | Hermes | Keeps prompt cache warm. |
| Three-layer tool-result truncation | Hermes | Avoid context blowups with disciplined defense in depth. |
| Spill-to-disk via the same backend the tool uses | Deep Agents | Elegant: `read_file` retrieves the spill, no special path. |
| Tree-sitter bash AST permissions | OpenCode | Regex bash policy is unsafe; AST + arity is the right level. |
| Progressive-disclosure skills (SKILL.md) | Claude Code → Pi/Codex/Hermes/DA | Standard format; future skills work for free. |
| Provider portability via own protocol | Codex | Loop doesn't know which provider it's talking to. |
| Sub-agents with shared `IterationBudget` | Hermes | Stops a single session tree from runaway-spending. |
| Tool schema masking | Codex | Defense in depth — hide dangerous params from the model. |
| LSP-aware tool feedback | OpenCode | Deferred to a possible Phase 4.5; high payoff for typed langs. |

# Appendix C — Open questions

1. **Embedding model for RAG.** OpenAI `text-embedding-3-small`, Voyage, or a local model via `sentence-transformers`? Decision needed before Phase 4. *Likely:* OpenAI for ergonomics, with `bge-small-en` as configurable fallback.
2. **Web-search provider.** Brave (cheap, decent), Tavily (built for agents, more expensive), or SerpAPI. *Likely:* Tavily for v1, swap later.
3. **REPL framework.** `prompt_toolkit` (best-in-class, more setup) vs. `rich` + plain stdin (faster to ship). *Likely:* `rich` + stdin in Phase 1, `prompt_toolkit` if it bites.
4. **Async style.** Pure async or sync-with-threadpool? *Decision:* async — provider SDKs both support it natively and tool-call parallelism wants it.
5. **Plan vs. todo distinction.** Are `update_plan` and `todo_write` the same tool? *Likely:* yes, single tool, `mode={"plan","todo"}` arg.
6. **Sub-agent transcripts.** Persist on disk or in-memory only? *Likely:* on disk, same JSONL format, linked by `parent_session_id`.
7. **Dogfooding strategy.** Use genie to develop genie starting when? *Decision:* end of Phase 1 — even imperfect dogfooding accelerates everything after.
8. **First MCP server to integrate.** GitHub MCP, filesystem MCP, or write our own first? *Likely:* GitHub MCP — immediately useful for the dogfooding loop.

---

*End of spec. Next action: scaffold Phase 0.*
