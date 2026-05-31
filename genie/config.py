"""Typed configuration loader (SPEC §13).

Precedence is ``defaults < TOML file < environment overrides``. The TOML source
is the single pluggable seam: file parsing lives in :func:`_read_toml`, so the
on-disk format could be swapped without touching the model definitions or the
:func:`load_config` factory.

Environment overrides are deliberately minimal:

* ``GENIE_PROVIDER_DEFAULT`` overrides ``provider.default``.
* Per-provider API keys are resolved indirectly: each provider names the env
  var holding its key (``api_key_env``), looked up via :meth:`Settings.resolve_api_key`.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("~/.genie/config.toml")
PROVIDER_DEFAULT_ENV = "GENIE_PROVIDER_DEFAULT"


def _expand(path: str) -> str:
    """Expand a leading ``~`` in a path-like string."""
    return str(Path(path).expanduser())


class ProviderProfile(BaseModel):
    """Shared base for per-provider profiles; carries the API-key env var name."""

    api_key_env: str


class AnthropicProviderConfig(ProviderProfile):
    """Anthropic provider profile."""

    api_key_env: str = "ANTHROPIC_API_KEY"


class OpenAIProviderConfig(ProviderProfile):
    """OpenAI provider profile.

    ``api`` selects the OpenAI surface: ``"chat_completions"`` (default — the
    only mode implemented today) or ``"responses"`` (deferred; see issue #49).
    The default is the working mode so a default-configured ``openai:`` run
    succeeds rather than raising.
    """

    api_key_env: str = "OPENAI_API_KEY"
    api: str = "chat_completions"


class ProviderConfig(BaseModel):
    """Provider selection and per-provider profiles (SPEC §13 ``[provider]``)."""

    default: str = "anthropic:claude-sonnet-4-6"
    anthropic: AnthropicProviderConfig = Field(default_factory=AnthropicProviderConfig)
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)


class LoopConfig(BaseModel):
    """Agent-loop limits (SPEC §13 ``[loop]``)."""

    max_iterations: int = 50
    compaction_threshold: float = 0.8


class BashToolConfig(BaseModel):
    """Bash tool execution limits (SPEC §13 ``[tools.bash]``)."""

    timeout_seconds: int = 30
    max_output_bytes: int = 8192


class ToolsConfig(BaseModel):
    """Tool configuration aggregate (SPEC §13 ``[tools.*]``)."""

    bash: BashToolConfig = Field(default_factory=BashToolConfig)


class SandboxConfig(BaseModel):
    """Sandbox backend selection (SPEC §13 ``[sandbox]``)."""

    backend: str = "local_subprocess"
    working_dir_only: bool = True


class ApprovalConfig(BaseModel):
    """Approval policy and dangerous-command patterns (SPEC §13 ``[approval]``)."""

    mode: str = "ask"
    dangerous_patterns: list[str] = Field(
        default_factory=lambda: ["rm -rf /", "git push", "curl .* | sh"]
    )


class MemoryConfig(BaseModel):
    """Project and user memory file locations (SPEC §13 ``[memory]``)."""

    project_file: str = "AGENTS.md"
    user_file: str = "~/.genie/MEMORY.md"

    def resolved_user_file(self) -> str:
        """Return ``user_file`` with a leading ``~`` expanded to the home dir."""
        return _expand(self.user_file)


class SkillsConfig(BaseModel):
    """Skill discovery directories (SPEC §13 ``[skills]``)."""

    dirs: list[str] = Field(default_factory=lambda: ["~/.genie/skills"])

    def resolved_dirs(self) -> list[str]:
        """Return ``dirs`` with leading ``~`` expanded to the home dir."""
        return [_expand(d) for d in self.dirs]


class Settings(BaseModel):
    """Top-level configuration aggregating every section (SPEC §13).

    Construct via :func:`load_config`, which layers a TOML file and environment
    overrides on top of these defaults. Constructing ``Settings()`` directly
    yields the documented defaults.
    """

    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)

    def provider_parts(self) -> tuple[str, str]:
        """Split ``provider.default`` into ``(provider_name, model)``.

        Splits on the first ``:``. Surrounding whitespace on either side is
        stripped. Raises :class:`ValueError` if the value has no colon or an
        empty/whitespace-only side.
        """
        default = self.provider.default
        name, sep, model = default.partition(":")
        name, model = name.strip(), model.strip()
        if not sep or not name or not model:
            raise ValueError(f"provider.default must be 'provider:model', got {default!r}")
        return name, model

    def resolve_api_key(self, provider_name: str, env: Mapping[str, str]) -> str | None:
        """Return the API key for ``provider_name`` from ``env``, or ``None``.

        Looks up the provider's configured ``api_key_env`` variable name in the
        supplied environment mapping. A missing entry — or one set to an empty
        string — returns ``None`` so the caller decides whether that is fatal;
        see :meth:`require_api_key`.
        """
        return env.get(self._api_key_env(provider_name)) or None

    def require_api_key(self, provider_name: str, env: Mapping[str, str]) -> str:
        """Like :meth:`resolve_api_key` but raise if the key is absent.

        Raises :class:`ValueError` naming the expected environment variable so
        the user knows exactly what to set.
        """
        env_var = self._api_key_env(provider_name)
        value = env.get(env_var)
        if not value:
            raise ValueError(
                f"Missing API key for provider {provider_name!r}: "
                f"set the {env_var} environment variable."
            )
        return value

    def _api_key_env(self, provider_name: str) -> str:
        """Return the env-var name holding ``provider_name``'s API key."""
        profile = getattr(self.provider, provider_name, None)
        if not isinstance(profile, ProviderProfile):
            raise ValueError(f"Unknown provider {provider_name!r}")
        return profile.api_key_env


def _read_toml(path: Path) -> dict:
    """Read and parse a TOML config file (the pluggable source seam).

    Returns an empty mapping if the file does not exist, so missing config is
    never an error. Path ``~`` is expanded before reading.
    """
    resolved = path.expanduser()
    if not resolved.is_file():
        return {}
    with resolved.open("rb") as fh:
        return tomllib.load(fh)


def _apply_env_overrides(data: dict, env: Mapping[str, str]) -> dict:
    """Layer the supported environment overrides onto a config mapping."""
    provider_default = env.get(PROVIDER_DEFAULT_ENV)
    if provider_default:
        provider = dict(data.get("provider") or {})
        provider["default"] = provider_default
        data = {**data, "provider": provider}
    return data


def load_config(
    path: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Load :class:`Settings`, layering TOML and env over the defaults.

    Precedence is ``defaults < TOML file < environment``. ``path`` defaults to
    ``~/.genie/config.toml``; a missing file yields pure defaults rather than an
    error. A partial TOML file overrides only the keys it specifies, leaving
    every other default intact. ``env`` defaults to ``os.environ`` and is the
    only place environment overrides (``GENIE_PROVIDER_DEFAULT``) are read; pass
    an explicit mapping to keep callers pure and testable.
    """
    if env is None:
        import os

        env = os.environ

    config_path = path if path is not None else DEFAULT_CONFIG_PATH
    data = _read_toml(config_path)
    data = _apply_env_overrides(data, env)
    return Settings.model_validate(data)
