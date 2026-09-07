"""Tests for the typed config loader (SPEC §13)."""

from __future__ import annotations

from pathlib import Path
from tomllib import TOMLDecodeError

import pytest
from pydantic import ValidationError

from genie.config import (
    PROVIDER_DEFAULT_ENV,
    Settings,
    load_config,
)


def test_defaults_when_no_file(tmp_path: Path):
    """A nonexistent path yields the documented defaults, not an error."""
    settings = load_config(tmp_path / "missing.toml", env={})

    assert settings.provider.default == "anthropic:claude-sonnet-4-6"
    assert settings.provider.anthropic.api_key_env == "ANTHROPIC_API_KEY"
    assert settings.provider.openai.api_key_env == "OPENAI_API_KEY"
    assert settings.provider.openai.api == "chat_completions"
    assert settings.loop.max_iterations == 50
    assert settings.loop.compaction_threshold == 0.8
    assert settings.tools.bash.timeout_seconds == 30
    assert settings.tools.bash.max_output_bytes == 8192
    assert settings.sandbox.backend == "local_subprocess"
    assert settings.sandbox.working_dir_only is True
    assert settings.approval.mode == "ask"
    assert settings.approval.dangerous_patterns == ["rm -rf /", "git push", "curl .* | sh"]
    assert settings.memory.project_file == "AGENTS.md"
    assert settings.memory.user_file == "~/.genie/MEMORY.md"
    assert settings.skills.dirs == ["~/.genie/skills"]


def test_default_path_constant_is_genie_config():
    """The default search path points at ~/.genie/config.toml."""
    from genie.config import DEFAULT_CONFIG_PATH

    assert Path("~/.genie/config.toml") == DEFAULT_CONFIG_PATH


@pytest.mark.parametrize(
    "section, key",
    [
        ("", "provdier"),
        ("provider", "defualt"),
        ("provider.anthropic", "api_key_en"),
        ("provider.openai", "api_key_en"),
        ("loop", "max_iteratoins"),
        ("tools", "bsh"),
        ("tools.bash", "timeout_second"),
        ("sandbox", "working_dir_onyl"),
        ("approval", "dangerous_pattern"),
        ("memory", "project_flie"),
        ("skills", "dir"),
    ],
)
def test_unknown_config_keys_are_rejected(tmp_path: Path, section: str, key: str):
    """Typos fail with the full field path at every configuration level."""
    config = tmp_path / "config.toml"
    header = f"[{section}]\n" if section else ""
    config.write_text(f'{header}{key} = "typo"\n')

    with pytest.raises(ValidationError) as exc_info:
        load_config(config, env={})

    error = exc_info.value.errors()[0]
    expected_path = (*section.split("."), key) if section else (key,)
    assert error["loc"] == expected_path
    assert error["type"] == "extra_forbidden"


def test_unknown_config_key_is_not_hidden_by_env_override(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text('[provider]\ndefualt = "openai:gpt-4o"\n')

    with pytest.raises(ValidationError, match=r"provider\.defualt"):
        load_config(config, env={PROVIDER_DEFAULT_ENV: "openai:gpt-4o-mini"})


def test_invalid_toml_names_expanded_config_path(tmp_path: Path, monkeypatch):
    """Syntax errors identify the actual file and preserve the parser diagnostic."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = tmp_path / "broken.toml"
    config.write_text("[provider\n")

    with pytest.raises(ValueError, match="Invalid TOML") as exc_info:
        load_config(Path("~/broken.toml"), env={})

    assert str(config) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, TOMLDecodeError)
    assert str(exc_info.value.__cause__) in str(exc_info.value)


def test_partial_toml_overrides_only_specified_keys(tmp_path: Path):
    """A partial TOML file overrides its keys and preserves other defaults."""
    config = tmp_path / "config.toml"
    config.write_text(
        "\n".join(
            [
                "[provider]",
                'default = "openai:gpt-4o-mini"',
                "",
                "[loop]",
                "max_iterations = 7",
                "",
                "[tools.bash]",
                "timeout_seconds = 99",
                "",
                "[approval]",
                'mode = "deny"',
            ]
        )
    )

    settings = load_config(config, env={})

    assert settings.provider.default == "openai:gpt-4o-mini"
    assert settings.loop.max_iterations == 7
    assert settings.tools.bash.timeout_seconds == 99
    assert settings.approval.mode == "deny"

    # Unspecified keys keep their defaults.
    assert settings.loop.compaction_threshold == 0.8
    assert settings.tools.bash.max_output_bytes == 8192
    assert settings.provider.anthropic.api_key_env == "ANTHROPIC_API_KEY"
    assert settings.approval.dangerous_patterns == ["rm -rf /", "git push", "curl .* | sh"]


def test_env_override_beats_toml(tmp_path: Path):
    """GENIE_PROVIDER_DEFAULT wins over the TOML provider.default."""
    config = tmp_path / "config.toml"
    config.write_text('[provider]\ndefault = "openai:gpt-4o-mini"\n')

    settings = load_config(
        config,
        env={PROVIDER_DEFAULT_ENV: "anthropic:claude-opus-4"},
    )

    assert settings.provider.default == "anthropic:claude-opus-4"


def test_env_override_with_no_file():
    """GENIE_PROVIDER_DEFAULT applies even when no TOML file exists."""
    settings = load_config(
        Path("/nonexistent/genie/config.toml"),
        env={PROVIDER_DEFAULT_ENV: "openai:gpt-4o"},
    )

    assert settings.provider.default == "openai:gpt-4o"
    # Other sections still defaulted.
    assert settings.loop.max_iterations == 50


def test_empty_env_override_is_ignored(tmp_path: Path):
    """An empty GENIE_PROVIDER_DEFAULT does not clobber the TOML value."""
    config = tmp_path / "config.toml"
    config.write_text('[provider]\ndefault = "openai:gpt-4o-mini"\n')

    settings = load_config(config, env={PROVIDER_DEFAULT_ENV: ""})

    assert settings.provider.default == "openai:gpt-4o-mini"


def test_env_defaults_to_os_environ_path(tmp_path: Path):
    """Omitting env reads os.environ without raising (defaults still load)."""
    settings = load_config(tmp_path / "missing.toml")

    assert isinstance(settings, Settings)
    assert settings.loop.max_iterations == 50


def test_provider_parts_splits_on_first_colon():
    """provider_parts splits into (name, model) on the first colon."""
    settings = Settings()
    settings.provider.default = "anthropic:claude-sonnet-4-6"

    assert settings.provider_parts() == ("anthropic", "claude-sonnet-4-6")


def test_provider_parts_keeps_later_colons_in_model():
    """Only the first colon is a separator; later colons stay in the model."""
    settings = Settings()
    settings.provider.default = "openai:org:weird:model"

    assert settings.provider_parts() == ("openai", "org:weird:model")


@pytest.mark.parametrize(
    "bad", ["anthropic", "", ":model", "provider:", " : ", "anthropic: ", " :model"]
)
def test_provider_parts_raises_on_malformed(bad: str):
    """Malformed provider.default (incl. whitespace-only sides) raises a clear ValueError."""
    settings = Settings()
    settings.provider.default = bad

    with pytest.raises(ValueError, match="provider:model"):
        settings.provider_parts()


def test_provider_parts_strips_surrounding_whitespace():
    """Whitespace around the provider/model is trimmed, not preserved."""
    settings = Settings()
    settings.provider.default = "  anthropic : claude-sonnet-4-6  "

    assert settings.provider_parts() == ("anthropic", "claude-sonnet-4-6")


def test_resolve_api_key_returns_value_when_present():
    """resolve_api_key returns the env value for the configured var name."""
    settings = Settings()
    env = {"ANTHROPIC_API_KEY": "sk-ant-123"}

    assert settings.resolve_api_key("anthropic", env) == "sk-ant-123"


def test_resolve_api_key_returns_none_when_absent():
    """resolve_api_key returns None when the env var is not set."""
    settings = Settings()

    assert settings.resolve_api_key("openai", {}) is None


def test_resolve_api_key_returns_none_for_empty_string():
    """An env var set to an empty string counts as absent (None)."""
    settings = Settings()
    env = {"ANTHROPIC_API_KEY": ""}

    assert settings.resolve_api_key("anthropic", env) is None


def test_resolve_api_key_honors_custom_env_var_name(tmp_path: Path):
    """A custom api_key_env in TOML is what resolve_api_key looks up."""
    config = tmp_path / "config.toml"
    config.write_text('[provider.anthropic]\napi_key_env = "MY_KEY"\n')

    settings = load_config(config, env={})

    assert settings.resolve_api_key("anthropic", {"MY_KEY": "secret"}) == "secret"
    assert settings.resolve_api_key("anthropic", {"ANTHROPIC_API_KEY": "x"}) is None


def test_resolve_api_key_unknown_provider_raises():
    """Resolving an unknown provider name raises ValueError."""
    settings = Settings()

    with pytest.raises(ValueError, match="Unknown provider"):
        settings.resolve_api_key("nope", {})


def test_require_api_key_returns_value_when_present():
    """require_api_key returns the value when the env var is set."""
    settings = Settings()

    assert settings.require_api_key("openai", {"OPENAI_API_KEY": "sk-oai"}) == "sk-oai"


def test_require_api_key_raises_naming_the_env_var():
    """require_api_key raises with the env var name in the message."""
    settings = Settings()

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.require_api_key("openai", {})


def test_loaded_skills_dirs_kept_verbatim(tmp_path: Path):
    """Dirs sourced from a TOML file are stored verbatim (no eager expansion)."""
    config = tmp_path / "config.toml"
    config.write_text('[skills]\ndirs = ["~/custom/skills"]\n')

    settings = load_config(config, env={})

    assert settings.skills.dirs == ["~/custom/skills"]
