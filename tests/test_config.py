"""Tests for the typed config loader (SPEC §13)."""

from __future__ import annotations

from pathlib import Path

import pytest

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
    assert settings.provider.openai.api == "responses"
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


@pytest.mark.parametrize("bad", ["anthropic", "", ":model", "provider:"])
def test_provider_parts_raises_on_malformed(bad: str):
    """Malformed provider.default raises a clear ValueError."""
    settings = Settings()
    settings.provider.default = bad

    with pytest.raises(ValueError, match="provider:model"):
        settings.provider_parts()


def test_resolve_api_key_returns_value_when_present():
    """resolve_api_key returns the env value for the configured var name."""
    settings = Settings()
    env = {"ANTHROPIC_API_KEY": "sk-ant-123"}

    assert settings.resolve_api_key("anthropic", env) == "sk-ant-123"


def test_resolve_api_key_returns_none_when_absent():
    """resolve_api_key returns None when the env var is not set."""
    settings = Settings()

    assert settings.resolve_api_key("openai", {}) is None


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


def test_memory_user_file_tilde_expansion():
    """memory.resolved_user_file expands a leading ~ to the home dir."""
    settings = Settings()
    expanded = settings.memory.resolved_user_file()

    assert "~" not in expanded
    assert expanded.endswith("/.genie/MEMORY.md")
    assert expanded == str(Path.home() / ".genie" / "MEMORY.md")


def test_skills_dirs_tilde_expansion():
    """skills.resolved_dirs expands a leading ~ in each entry."""
    settings = Settings()
    settings.skills.dirs = ["~/.genie/skills", "/abs/path"]

    resolved = settings.skills.resolved_dirs()

    assert resolved[0] == str(Path.home() / ".genie" / "skills")
    assert resolved[1] == "/abs/path"
    assert all("~" not in d for d in resolved)


def test_loaded_skills_dirs_expand(tmp_path: Path):
    """Tilde expansion works on dirs sourced from a TOML file."""
    config = tmp_path / "config.toml"
    config.write_text('[skills]\ndirs = ["~/custom/skills"]\n')

    settings = load_config(config, env={})

    assert settings.skills.dirs == ["~/custom/skills"]
    assert settings.skills.resolved_dirs() == [str(Path.home() / "custom" / "skills")]
