"""The single startup wiring point for provider clients.

``provider_factory`` turns a ``"provider:model"`` spec string into a concrete
:class:`ProviderClient`. Real SDK-backed providers are imported **lazily**,
inside their loader functions, so importing this module never imports the
``anthropic`` or ``openai`` SDKs — and never requires their client modules to
exist yet. This is what keeps this PR independent of the not-yet-built
``anthropic_client`` / ``openai_client`` modules: a missing module degrades to
a clear, actionable error only when that provider is actually requested.

Adding a provider later is a one-line edit to :data:`_REGISTRY` — the
extension seam.
"""

from __future__ import annotations

from collections.abc import Callable

from genie.providers.base import ProviderClient
from genie.providers.fake import FakeProvider


def _load_fake(model: str, **kwargs) -> ProviderClient:
    """Loader for the always-available scripted provider."""
    return FakeProvider(model=model, **kwargs)


def _load_anthropic(model: str, *, settings: object | None = None, **kwargs) -> ProviderClient:
    """Lazily load the Anthropic adapter (built in a later PR)."""
    try:
        # Built in a later PR; guarded so this PR stays self-contained.
        from genie.providers.anthropic_client import (  # pyright: ignore[reportMissingImports]
            AnthropicClient,
        )
    except ImportError as exc:  # pragma: no cover - exercised via factory test
        raise RuntimeError(
            "anthropic provider not yet available (genie.providers.anthropic_client missing)"
        ) from exc
    return AnthropicClient(model=model, settings=settings, **kwargs)


def _load_openai(model: str, *, settings: object | None = None, **kwargs) -> ProviderClient:
    """Lazily load the OpenAI adapter (built in a later PR)."""
    try:
        # Built in a later PR; guarded so this PR stays self-contained.
        from genie.providers.openai_client import (  # pyright: ignore[reportMissingImports]
            OpenAIClient,
        )
    except ImportError as exc:  # pragma: no cover - exercised via factory test
        raise RuntimeError(
            "openai provider not yet available (genie.providers.openai_client missing)"
        ) from exc
    return OpenAIClient(model=model, settings=settings, **kwargs)


# Registry of provider name -> loader callable. Adding a provider is one line.
_REGISTRY: dict[str, Callable[..., ProviderClient]] = {
    "fake": _load_fake,
    "anthropic": _load_anthropic,
    "openai": _load_openai,
}


def provider_factory(spec: str, *, settings: object | None = None, **kwargs) -> ProviderClient:
    """Build a :class:`ProviderClient` from a ``"provider:model"`` spec.

    Args:
        spec: ``"provider:model"``, e.g. ``"anthropic:claude-sonnet-4-6"`` or
            ``"fake:fake-1"``. Split on the first colon; the model may itself
            contain colons.
        settings: Optional settings object passed through to real providers so
            they can read credentials/profile; the ``fake`` provider ignores it.
        **kwargs: Forwarded to the concrete provider constructor.

    Returns:
        A ready-to-use provider client.

    Raises:
        ValueError: If ``spec`` has no colon, or names an unknown provider.
        RuntimeError: If a known provider's client module is not yet available.
    """
    if ":" not in spec:
        raise ValueError(
            f"invalid provider spec {spec!r}: expected 'provider:model' "
            "(e.g. 'anthropic:claude-sonnet-4-6')"
        )
    name, model = spec.split(":", 1)
    loader = _REGISTRY.get(name)
    if loader is None:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown provider {name!r}; supported providers: {supported}")
    if name == "fake":
        return loader(model, **kwargs)
    return loader(model, settings=settings, **kwargs)
