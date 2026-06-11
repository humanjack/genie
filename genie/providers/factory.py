"""The single startup wiring point for provider clients.

``provider_factory`` turns a ``"provider:model"`` spec string into a concrete
:class:`ProviderClient`. Real SDK-backed adapters are imported **lazily**,
inside their loader functions, so importing this module stays cheap and pulls
in only the provider actually requested.

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
    """Lazily load the Anthropic adapter."""
    from genie.providers.anthropic_client import AnthropicClient

    return AnthropicClient(model=model, settings=settings, **kwargs)


def _load_openai(model: str, *, settings: object | None = None, **kwargs) -> ProviderClient:
    """Lazily load the OpenAI adapter."""
    from genie.providers.openai_client import OpenAIClient

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
