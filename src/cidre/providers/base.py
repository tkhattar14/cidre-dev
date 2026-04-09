from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def dimensions(self) -> int: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def get_provider(
    provider_type: str,
    model: str,
    api_key: str | None = None,
    base_url: str = "http://localhost:11434",
) -> EmbeddingProvider:
    if provider_type == "ollama":
        from cidre.providers.ollama import OllamaEmbedding
        return OllamaEmbedding(model=model, base_url=base_url)
    elif provider_type == "openai":
        if not api_key:
            raise ValueError("OpenAI provider requires an API key")
        from cidre.providers.openai import OpenAIEmbedding
        return OpenAIEmbedding(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
