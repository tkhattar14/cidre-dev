from __future__ import annotations
from cidre.config import EMBEDDING_MODELS


class OpenAIEmbedding:
    def __init__(self, model: str = "text-embedding-3-large", api_key: str = ""):
        self._model = model
        self._api_key = api_key

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    @property
    def dimensions(self) -> int:
        return EMBEDDING_MODELS.get(self._model, {}).get("dimensions", 3072)

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        client = OpenAI(api_key=self._api_key)
        resp = client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]
