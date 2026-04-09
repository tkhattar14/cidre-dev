from __future__ import annotations
import base64
from pathlib import Path
import httpx
from cidre.config import EMBEDDING_MODELS


class OllamaEmbedding:
    def __init__(self, model: str = "embeddinggemma", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    @property
    def dimensions(self) -> int:
        return EMBEDDING_MODELS.get(self._model, {}).get("dimensions", 768)

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            resp = httpx.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": text},
                timeout=120,
            )
            resp.raise_for_status()
            results.append(resp.json()["embeddings"][0])
        return results


class OllamaLLM:
    def __init__(self, model: str = "gemma4:26b-a4b", base_url: str = "http://localhost:11434"):
        self.model = model
        self._base_url = base_url

    def generate(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self._base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def generate_with_image(self, prompt: str, image_path: Path) -> str:
        image_bytes = image_path.read_bytes()
        b64_image = base64.b64encode(image_bytes).decode()
        resp = httpx.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["response"]
