from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tomli_w

CIDRE_HOME = Path.home() / ".cidre"

DEFAULT_EXCLUDE = [
    "*.py", "*.js", "*.ts", "*.swift", "*.go", "*.rs",
    "*.java", "*.c", "*.cpp", "*.h",
    "node_modules", ".git", ".venv", "__pycache__", "Library", ".Trash",
]

EMBEDDING_MODELS = {
    "embeddinggemma": {"provider": "ollama", "dimensions": 768},
    "bge-m3": {"provider": "ollama", "dimensions": 1024},
    "qwen3-embedding:4b": {"provider": "ollama", "dimensions": 4096},
    "text-embedding-3-large": {"provider": "openai", "dimensions": 3072},
}

LLM_MODELS = {
    "gemma4": {"memory_gb": 9.6, "label": "E4B 8B vision+audio (default)"},
    "gemma4:26b": {"memory_gb": 17, "label": "26B MoE (recommended for 32GB+)"},
    "gemma4:31b": {"memory_gb": 20, "label": "31B Dense (for 48GB+)"},
}


@dataclass
class CidreConfig:
    llm_model: str = "gemma4"
    embedding_provider: str = "ollama"
    embedding_model: str = "embeddinggemma"
    embedding_dimensions: int = 768
    sources_watched: list[str] = field(default_factory=list)
    photos_enabled: bool = False
    exclude_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))


def save_config(config: CidreConfig, path: Path) -> None:
    data = {
        "general": {"llm_model": config.llm_model},
        "embedding": {
            "provider": config.embedding_provider,
            "model": config.embedding_model,
            "dimensions": config.embedding_dimensions,
        },
        "sources": {
            "watched": config.sources_watched,
            "photos": config.photos_enabled,
        },
        "exclude": {"patterns": config.exclude_patterns},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tomli_w.dumps(data).encode())


def load_config(path: Path) -> CidreConfig:
    if not path.exists():
        return CidreConfig()

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    data = tomllib.loads(path.read_text())
    config = CidreConfig()
    if "general" in data:
        config.llm_model = data["general"].get("llm_model", config.llm_model)
    if "embedding" in data:
        config.embedding_provider = data["embedding"].get("provider", config.embedding_provider)
        config.embedding_model = data["embedding"].get("model", config.embedding_model)
        config.embedding_dimensions = data["embedding"].get("dimensions", config.embedding_dimensions)
    if "sources" in data:
        config.sources_watched = data["sources"].get("watched", config.sources_watched)
        config.photos_enabled = data["sources"].get("photos", config.photos_enabled)
    if "exclude" in data:
        config.exclude_patterns = data["exclude"].get("patterns", config.exclude_patterns)
    return config
