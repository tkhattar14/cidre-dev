# Cidre Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Cidre, a local-only CLI tool for macOS that indexes documents, photos, videos, and notes using Gemma 4 via Ollama and provides instant semantic search via configurable embeddings.

**Architecture:** Python monorepo with modular packages — providers (embedding/LLM abstraction), indexer (file scanning, pipeline, Apple Photos, daemon), search (vector engine, query parsing), vision (image/video description, categorization). CLI via typer, storage via SQLite + sqlite-vec. Landing page as static HTML/CSS.

**Tech Stack:** Python 3.12+, typer, sqlite-vec, osxphotos, watchdog, pdfplumber, Ollama, launchd

**Spec:** `docs/superpowers/specs/2026-04-09-cidre-design.md`

---

## Task 1: Project Scaffolding

**Agent:** core-architect
**Files:**
- Create: `pyproject.toml`
- Create: `src/cidre/__init__.py`
- Create: `src/cidre/cli.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `LICENSE`
- Create: `README.md`
- Create: `.gitignore`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/tushar/Documents/matrix/gemma4_search
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "cidre"
version = "0.1.0"
description = "Local-first search intelligence for macOS"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
dependencies = [
    "typer>=0.15",
    "rich>=13.0",
    "sqlite-vec>=0.1",
    "osxphotos>=0.68",
    "watchdog>=4.0",
    "pdfplumber>=0.11",
    "httpx>=0.27",
    "tomli>=2.0;python_version<'3.11'",
    "tomli-w>=1.0",
    "pillow>=10.0",
]

[project.optional-dependencies]
openai = ["openai>=1.0"]
dev = [
    "pytest>=8.0",
    "pytest-tmp-files>=0.1",
    "ruff>=0.5",
    "mypy>=1.10",
]

[project.scripts]
cidre = "cidre.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/cidre"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create src/cidre/__init__.py**

```python
"""Cidre — Local-first search intelligence for macOS."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create minimal CLI entry point**

```python
# src/cidre/cli.py
import typer

app = typer.Typer(
    name="cidre",
    help="Local-first search intelligence for macOS",
    no_args_is_help=True,
)


@app.command()
def version():
    """Show Cidre version."""
    from cidre import __version__
    typer.echo(f"cidre {__version__}")
```

- [ ] **Step 5: Create tests scaffolding**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
import pytest
from pathlib import Path


@pytest.fixture
def tmp_cidre_home(tmp_path):
    """Provides a temporary ~/.cidre directory for testing."""
    cidre_home = tmp_path / ".cidre"
    cidre_home.mkdir()
    return cidre_home


@pytest.fixture
def sample_files(tmp_path):
    """Creates sample files for indexing tests."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("# Meeting Notes\nDiscussed project timeline.")
    (docs / "readme.txt").write_text("Project readme content.")

    photos = tmp_path / "photos"
    photos.mkdir()

    return tmp_path
```

- [ ] **Step 6: Create LICENSE (MIT)**

```
MIT License

Copyright (c) 2026 Cidre Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 7: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
dist/
*.egg-info/
.mypy_cache/
.ruff_cache/
.pytest_cache/
.cidre/
*.db
```

- [ ] **Step 8: Create README.md**

```markdown
# Cidre

Local-first search intelligence for macOS.

Index your documents, photos, videos, and notes with AI — search them instantly. Nothing leaves your machine.

## Features

- Semantic search across photos, PDFs, markdown, videos
- Apple Photos integration (read-only)
- Auto-categorization via Gemma 4
- Background indexing daemon
- Choose your embedding model (EmbeddingGemma, bge-m3, qwen3, OpenAI)
- 100% local — powered by Ollama

## Requirements

- macOS (Apple Silicon recommended)
- [Ollama](https://ollama.com) installed
- Python 3.12+

## Install

```bash
pip install cidre
```

## Quick Start

```bash
cidre init
cidre add ~/Documents
cidre add --photos
cidre index
cidre search "sunset photos"
```

## License

MIT
```

- [ ] **Step 9: Install in dev mode and verify**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cidre version
```

Expected: `cidre 0.1.0`

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml src/ tests/ LICENSE README.md .gitignore
git commit -m "feat: scaffold cidre project with CLI entry point"
```

---

## Task 2: Config System

**Agent:** core-architect
**Files:**
- Create: `src/cidre/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for config**

```python
# tests/test_config.py
from pathlib import Path
from cidre.config import CidreConfig, load_config, save_config


def test_default_config():
    config = CidreConfig()
    assert config.llm_model == "gemma4:26b-a4b"
    assert config.embedding_provider == "ollama"
    assert config.embedding_model == "embeddinggemma"
    assert config.embedding_dimensions == 768
    assert config.sources_watched == []
    assert config.photos_enabled is False
    assert config.exclude_patterns == [
        "*.py", "*.js", "*.ts", "*.swift", "*.go", "*.rs",
        "*.java", "*.c", "*.cpp", "*.h",
        "node_modules", ".git", ".venv", "__pycache__", "Library", ".Trash",
    ]


def test_save_and_load_config(tmp_cidre_home):
    config = CidreConfig()
    config.sources_watched = ["~/Documents", "~/Desktop"]
    config.photos_enabled = True
    save_config(config, tmp_cidre_home / "config.toml")

    loaded = load_config(tmp_cidre_home / "config.toml")
    assert loaded.sources_watched == ["~/Documents", "~/Desktop"]
    assert loaded.photos_enabled is True
    assert loaded.llm_model == "gemma4:26b-a4b"


def test_load_missing_config_returns_default(tmp_cidre_home):
    loaded = load_config(tmp_cidre_home / "config.toml")
    assert loaded.llm_model == "gemma4:26b-a4b"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'cidre.config'`

- [ ] **Step 3: Implement config.py**

```python
# src/cidre/config.py
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
    "gemma4:26b-a4b": {"memory_gb": 7, "label": "26B MoE (recommended for 24GB+)"},
    "gemma4:e4b": {"memory_gb": 2.4, "label": "E4B (for 16GB machines)"},
    "gemma4:31b": {"memory_gb": 11, "label": "31B Dense (for 32GB+)"},
}


@dataclass
class CidreConfig:
    llm_model: str = "gemma4:26b-a4b"
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/cidre/config.py tests/test_config.py
git commit -m "feat: add config system with TOML persistence"
```

---

## Task 3: Database Schema

**Agent:** core-architect
**Files:**
- Create: `src/cidre/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write failing tests for database**

```python
# tests/test_db.py
import sqlite3
from pathlib import Path
from cidre.db import init_db, insert_item, search_by_metadata, get_item_by_path, ItemRow


def test_init_db_creates_tables(tmp_path):
    db_path = tmp_path / "cidre.db"
    conn = init_db(db_path, embedding_dimensions=768)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "items" in tables
    conn.close()


def test_insert_and_retrieve_item(tmp_path):
    db_path = tmp_path / "cidre.db"
    conn = init_db(db_path, embedding_dimensions=768)
    embedding = [0.1] * 768

    insert_item(conn, ItemRow(
        file_path="/Users/test/photo.jpg",
        file_hash="abc123",
        file_type="photo",
        file_size=2400000,
        modified_at="2024-06-15T10:30:00",
        ai_description="A sunset over the ocean",
        categories=["travel", "landscape"],
        summary="Sunset photo from beach vacation",
        embedding=embedding,
        source="filesystem",
    ))

    item = get_item_by_path(conn, "/Users/test/photo.jpg")
    assert item is not None
    assert item.ai_description == "A sunset over the ocean"
    assert item.categories == ["travel", "landscape"]
    assert item.file_hash == "abc123"
    conn.close()


def test_search_by_metadata_type_filter(tmp_path):
    db_path = tmp_path / "cidre.db"
    conn = init_db(db_path, embedding_dimensions=768)
    embedding = [0.1] * 768

    insert_item(conn, ItemRow(
        file_path="/test/doc.pdf",
        file_hash="hash1",
        file_type="document",
        file_size=100000,
        modified_at="2025-01-10T00:00:00",
        ai_description="Health insurance invoice",
        categories=["receipt", "insurance"],
        summary="Niva Bupa invoice",
        embedding=embedding,
        source="filesystem",
    ))
    insert_item(conn, ItemRow(
        file_path="/test/photo.jpg",
        file_hash="hash2",
        file_type="photo",
        file_size=200000,
        modified_at="2025-01-10T00:00:00",
        ai_description="Mountain landscape",
        categories=["landscape"],
        summary="Mountain photo",
        embedding=embedding,
        source="filesystem",
    ))

    results = search_by_metadata(conn, file_type="document")
    assert len(results) == 1
    assert results[0].file_path == "/test/doc.pdf"
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_db.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'cidre.db'`

- [ ] **Step 3: Implement db.py**

```python
# src/cidre/db.py
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec


@dataclass
class ItemRow:
    file_path: str
    file_hash: str
    file_type: str  # "photo", "video", "document", "markdown"
    file_size: int
    modified_at: str
    ai_description: str
    categories: list[str]
    summary: str
    embedding: list[float]
    source: str  # "filesystem" | "apple_photos"
    rowid: int | None = None


def init_db(db_path: Path, embedding_dimensions: int) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS items (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            file_hash TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            modified_at TEXT NOT NULL,
            ai_description TEXT NOT NULL,
            categories TEXT NOT NULL DEFAULT '[]',
            summary TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'filesystem'
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS items_vec USING vec0(
            rowid INTEGER PRIMARY KEY,
            embedding float[{embedding_dimensions}]
        );

        CREATE INDEX IF NOT EXISTS idx_items_file_type ON items(file_type);
        CREATE INDEX IF NOT EXISTS idx_items_source ON items(source);
        CREATE INDEX IF NOT EXISTS idx_items_modified_at ON items(modified_at);
    """)
    return conn


def insert_item(conn: sqlite3.Connection, item: ItemRow) -> int:
    cursor = conn.execute(
        """INSERT OR REPLACE INTO items
           (file_path, file_hash, file_type, file_size, modified_at,
            ai_description, categories, summary, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            item.file_path, item.file_hash, item.file_type, item.file_size,
            item.modified_at, item.ai_description, json.dumps(item.categories),
            item.summary, item.source,
        ),
    )
    rowid = cursor.lastrowid
    conn.execute(
        "INSERT OR REPLACE INTO items_vec (rowid, embedding) VALUES (?, ?)",
        (rowid, json.dumps(item.embedding)),
    )
    conn.commit()
    return rowid


def get_item_by_path(conn: sqlite3.Connection, file_path: str) -> ItemRow | None:
    row = conn.execute(
        "SELECT rowid, * FROM items WHERE file_path = ?", (file_path,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_item(row)


def search_by_metadata(
    conn: sqlite3.Connection,
    file_type: str | None = None,
    source: str | None = None,
    category: str | None = None,
) -> list[ItemRow]:
    query = "SELECT rowid, * FROM items WHERE 1=1"
    params: list = []
    if file_type:
        query += " AND file_type = ?"
        params.append(file_type)
    if source:
        query += " AND source = ?"
        params.append(source)
    if category:
        query += " AND categories LIKE ?"
        params.append(f'%"{category}"%')

    rows = conn.execute(query, params).fetchall()
    return [_row_to_item(r) for r in rows]


def search_by_vector(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    k: int = 20,
) -> list[tuple[int, float]]:
    """Returns list of (rowid, distance) pairs."""
    rows = conn.execute(
        "SELECT rowid, distance FROM items_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (json.dumps(query_embedding), k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_items_by_rowids(conn: sqlite3.Connection, rowids: list[int]) -> list[ItemRow]:
    if not rowids:
        return []
    placeholders = ",".join("?" * len(rowids))
    rows = conn.execute(
        f"SELECT rowid, * FROM items WHERE rowid IN ({placeholders})", rowids
    ).fetchall()
    return [_row_to_item(r) for r in rows]


def get_index_stats(conn: sqlite3.Connection) -> dict:
    stats = {}
    row = conn.execute("SELECT COUNT(*) FROM items").fetchone()
    stats["total"] = row[0]
    for ftype in ("photo", "video", "document", "markdown"):
        row = conn.execute("SELECT COUNT(*) FROM items WHERE file_type = ?", (ftype,)).fetchone()
        stats[ftype] = row[0]
    row = conn.execute("SELECT COUNT(*) FROM items WHERE source = 'apple_photos'").fetchone()
    stats["apple_photos"] = row[0]
    return stats


def _row_to_item(row: tuple) -> ItemRow:
    return ItemRow(
        rowid=row[0],
        file_path=row[1],
        file_hash=row[2],
        file_type=row[3],
        file_size=row[4],
        modified_at=row[5],
        ai_description=row[6],
        categories=json.loads(row[7]),
        summary=row[8],
        source=row[9],
        embedding=[],  # not loaded from items table
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/cidre/db.py tests/test_db.py
git commit -m "feat: add SQLite + sqlite-vec database with item storage and vector search"
```

---

## Task 4: Embedding Provider Abstraction

**Agent:** core-architect
**Files:**
- Create: `src/cidre/providers/__init__.py`
- Create: `src/cidre/providers/base.py`
- Create: `src/cidre/providers/ollama.py`
- Create: `src/cidre/providers/openai.py`
- Create: `tests/test_providers.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_providers.py
from unittest.mock import patch, MagicMock
from cidre.providers.base import get_provider
from cidre.providers.ollama import OllamaEmbedding, OllamaLLM
from cidre.providers.openai import OpenAIEmbedding


def test_ollama_embedding_properties():
    provider = OllamaEmbedding(model="embeddinggemma")
    assert provider.name == "ollama:embeddinggemma"
    assert provider.dimensions == 768


def test_ollama_embedding_bge_m3():
    provider = OllamaEmbedding(model="bge-m3")
    assert provider.dimensions == 1024


def test_openai_embedding_properties():
    provider = OpenAIEmbedding(model="text-embedding-3-large", api_key="test-key")
    assert provider.name == "openai:text-embedding-3-large"
    assert provider.dimensions == 3072


def test_get_provider_ollama():
    provider = get_provider("ollama", "embeddinggemma")
    assert isinstance(provider, OllamaEmbedding)
    assert provider.dimensions == 768


def test_get_provider_openai():
    provider = get_provider("openai", "text-embedding-3-large", api_key="test")
    assert isinstance(provider, OpenAIEmbedding)


def test_ollama_llm_properties():
    llm = OllamaLLM(model="gemma4:26b-a4b")
    assert llm.model == "gemma4:26b-a4b"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_providers.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement providers/base.py**

```python
# src/cidre/providers/__init__.py
```

```python
# src/cidre/providers/base.py
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
```

- [ ] **Step 4: Implement providers/ollama.py**

```python
# src/cidre/providers/ollama.py
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
```

- [ ] **Step 5: Implement providers/openai.py**

```python
# src/cidre/providers/openai.py
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
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_providers.py -v
```

Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add src/cidre/providers/ tests/test_providers.py
git commit -m "feat: add embedding provider abstraction with Ollama and OpenAI backends"
```

---

## Task 5: File Scanner

**Agent:** indexer-agent
**Files:**
- Create: `src/cidre/indexer/__init__.py`
- Create: `src/cidre/indexer/scanner.py`
- Create: `tests/test_scanner.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scanner.py
from pathlib import Path
from cidre.indexer.scanner import scan_directory, file_hash, classify_file, should_exclude


def test_classify_photo():
    assert classify_file(Path("photo.jpg")) == "photo"
    assert classify_file(Path("image.png")) == "photo"
    assert classify_file(Path("screenshot.heic")) == "photo"
    assert classify_file(Path("pic.webp")) == "photo"


def test_classify_video():
    assert classify_file(Path("clip.mp4")) == "video"
    assert classify_file(Path("movie.mov")) == "video"


def test_classify_document():
    assert classify_file(Path("invoice.pdf")) == "document"


def test_classify_markdown():
    assert classify_file(Path("notes.md")) == "markdown"


def test_classify_unknown():
    assert classify_file(Path("app.exe")) is None
    assert classify_file(Path("code.py")) is None


def test_should_exclude():
    patterns = ["*.py", "node_modules", ".git"]
    assert should_exclude(Path("src/main.py"), patterns) is True
    assert should_exclude(Path("node_modules/pkg/index.js"), patterns) is True
    assert should_exclude(Path(".git/HEAD"), patterns) is True
    assert should_exclude(Path("docs/notes.md"), patterns) is False
    assert should_exclude(Path("photo.jpg"), patterns) is False


def test_file_hash(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    h1 = file_hash(f)
    assert isinstance(h1, str)
    assert len(h1) == 64  # sha256 hex

    f.write_text("different content")
    h2 = file_hash(f)
    assert h1 != h2


def test_scan_directory(tmp_path):
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (tmp_path / "notes.md").write_text("# Notes")
    (tmp_path / "code.py").write_text("print('hello')")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "doc.pdf").write_bytes(b"%PDF-1.4")

    results = list(scan_directory(tmp_path, exclude_patterns=["*.py"]))
    paths = [r.path for r in results]

    assert tmp_path / "photo.jpg" in paths
    assert tmp_path / "notes.md" in paths
    assert tmp_path / "sub" / "doc.pdf" in paths
    assert tmp_path / "code.py" not in paths
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scanner.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement scanner.py**

```python
# src/cidre/indexer/__init__.py
```

```python
# src/cidre/indexer/scanner.py
from __future__ import annotations

import hashlib
import fnmatch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

FILE_TYPES: dict[str, str] = {
    ".jpg": "photo", ".jpeg": "photo", ".png": "photo",
    ".heic": "photo", ".heif": "photo", ".webp": "photo",
    ".tiff": "photo", ".tif": "photo", ".bmp": "photo",
    ".mp4": "video", ".mov": "video", ".m4v": "video",
    ".avi": "video", ".mkv": "video",
    ".pdf": "document",
    ".md": "markdown", ".markdown": "markdown",
}


@dataclass
class ScannedFile:
    path: Path
    file_type: str
    file_hash: str
    file_size: int
    modified_at: str


def classify_file(path: Path) -> str | None:
    return FILE_TYPES.get(path.suffix.lower())


def should_exclude(path: Path, patterns: list[str]) -> bool:
    path_str = str(path)
    for pattern in patterns:
        if fnmatch.fnmatch(path.name, pattern):
            return True
        for part in path.parts:
            if part == pattern:
                return True
    return False


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_directory(
    directory: Path,
    exclude_patterns: list[str] | None = None,
) -> list[ScannedFile]:
    exclude_patterns = exclude_patterns or []
    results: list[ScannedFile] = []

    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if should_exclude(path.relative_to(directory), exclude_patterns):
            continue
        ftype = classify_file(path)
        if ftype is None:
            continue

        stat = path.stat()
        results.append(ScannedFile(
            path=path,
            file_type=ftype,
            file_hash=file_hash(path),
            file_size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
        ))

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scanner.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/cidre/indexer/ tests/test_scanner.py
git commit -m "feat: add file scanner with type classification and exclude patterns"
```

---

## Task 6: Vision — Image & Video Description

**Agent:** vision-agent
**Files:**
- Create: `src/cidre/vision/__init__.py`
- Create: `src/cidre/vision/describe.py`
- Create: `src/cidre/vision/categorize.py`
- Create: `tests/test_vision.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vision.py
from unittest.mock import MagicMock, patch
from cidre.vision.describe import describe_image, describe_video, describe_document, describe_markdown
from cidre.vision.categorize import parse_categories, CATEGORY_PROMPT_SUFFIX


def test_describe_image_calls_llm_with_image(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")

    mock_llm = MagicMock()
    mock_llm.generate_with_image.return_value = (
        'Description: A sunset over the ocean\n'
        'Categories: travel, landscape, sunset\n'
        'Summary: Beach sunset photo'
    )

    result = describe_image(mock_llm, img)
    mock_llm.generate_with_image.assert_called_once()
    assert "sunset" in result["description"].lower()
    assert isinstance(result["categories"], list)
    assert isinstance(result["summary"], str)


def test_describe_markdown_calls_llm():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = (
        'Description: Meeting notes about project timeline\n'
        'Categories: work, notes\n'
        'Summary: Notes from team meeting'
    )

    result = describe_markdown(mock_llm, "# Meeting\nDiscussed timeline.")
    mock_llm.generate.assert_called_once()
    assert "description" in result
    assert "categories" in result


def test_parse_categories():
    raw = "travel, landscape, sunset"
    result = parse_categories(raw)
    assert result == ["travel", "landscape", "sunset"]


def test_parse_categories_with_brackets():
    raw = "[travel, landscape]"
    result = parse_categories(raw)
    assert result == ["travel", "landscape"]


def test_parse_categories_empty():
    assert parse_categories("") == []
    assert parse_categories("none") == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_vision.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement vision/describe.py**

```python
# src/cidre/vision/__init__.py
```

```python
# src/cidre/vision/describe.py
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from cidre.vision.categorize import parse_categories

IMAGE_PROMPT = """Analyze this image and provide:
Description: A one-sentence description of what the image shows.
Categories: A comma-separated list of categories (e.g., travel, receipt, screenshot, landscape, people, food, document, pet, vehicle, building).
Summary: A brief 1-2 sentence summary.

Respond in exactly this format:
Description: <description>
Categories: <categories>
Summary: <summary>"""

DOCUMENT_PROMPT = """Analyze this document text and provide:
Description: A one-sentence description of what the document is about.
Categories: A comma-separated list of categories (e.g., receipt, invoice, insurance, legal, medical, financial, personal, work).
Summary: A brief 1-2 sentence summary.

Document text:
{text}

Respond in exactly this format:
Description: <description>
Categories: <categories>
Summary: <summary>"""

MARKDOWN_PROMPT = """Analyze this markdown content and provide:
Description: A one-sentence description of what this note is about.
Categories: A comma-separated list of categories (e.g., notes, journal, work, personal, research, project, meeting).
Summary: A brief 1-2 sentence summary.

Content:
{text}

Respond in exactly this format:
Description: <description>
Categories: <categories>
Summary: <summary>"""


def _parse_response(raw: str) -> dict:
    result = {"description": "", "categories": [], "summary": ""}
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("description:"):
            result["description"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("categories:"):
            result["categories"] = parse_categories(line.split(":", 1)[1].strip())
        elif line.lower().startswith("summary:"):
            result["summary"] = line.split(":", 1)[1].strip()
    return result


def describe_image(llm, image_path: Path) -> dict:
    raw = llm.generate_with_image(IMAGE_PROMPT, image_path)
    return _parse_response(raw)


def describe_document(llm, text: str) -> dict:
    prompt = DOCUMENT_PROMPT.format(text=text[:4000])
    raw = llm.generate(prompt)
    return _parse_response(raw)


def describe_markdown(llm, text: str) -> dict:
    prompt = MARKDOWN_PROMPT.format(text=text[:4000])
    raw = llm.generate(prompt)
    return _parse_response(raw)


def describe_video(llm, video_path: Path, frames_dir: Path | None = None) -> dict:
    """Extract key frames and describe them, then combine."""
    if frames_dir is None:
        frames_dir = Path(tempfile.mkdtemp())

    # Extract 1 frame per 10 seconds using ffmpeg
    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vf", "fps=1/10",
            "-frames:v", "10",
            str(frames_dir / "frame_%03d.jpg"),
        ],
        capture_output=True,
        check=False,
    )

    frame_descriptions = []
    for frame_path in sorted(frames_dir.glob("frame_*.jpg")):
        desc = describe_image(llm, frame_path)
        frame_descriptions.append(desc["description"])

    if not frame_descriptions:
        return {"description": "Video file (could not extract frames)", "categories": ["video"], "summary": ""}

    combined = " | ".join(frame_descriptions)
    prompt = f"""These are descriptions of frames from a video, taken every 10 seconds:
{combined}

Provide a unified description:
Description: <one sentence describing the overall video>
Categories: <comma-separated categories>
Summary: <1-2 sentence summary>"""

    raw = llm.generate(prompt)
    return _parse_response(raw)
```

- [ ] **Step 4: Implement vision/categorize.py**

```python
# src/cidre/vision/categorize.py
from __future__ import annotations

import re

CATEGORY_PROMPT_SUFFIX = (
    "Categories: A comma-separated list of categories."
)


def parse_categories(raw: str) -> list[str]:
    if not raw or raw.lower() in ("none", "n/a", "[]"):
        return []
    cleaned = raw.strip().strip("[]")
    cats = [c.strip().lower() for c in re.split(r"[,;]", cleaned)]
    return [c for c in cats if c and c not in ("none", "n/a")]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_vision.py -v
```

Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/cidre/vision/ tests/test_vision.py
git commit -m "feat: add vision module for image, video, document, and markdown description"
```

---

## Task 7: Indexing Pipeline

**Agent:** indexer-agent
**Files:**
- Create: `src/cidre/indexer/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline.py
from pathlib import Path
from unittest.mock import MagicMock, patch
from cidre.indexer.pipeline import IndexingPipeline
from cidre.db import init_db, get_item_by_path


def test_pipeline_indexes_markdown(tmp_path):
    db_path = tmp_path / "cidre.db"
    conn = init_db(db_path, embedding_dimensions=768)

    md_file = tmp_path / "docs" / "notes.md"
    md_file.parent.mkdir()
    md_file.write_text("# Meeting Notes\nDiscussed project timeline and deadlines.")

    mock_llm = MagicMock()
    mock_llm.generate.return_value = (
        "Description: Meeting notes about project timeline\n"
        "Categories: work, notes, meeting\n"
        "Summary: Team meeting notes discussing deadlines"
    )

    mock_embedder = MagicMock()
    mock_embedder.dimensions = 768
    mock_embedder.embed.return_value = [[0.1] * 768]

    pipeline = IndexingPipeline(conn=conn, llm=mock_llm, embedder=mock_embedder)
    pipeline.index_file(md_file, file_type="markdown")

    item = get_item_by_path(conn, str(md_file))
    assert item is not None
    assert "meeting" in item.ai_description.lower()
    assert "work" in item.categories
    conn.close()


def test_pipeline_skips_unchanged_file(tmp_path):
    db_path = tmp_path / "cidre.db"
    conn = init_db(db_path, embedding_dimensions=768)

    md_file = tmp_path / "notes.md"
    md_file.write_text("# Notes")

    mock_llm = MagicMock()
    mock_llm.generate.return_value = (
        "Description: Simple notes\nCategories: notes\nSummary: Notes file"
    )
    mock_embedder = MagicMock()
    mock_embedder.dimensions = 768
    mock_embedder.embed.return_value = [[0.1] * 768]

    pipeline = IndexingPipeline(conn=conn, llm=mock_llm, embedder=mock_embedder)
    pipeline.index_file(md_file, file_type="markdown")
    pipeline.index_file(md_file, file_type="markdown")

    # LLM should only be called once — second call skipped due to same hash
    assert mock_llm.generate.call_count == 1
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement pipeline.py**

```python
# src/cidre/indexer/pipeline.py
from __future__ import annotations

import sqlite3
from pathlib import Path

import pdfplumber

from cidre.db import insert_item, get_item_by_path, ItemRow
from cidre.indexer.scanner import file_hash
from cidre.vision.describe import describe_image, describe_document, describe_markdown, describe_video


class IndexingPipeline:
    def __init__(self, conn: sqlite3.Connection, llm, embedder):
        self._conn = conn
        self._llm = llm
        self._embedder = embedder

    def index_file(self, path: Path, file_type: str, source: str = "filesystem") -> bool:
        """Index a single file. Returns True if indexed, False if skipped."""
        current_hash = file_hash(path)

        existing = get_item_by_path(self._conn, str(path))
        if existing and existing.file_hash == current_hash:
            return False

        stat = path.stat()
        description = self._describe(path, file_type)

        embedding = self._embedder.embed([description["description"]])[0]

        insert_item(self._conn, ItemRow(
            file_path=str(path),
            file_hash=current_hash,
            file_type=file_type,
            file_size=stat.st_size,
            modified_at=__import__("datetime").datetime.fromtimestamp(stat.st_mtime).isoformat(),
            ai_description=description["description"],
            categories=description["categories"],
            summary=description["summary"],
            embedding=embedding,
            source=source,
        ))
        return True

    def index_batch(self, files: list[tuple[Path, str]], source: str = "filesystem") -> int:
        """Index a batch of files. Returns count of newly indexed files."""
        count = 0
        for path, file_type in files:
            try:
                if self.index_file(path, file_type, source):
                    count += 1
            except Exception as e:
                print(f"Error indexing {path}: {e}")
        return count

    def _describe(self, path: Path, file_type: str) -> dict:
        if file_type == "photo":
            return describe_image(self._llm, path)
        elif file_type == "video":
            return describe_video(self._llm, path)
        elif file_type == "document":
            text = self._extract_pdf_text(path)
            return describe_document(self._llm, text)
        elif file_type == "markdown":
            text = path.read_text(errors="replace")
            return describe_markdown(self._llm, text)
        else:
            return {"description": f"{file_type} file", "categories": [], "summary": ""}

    def _extract_pdf_text(self, path: Path) -> str:
        try:
            with pdfplumber.open(path) as pdf:
                pages = []
                for page in pdf.pages[:10]:  # limit to first 10 pages
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                return "\n\n".join(pages)
        except Exception:
            return ""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/cidre/indexer/pipeline.py tests/test_pipeline.py
git commit -m "feat: add indexing pipeline with incremental hashing and multi-type support"
```

---

## Task 8: Apple Photos Importer

**Agent:** indexer-agent
**Files:**
- Create: `src/cidre/indexer/photos.py`
- Create: `tests/test_photos.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_photos.py
from unittest.mock import patch, MagicMock
from cidre.indexer.photos import list_apple_photos, PhotoInfo


def test_photo_info_dataclass():
    info = PhotoInfo(
        uuid="ABC-123",
        original_path="/path/to/photo.jpg",
        date="2024-06-15T10:30:00",
        title="Beach sunset",
    )
    assert info.uuid == "ABC-123"
    assert info.original_path == "/path/to/photo.jpg"


@patch("cidre.indexer.photos.osxphotos.PhotosDB")
def test_list_apple_photos_returns_photo_infos(mock_db_cls):
    mock_photo = MagicMock()
    mock_photo.uuid = "uuid-1"
    mock_photo.path = "/path/to/IMG_001.jpg"
    mock_photo.date = MagicMock()
    mock_photo.date.isoformat.return_value = "2024-06-15T10:30:00"
    mock_photo.title = "My Photo"
    mock_photo.ismissing = False

    mock_db = MagicMock()
    mock_db.photos.return_value = [mock_photo]
    mock_db_cls.return_value = mock_db

    photos = list_apple_photos()
    assert len(photos) == 1
    assert photos[0].uuid == "uuid-1"
    assert photos[0].original_path == "/path/to/IMG_001.jpg"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_photos.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement photos.py**

```python
# src/cidre/indexer/photos.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import osxphotos


@dataclass
class PhotoInfo:
    uuid: str
    original_path: str
    date: str
    title: str | None


def list_apple_photos(
    limit: int | None = None,
) -> list[PhotoInfo]:
    """List photos from Apple Photos library. Read-only — no exports."""
    db = osxphotos.PhotosDB()
    photos = db.photos()

    results = []
    for photo in photos:
        if photo.ismissing:
            continue
        if photo.path is None:
            continue

        info = PhotoInfo(
            uuid=photo.uuid,
            original_path=photo.path,
            date=photo.date.isoformat() if photo.date else "",
            title=photo.title,
        )
        results.append(info)

        if limit and len(results) >= limit:
            break

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_photos.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/cidre/indexer/photos.py tests/test_photos.py
git commit -m "feat: add Apple Photos importer via osxphotos (read-only)"
```

---

## Task 9: Search Engine

**Agent:** search-agent
**Files:**
- Create: `src/cidre/search/__init__.py`
- Create: `src/cidre/search/engine.py`
- Create: `src/cidre/search/query.py`
- Create: `tests/test_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_search.py
from datetime import datetime
from cidre.search.query import parse_query, QueryIntent
from cidre.search.engine import SearchEngine
from cidre.db import init_db, insert_item, ItemRow
from unittest.mock import MagicMock


def test_parse_simple_query():
    intent = parse_query("sunset photos")
    assert intent.text == "sunset photos"
    assert intent.file_type is None
    assert intent.date_start is None


def test_parse_query_with_photo_hint():
    intent = parse_query("photos of mountains")
    assert intent.text == "mountains"
    assert intent.file_type == "photo"


def test_parse_query_with_date():
    intent = parse_query("invoices from 2025")
    assert intent.text == "invoices"
    assert intent.date_start is not None
    assert intent.date_start.year == 2025
    assert intent.date_end.year == 2025


def test_parse_query_last_year():
    intent = parse_query("receipts from last year")
    now = datetime.now()
    assert intent.date_start.year == now.year - 1


def test_search_engine_returns_ranked_results(tmp_path):
    db_path = tmp_path / "cidre.db"
    conn = init_db(db_path, embedding_dimensions=4)

    insert_item(conn, ItemRow(
        file_path="/test/sunset.jpg",
        file_hash="h1",
        file_type="photo",
        file_size=1000,
        modified_at="2024-06-15T00:00:00",
        ai_description="A golden sunset over the ocean",
        categories=["landscape", "travel"],
        summary="Sunset photo",
        embedding=[0.9, 0.1, 0.0, 0.0],
        source="filesystem",
    ))
    insert_item(conn, ItemRow(
        file_path="/test/invoice.pdf",
        file_hash="h2",
        file_type="document",
        file_size=2000,
        modified_at="2025-01-10T00:00:00",
        ai_description="Health insurance invoice from Niva Bupa",
        categories=["receipt", "insurance"],
        summary="Insurance invoice",
        embedding=[0.0, 0.0, 0.9, 0.1],
        source="filesystem",
    ))

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.85, 0.15, 0.0, 0.0]]

    engine = SearchEngine(conn=conn, embedder=mock_embedder)
    results = engine.search("sunset photos")

    assert len(results) >= 1
    assert results[0].file_path == "/test/sunset.jpg"
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_search.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement search/query.py**

```python
# src/cidre/search/__init__.py
```

```python
# src/cidre/search/query.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class QueryIntent:
    text: str
    file_type: str | None = None
    date_start: datetime | None = None
    date_end: datetime | None = None
    category: str | None = None


def parse_query(raw: str) -> QueryIntent:
    text = raw.strip()
    file_type = None
    date_start = None
    date_end = None

    # Detect type hints
    photo_patterns = [r"^photos?\s+of\s+", r"^images?\s+of\s+", r"^pictures?\s+of\s+"]
    for pattern in photo_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            file_type = "photo"
            text = text[match.end():]
            break

    doc_patterns = [r"^documents?\s+about\s+", r"^docs?\s+about\s+"]
    for pattern in doc_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            file_type = "document"
            text = text[match.end():]
            break

    # Detect year references: "from 2025"
    year_match = re.search(r"\bfrom\s+(\d{4})\b", text, re.IGNORECASE)
    if year_match:
        year = int(year_match.group(1))
        date_start = datetime(year, 1, 1)
        date_end = datetime(year, 12, 31, 23, 59, 59)
        text = text[:year_match.start()].strip()

    # Detect "last year"
    last_year_match = re.search(r"\bfrom\s+last\s+year\b", text, re.IGNORECASE)
    if last_year_match:
        year = datetime.now().year - 1
        date_start = datetime(year, 1, 1)
        date_end = datetime(year, 12, 31, 23, 59, 59)
        text = text[:last_year_match.start()].strip()

    # Detect "this month"
    this_month_match = re.search(r"\bfrom\s+this\s+month\b", text, re.IGNORECASE)
    if this_month_match:
        now = datetime.now()
        date_start = datetime(now.year, now.month, 1)
        date_end = now
        text = text[:this_month_match.start()].strip()

    # Detect Q1-Q4 references: "from Q1 2025"
    quarter_match = re.search(r"\bfrom\s+Q([1-4])\s+(\d{4})\b", text, re.IGNORECASE)
    if quarter_match:
        q = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        start_month = (q - 1) * 3 + 1
        end_month = start_month + 2
        date_start = datetime(year, start_month, 1)
        if end_month == 12:
            date_end = datetime(year, 12, 31, 23, 59, 59)
        else:
            date_end = datetime(year, end_month + 1, 1)
        text = text[:quarter_match.start()].strip()

    return QueryIntent(
        text=text,
        file_type=file_type,
        date_start=date_start,
        date_end=date_end,
    )
```

- [ ] **Step 4: Implement search/engine.py**

```python
# src/cidre/search/engine.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from cidre.db import search_by_vector, get_items_by_rowids, ItemRow
from cidre.search.query import parse_query, QueryIntent


@dataclass
class SearchResult:
    file_path: str
    file_type: str
    ai_description: str
    categories: list[str]
    summary: str
    file_size: int
    modified_at: str
    score: float
    source: str


class SearchEngine:
    def __init__(self, conn: sqlite3.Connection, embedder):
        self._conn = conn
        self._embedder = embedder

    def search(
        self,
        query: str,
        k: int = 20,
        file_type: str | None = None,
        category: str | None = None,
    ) -> list[SearchResult]:
        intent = parse_query(query)

        if file_type:
            intent.file_type = file_type
        if category:
            intent.category = category

        query_embedding = self._embedder.embed([intent.text])[0]
        vector_results = search_by_vector(self._conn, query_embedding, k=k * 2)

        if not vector_results:
            return []

        rowids = [r[0] for r in vector_results]
        distances = {r[0]: r[1] for r in vector_results}
        items = get_items_by_rowids(self._conn, rowids)

        results = []
        for item in items:
            if intent.file_type and item.file_type != intent.file_type:
                continue
            if intent.date_start and item.modified_at < intent.date_start.isoformat():
                continue
            if intent.date_end and item.modified_at > intent.date_end.isoformat():
                continue
            if intent.category and intent.category not in item.categories:
                continue

            distance = distances.get(item.rowid, 1.0)
            score = max(0.0, 1.0 - distance)

            results.append(SearchResult(
                file_path=item.file_path,
                file_type=item.file_type,
                ai_description=item.ai_description,
                categories=item.categories,
                summary=item.summary,
                file_size=item.file_size,
                modified_at=item.modified_at,
                score=round(score, 4),
                source=item.source,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_search.py -v
```

Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/cidre/search/ tests/test_search.py
git commit -m "feat: add search engine with query parsing, vector search, and metadata filtering"
```

---

## Task 10: Full CLI Implementation

**Agent:** core-architect
**Files:**
- Modify: `src/cidre/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli.py
from typer.testing import CliRunner
from cidre.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "cidre" in result.output


def test_status_no_config(tmp_path, monkeypatch):
    monkeypatch.setattr("cidre.cli.CIDRE_HOME", tmp_path / ".cidre")
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "not initialized" in result.output.lower() or "cidre init" in result.output.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL (status command doesn't exist yet)

- [ ] **Step 3: Implement full CLI**

```python
# src/cidre/cli.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cidre import __version__
from cidre.config import (
    CIDRE_HOME, CidreConfig, EMBEDDING_MODELS, LLM_MODELS,
    load_config, save_config,
)

app = typer.Typer(
    name="cidre",
    help="Local-first search intelligence for macOS",
    no_args_is_help=True,
)
console = Console()


def _get_config() -> CidreConfig | None:
    config_path = CIDRE_HOME / "config.toml"
    if not config_path.exists():
        return None
    return load_config(config_path)


def _ensure_initialized() -> CidreConfig:
    config = _get_config()
    if config is None:
        console.print("[red]Cidre is not initialized. Run: cidre init[/red]")
        raise typer.Exit(1)
    return config


def _get_unified_memory_gb() -> float:
    try:
        output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(output.strip()) / (1024 ** 3)
    except Exception:
        return 0


@app.command()
def version():
    """Show Cidre version."""
    typer.echo(f"cidre {__version__}")


@app.command()
def init():
    """Initialize Cidre — choose models and set up config."""
    if (CIDRE_HOME / "config.toml").exists():
        if not typer.confirm("Cidre is already initialized. Re-initialize?"):
            raise typer.Exit()

    console.print("\n[bold]Welcome to Cidre[/bold] — local-first search intelligence\n")

    # Check Ollama
    try:
        subprocess.check_output(["ollama", "--version"], stderr=subprocess.STDOUT)
        console.print("[green]Ollama found.[/green]")
    except FileNotFoundError:
        console.print("[red]Ollama not found. Install it from https://ollama.com[/red]")
        raise typer.Exit(1)

    # Detect memory and recommend LLM
    mem_gb = _get_unified_memory_gb()
    console.print(f"Detected [bold]{mem_gb:.0f} GB[/bold] unified memory.\n")

    console.print("[bold]Choose LLM model:[/bold]")
    llm_choices = list(LLM_MODELS.keys())
    for i, model in enumerate(llm_choices, 1):
        info = LLM_MODELS[model]
        rec = " (recommended)" if (mem_gb >= 24 and "26B" in info["label"]) else ""
        rec = " (recommended)" if (mem_gb < 24 and "E4B" in info["label"]) else rec
        console.print(f"  {i}. {model} — {info['label']}, ~{info['memory_gb']}GB{rec}")

    llm_idx = typer.prompt("Select", type=int, default=1) - 1
    llm_model = llm_choices[max(0, min(llm_idx, len(llm_choices) - 1))]

    # Choose embedding model
    console.print("\n[bold]Choose embedding model:[/bold]")
    emb_choices = list(EMBEDDING_MODELS.keys())
    for i, model in enumerate(emb_choices, 1):
        info = EMBEDDING_MODELS[model]
        local = "local" if info["provider"] == "ollama" else "[yellow]cloud — data leaves machine[/yellow]"
        console.print(f"  {i}. {model} — {info['dimensions']}d, {local}")

    emb_idx = typer.prompt("Select", type=int, default=1) - 1
    emb_model = emb_choices[max(0, min(emb_idx, len(emb_choices) - 1))]
    emb_info = EMBEDDING_MODELS[emb_model]

    api_key = ""
    if emb_info["provider"] == "openai":
        api_key = typer.prompt("OpenAI API key")

    config = CidreConfig(
        llm_model=llm_model,
        embedding_provider=emb_info["provider"],
        embedding_model=emb_model,
        embedding_dimensions=emb_info["dimensions"],
    )

    CIDRE_HOME.mkdir(parents=True, exist_ok=True)
    (CIDRE_HOME / "logs").mkdir(exist_ok=True)
    save_config(config, CIDRE_HOME / "config.toml")

    console.print(f"\n[green]Cidre initialized![/green]")
    console.print(f"  LLM: {llm_model}")
    console.print(f"  Embedding: {emb_model} ({emb_info['dimensions']}d)")
    console.print(f"\nNext: [bold]cidre add ~/Documents[/bold]")


@app.command()
def add(
    path: str = typer.Argument(None, help="Directory path to add"),
    photos: bool = typer.Option(False, "--photos", help="Add Apple Photos library"),
):
    """Add a directory or Apple Photos to the index sources."""
    config = _ensure_initialized()

    if photos:
        config.photos_enabled = True
        save_config(config, CIDRE_HOME / "config.toml")
        console.print("[green]Apple Photos library added to sources.[/green]")
        return

    if path is None:
        console.print("[red]Provide a directory path or use --photos[/red]")
        raise typer.Exit(1)

    resolved = str(Path(path).expanduser().resolve())
    if not Path(resolved).is_dir():
        console.print(f"[red]Not a directory: {resolved}[/red]")
        raise typer.Exit(1)

    if resolved not in config.sources_watched:
        config.sources_watched.append(resolved)
        save_config(config, CIDRE_HOME / "config.toml")
        console.print(f"[green]Added: {resolved}[/green]")
    else:
        console.print(f"Already added: {resolved}")


@app.command()
def exclude(patterns: list[str] = typer.Argument(..., help="Glob patterns to exclude")):
    """Add exclude patterns."""
    config = _ensure_initialized()
    added = []
    for p in patterns:
        if p not in config.exclude_patterns:
            config.exclude_patterns.append(p)
            added.append(p)
    if added:
        save_config(config, CIDRE_HOME / "config.toml")
        console.print(f"[green]Added exclude patterns: {', '.join(added)}[/green]")
    else:
        console.print("All patterns already excluded.")


@app.command()
def status():
    """Show index status and statistics."""
    config = _get_config()
    if config is None:
        console.print("Cidre is not initialized. Run: [bold]cidre init[/bold]")
        return

    console.print(f"[bold]Cidre Status[/bold]\n")
    console.print(f"LLM: {config.llm_model}")
    console.print(f"Embedding: {config.embedding_model} ({config.embedding_dimensions}d)")
    console.print(f"Photos: {'enabled' if config.photos_enabled else 'disabled'}")
    console.print(f"Watched dirs: {len(config.sources_watched)}")
    for d in config.sources_watched:
        console.print(f"  - {d}")

    db_path = CIDRE_HOME / "cidre.db"
    if db_path.exists():
        from cidre.db import init_db, get_index_stats
        conn = init_db(db_path, config.embedding_dimensions)
        stats = get_index_stats(conn)
        conn.close()

        console.print(f"\n[bold]Index:[/bold]")
        table = Table()
        table.add_column("Type")
        table.add_column("Count", justify="right")
        for key, val in stats.items():
            table.add_row(key, str(val))
        console.print(table)
    else:
        console.print("\n[dim]No index yet. Run: cidre index[/dim]")


@app.command()
def index(
    only: str = typer.Option(None, "--only", help="Only index: photos, docs, markdown, videos"),
):
    """Run indexing on all configured sources."""
    config = _ensure_initialized()

    from cidre.db import init_db
    from cidre.indexer.scanner import scan_directory
    from cidre.indexer.pipeline import IndexingPipeline
    from cidre.providers.base import get_provider
    from cidre.providers.ollama import OllamaLLM

    conn = init_db(CIDRE_HOME / "cidre.db", config.embedding_dimensions)
    embedder = get_provider(config.embedding_provider, config.embedding_model)
    llm = OllamaLLM(model=config.llm_model)
    pipeline = IndexingPipeline(conn=conn, llm=llm, embedder=embedder)

    total = 0
    type_filter = only if only else None

    for dir_path in config.sources_watched:
        resolved = Path(dir_path).expanduser()
        if not resolved.is_dir():
            console.print(f"[yellow]Skipping (not found): {dir_path}[/yellow]")
            continue

        console.print(f"Scanning {dir_path}...")
        scanned = scan_directory(resolved, config.exclude_patterns)

        if type_filter:
            type_map = {"photos": "photo", "docs": "document", "markdown": "markdown", "videos": "video"}
            ft = type_map.get(type_filter, type_filter)
            scanned = [s for s in scanned if s.file_type == ft]

        files = [(s.path, s.file_type) for s in scanned]
        console.print(f"  Found {len(files)} files to process...")
        count = pipeline.index_batch(files)
        total += count
        console.print(f"  Indexed {count} new files.")

    if config.photos_enabled:
        console.print("Importing Apple Photos...")
        from cidre.indexer.photos import list_apple_photos
        photos = list_apple_photos()
        photo_files = [(Path(p.original_path), "photo") for p in photos if p.original_path]
        count = pipeline.index_batch(photo_files, source="apple_photos")
        total += count
        console.print(f"  Indexed {count} new photos from Apple Photos.")

    conn.close()
    console.print(f"\n[green]Done! Indexed {total} new items total.[/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    type: str = typer.Option(None, "--type", "-t", help="Filter by type: photo, document, markdown, video"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by category"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
):
    """Search your indexed files."""
    config = _ensure_initialized()

    from cidre.db import init_db
    from cidre.search.engine import SearchEngine
    from cidre.providers.base import get_provider

    db_path = CIDRE_HOME / "cidre.db"
    if not db_path.exists():
        console.print("[red]No index found. Run: cidre index[/red]")
        raise typer.Exit(1)

    conn = init_db(db_path, config.embedding_dimensions)
    embedder = get_provider(config.embedding_provider, config.embedding_model)
    engine = SearchEngine(conn=conn, embedder=embedder)

    results = engine.search(query, k=limit, file_type=type, category=category)
    conn.close()

    if not results:
        console.print("[dim]No results found.[/dim]")
        return

    for i, r in enumerate(results, 1):
        cats = ", ".join(r.categories) if r.categories else ""
        size_mb = r.file_size / (1024 * 1024)
        date = r.modified_at[:10] if r.modified_at else ""

        console.print(f"\n[bold]{i}. {Path(r.file_path).name}[/bold]  [{cats}]")
        console.print(f'   "{r.ai_description}"')
        console.print(f"   {r.file_path} · {size_mb:.1f} MB · {date}")
        console.print(f"   Score: {r.score}")


@app.command()
def categories():
    """List all categories in the index."""
    config = _ensure_initialized()

    from cidre.db import init_db
    import json

    db_path = CIDRE_HOME / "cidre.db"
    if not db_path.exists():
        console.print("[red]No index found. Run: cidre index[/red]")
        raise typer.Exit(1)

    conn = init_db(db_path, config.embedding_dimensions)
    rows = conn.execute("SELECT categories FROM items").fetchall()
    conn.close()

    cat_counts: dict[str, int] = {}
    for row in rows:
        for cat in json.loads(row[0]):
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    if not cat_counts:
        console.print("[dim]No categories found.[/dim]")
        return

    table = Table(title="Categories")
    table.add_column("Category")
    table.add_column("Count", justify="right")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        table.add_row(cat, str(count))
    console.print(table)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cli.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/cidre/cli.py tests/test_cli.py
git commit -m "feat: implement full CLI with init, add, exclude, index, search, status, categories"
```

---

## Task 11: File Watcher Daemon

**Agent:** indexer-agent
**Files:**
- Create: `src/cidre/indexer/daemon.py`
- Create: `tests/test_daemon.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_daemon.py
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from cidre.indexer.daemon import CidreEventHandler, generate_launchd_plist


def test_event_handler_queues_new_file(tmp_path):
    mock_callback = MagicMock()
    handler = CidreEventHandler(
        callback=mock_callback,
        exclude_patterns=["*.py"],
    )

    class FakeEvent:
        src_path = str(tmp_path / "photo.jpg")
        is_directory = False

    handler.on_created(FakeEvent())
    assert mock_callback.call_count == 1
    assert "photo.jpg" in mock_callback.call_args[0][0]


def test_event_handler_ignores_excluded_file(tmp_path):
    mock_callback = MagicMock()
    handler = CidreEventHandler(
        callback=mock_callback,
        exclude_patterns=["*.py"],
    )

    class FakeEvent:
        src_path = str(tmp_path / "code.py")
        is_directory = False

    handler.on_created(FakeEvent())
    assert mock_callback.call_count == 0


def test_generate_launchd_plist():
    plist = generate_launchd_plist()
    assert "com.cidre.watcher" in plist
    assert "cidre" in plist
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_daemon.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement daemon.py**

```python
# src/cidre/indexer/daemon.py
from __future__ import annotations

import sys
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from cidre.indexer.scanner import classify_file, should_exclude


class CidreEventHandler(FileSystemEventHandler):
    def __init__(self, callback, exclude_patterns: list[str] | None = None):
        super().__init__()
        self._callback = callback
        self._exclude = exclude_patterns or []

    def _should_process(self, path_str: str) -> bool:
        path = Path(path_str)
        if should_exclude(path, self._exclude):
            return False
        return classify_file(path) is not None

    def on_created(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            self._callback(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            self._callback(event.src_path)


def start_watcher(directories: list[str], callback, exclude_patterns: list[str] | None = None):
    """Start watching directories for file changes. Blocking call."""
    handler = CidreEventHandler(callback=callback, exclude_patterns=exclude_patterns)
    observer = Observer()

    for dir_path in directories:
        resolved = Path(dir_path).expanduser().resolve()
        if resolved.is_dir():
            observer.schedule(handler, str(resolved), recursive=True)

    observer.start()
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def generate_launchd_plist() -> str:
    python_path = sys.executable
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.cidre.watcher</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>cidre.cli</string>
        <string>watch</string>
        <string>start</string>
        <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{Path.home()}/.cidre/logs/daemon.log</string>
    <key>StandardErrorPath</key>
    <string>{Path.home()}/.cidre/logs/daemon.log</string>
</dict>
</plist>"""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_daemon.py -v
```

Expected: 3 passed

- [ ] **Step 5: Add watch commands to CLI**

Add to `src/cidre/cli.py` before the final line:

```python
@app.command("watch")
def watch(
    action: str = typer.Argument(..., help="start or stop"),
    foreground: bool = typer.Option(False, "--foreground", help="Run in foreground (for launchd)"),
):
    """Start or stop the background file watcher."""
    config = _ensure_initialized()

    plist_path = Path.home() / "Library/LaunchAgents/com.cidre.watcher.plist"

    if action == "start":
        if foreground:
            from cidre.db import init_db
            from cidre.indexer.daemon import start_watcher
            from cidre.indexer.pipeline import IndexingPipeline
            from cidre.indexer.scanner import classify_file
            from cidre.providers.base import get_provider
            from cidre.providers.ollama import OllamaLLM

            conn = init_db(CIDRE_HOME / "cidre.db", config.embedding_dimensions)
            embedder = get_provider(config.embedding_provider, config.embedding_model)
            llm = OllamaLLM(model=config.llm_model)
            pipeline = IndexingPipeline(conn=conn, llm=llm, embedder=embedder)

            def on_file_change(path_str: str):
                p = Path(path_str)
                ft = classify_file(p)
                if ft:
                    pipeline.index_file(p, ft)

            console.print("Watching for changes... (Ctrl+C to stop)")
            start_watcher(config.sources_watched, on_file_change, config.exclude_patterns)
        else:
            from cidre.indexer.daemon import generate_launchd_plist
            plist_content = generate_launchd_plist()
            plist_path.parent.mkdir(parents=True, exist_ok=True)
            plist_path.write_text(plist_content)
            subprocess.run(["launchctl", "load", str(plist_path)])
            console.print("[green]Watcher daemon started.[/green]")

    elif action == "stop":
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)])
            plist_path.unlink()
            console.print("[green]Watcher daemon stopped.[/green]")
        else:
            console.print("[dim]No daemon running.[/dim]")
    else:
        console.print("[red]Use: cidre watch start|stop[/red]")
```

- [ ] **Step 6: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add src/cidre/indexer/daemon.py src/cidre/cli.py tests/test_daemon.py
git commit -m "feat: add file watcher daemon with launchd integration"
```

---

## Task 12: Landing Page

**Agent:** site-agent
**Files:**
- Create: `site/index.html`
- Create: `site/style.css`

- [ ] **Step 1: Create index.html**

```html
<!-- site/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cidre — Local-first search intelligence for macOS</title>
    <link rel="stylesheet" href="style.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <main>
        <section class="hero">
            <h1>cidre</h1>
            <p class="tagline">Local-first search intelligence for macOS</p>

            <div class="terminal">
                <div class="terminal-bar">
                    <span class="dot red"></span>
                    <span class="dot yellow"></span>
                    <span class="dot green"></span>
                </div>
                <div class="terminal-body">
                    <p class="prompt">$ cidre search "sunset photos"</p>
                    <p class="blank"></p>
                    <p class="result"><span class="num">1.</span> <span class="file">IMG_7202.jpg</span> <span class="tags">[travel, landscape]</span></p>
                    <p class="desc">&nbsp;&nbsp;&nbsp;"Golden sunset over the ocean from a beach"</p>
                    <p class="meta">&nbsp;&nbsp;&nbsp;~/Documents/IMG_7202.jpg &middot; 2.4 MB &middot; 2024-06-15</p>
                    <p class="score">&nbsp;&nbsp;&nbsp;Score: 0.92</p>
                    <p class="blank"></p>
                    <p class="result"><span class="num">2.</span> <span class="file">vacation_23.png</span> <span class="tags">[landscape]</span></p>
                    <p class="desc">&nbsp;&nbsp;&nbsp;"Mountain sunset with orange and purple sky"</p>
                    <p class="meta">&nbsp;&nbsp;&nbsp;~/Pictures/vacation_23.png &middot; 1.8 MB &middot; 2023-08-20</p>
                    <p class="score">&nbsp;&nbsp;&nbsp;Score: 0.87</p>
                </div>
            </div>

            <p class="privacy">Your files. Your photos. Your machine. Nothing leaves the laptop.</p>
        </section>

        <section class="how-it-works">
            <h2>How it works</h2>
            <div class="steps">
                <div class="step">
                    <div class="step-num">1</div>
                    <h3>Index</h3>
                    <p>Gemma 4 describes and categorizes your photos, documents, videos, and notes.</p>
                </div>
                <div class="step">
                    <div class="step-num">2</div>
                    <h3>Embed</h3>
                    <p>Descriptions are converted to vectors and stored locally in SQLite.</p>
                </div>
                <div class="step">
                    <div class="step-num">3</div>
                    <h3>Search</h3>
                    <p>Queries are matched instantly via semantic similarity. No LLM call needed.</p>
                </div>
            </div>
        </section>

        <section class="features">
            <h2>Features</h2>
            <ul class="feature-grid">
                <li>Photos, docs, videos, notes</li>
                <li>Apple Photos integration</li>
                <li>Auto-categorization</li>
                <li>Background file watching</li>
                <li>Choose your embedding model</li>
                <li>100% local &mdash; powered by Ollama</li>
            </ul>
        </section>

        <section class="install">
            <h2>Get started</h2>
            <div class="terminal small">
                <div class="terminal-bar">
                    <span class="dot red"></span>
                    <span class="dot yellow"></span>
                    <span class="dot green"></span>
                </div>
                <div class="terminal-body">
                    <p class="prompt">$ pip install cidre</p>
                    <p class="prompt">$ cidre init</p>
                    <p class="prompt">$ cidre add ~/Documents</p>
                    <p class="prompt">$ cidre add --photos</p>
                    <p class="prompt">$ cidre index</p>
                    <p class="prompt">$ cidre search "sunset photos"</p>
                </div>
            </div>
            <p class="github"><a href="https://github.com/cidre-search/cidre">View on GitHub</a></p>
        </section>

        <footer>
            <p>MIT License &middot; Built for macOS &middot; Powered by Gemma 4 + Ollama</p>
        </footer>
    </main>
</body>
</html>
```

- [ ] **Step 2: Create style.css**

```css
/* site/style.css */
:root {
    --bg: #0a0a0a;
    --fg: #e0e0e0;
    --dim: #666;
    --accent: #c8a86e;
    --terminal-bg: #141414;
    --terminal-border: #2a2a2a;
    --green: #4ec990;
    --red: #e05252;
    --yellow: #e0c252;
    --blue: #5c9ee0;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    background: var(--bg);
    color: var(--fg);
    font-family: 'Inter', -apple-system, sans-serif;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

main {
    max-width: 720px;
    margin: 0 auto;
    padding: 80px 24px;
}

/* Hero */
.hero { text-align: center; margin-bottom: 100px; }

h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
}

.tagline {
    color: var(--dim);
    font-size: 1.15rem;
    margin-bottom: 48px;
}

.privacy {
    color: var(--accent);
    font-size: 0.95rem;
    margin-top: 32px;
    font-weight: 500;
}

/* Terminal */
.terminal {
    background: var(--terminal-bg);
    border: 1px solid var(--terminal-border);
    border-radius: 10px;
    overflow: hidden;
    text-align: left;
    margin: 0 auto;
    max-width: 600px;
}

.terminal.small { max-width: 480px; margin: 0 auto; }

.terminal-bar {
    background: #1a1a1a;
    padding: 10px 14px;
    display: flex;
    gap: 6px;
}

.dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}
.dot.red { background: var(--red); }
.dot.yellow { background: var(--yellow); }
.dot.green { background: var(--green); }

.terminal-body {
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.7;
}

.prompt { color: var(--green); }
.prompt::before { content: none; }
.result .num { color: var(--dim); }
.result .file { color: #fff; font-weight: 700; }
.result .tags { color: var(--accent); }
.desc { color: var(--fg); opacity: 0.8; }
.meta { color: var(--dim); font-size: 0.75rem; }
.score { color: var(--blue); font-size: 0.75rem; }
.blank { height: 0.8em; }

/* Sections */
h2 {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 32px;
    text-align: center;
}

.how-it-works { margin-bottom: 80px; }

.steps {
    display: flex;
    gap: 32px;
    justify-content: center;
}

.step {
    flex: 1;
    text-align: center;
    max-width: 200px;
}

.step-num {
    width: 36px;
    height: 36px;
    background: var(--accent);
    color: var(--bg);
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 12px;
}

.step h3 {
    font-size: 1rem;
    margin-bottom: 8px;
}

.step p {
    color: var(--dim);
    font-size: 0.85rem;
}

/* Features */
.features { margin-bottom: 80px; }

.feature-grid {
    list-style: none;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px 32px;
    max-width: 480px;
    margin: 0 auto;
}

.feature-grid li {
    color: var(--fg);
    font-size: 0.9rem;
    padding-left: 20px;
    position: relative;
}

.feature-grid li::before {
    content: "~";
    position: absolute;
    left: 0;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
}

/* Install */
.install {
    text-align: center;
    margin-bottom: 80px;
}

.github {
    margin-top: 24px;
}

.github a {
    color: var(--accent);
    text-decoration: none;
    border-bottom: 1px solid var(--terminal-border);
    padding-bottom: 2px;
    transition: border-color 0.2s;
}

.github a:hover {
    border-color: var(--accent);
}

/* Footer */
footer {
    text-align: center;
    color: var(--dim);
    font-size: 0.8rem;
    padding-top: 40px;
    border-top: 1px solid var(--terminal-border);
}

/* Responsive */
@media (max-width: 600px) {
    h1 { font-size: 2.5rem; }
    .steps { flex-direction: column; align-items: center; }
    .feature-grid { grid-template-columns: 1fr; }
    main { padding: 40px 16px; }
}
```

- [ ] **Step 3: Test locally**

```bash
open site/index.html
```

Verify: dark theme, terminal mockup renders correctly, responsive on resize.

- [ ] **Step 4: Commit**

```bash
git add site/
git commit -m "feat: add cidre.dev landing page — minimalist dark theme"
```

---

## Task 13: Integration Test — End-to-End

**Agent:** core-architect
**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration test using mocked Ollama responses."""
from pathlib import Path
from unittest.mock import MagicMock

from cidre.config import CidreConfig, save_config
from cidre.db import init_db, get_index_stats
from cidre.indexer.scanner import scan_directory
from cidre.indexer.pipeline import IndexingPipeline
from cidre.search.engine import SearchEngine


def test_full_index_and_search_flow(tmp_path):
    # Setup config
    cidre_home = tmp_path / ".cidre"
    cidre_home.mkdir()
    config = CidreConfig(embedding_dimensions=32)
    save_config(config, cidre_home / "config.toml")

    # Create test files
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "meeting.md").write_text("# Q1 Planning\nBudget review and timeline.")
    (docs / "recipe.md").write_text("# Pasta Recipe\nBoil water, add pasta, cook 10 mins.")
    (docs / "code.py").write_text("print('hello')")  # should be excluded

    # Init DB
    conn = init_db(cidre_home / "cidre.db", embedding_dimensions=32)

    # Mock LLM
    mock_llm = MagicMock()
    call_count = {"n": 0}

    def fake_generate(prompt):
        call_count["n"] += 1
        if "Q1 Planning" in prompt or "Budget" in prompt:
            return "Description: Quarterly planning meeting notes\nCategories: work, meeting, planning\nSummary: Q1 budget and timeline review"
        return "Description: Italian pasta cooking recipe\nCategories: food, recipe, cooking\nSummary: Simple pasta recipe with instructions"

    mock_llm.generate.side_effect = fake_generate

    # Mock embedder — use deterministic fake embeddings
    mock_embedder = MagicMock()
    mock_embedder.dimensions = 32

    def fake_embed(texts):
        results = []
        for text in texts:
            if "planning" in text.lower() or "meeting" in text.lower():
                results.append([0.9, 0.1] + [0.0] * 30)
            elif "pasta" in text.lower() or "recipe" in text.lower():
                results.append([0.0, 0.0] + [0.9, 0.1] + [0.0] * 28)
            else:
                results.append([0.5] * 32)
        return results

    mock_embedder.embed.side_effect = fake_embed

    # Scan and index
    scanned = scan_directory(docs, exclude_patterns=["*.py"])
    assert len(scanned) == 2  # only .md files

    pipeline = IndexingPipeline(conn=conn, llm=mock_llm, embedder=mock_embedder)
    count = pipeline.index_batch([(s.path, s.file_type) for s in scanned])
    assert count == 2

    # Verify stats
    stats = get_index_stats(conn)
    assert stats["total"] == 2
    assert stats["markdown"] == 2

    # Search for meeting-related content
    engine = SearchEngine(conn=conn, embedder=mock_embedder)
    results = engine.search("quarterly planning meeting")
    assert len(results) >= 1
    assert "meeting" in results[0].ai_description.lower() or "planning" in results[0].ai_description.lower()

    # Search for recipe
    results = engine.search("pasta cooking recipe")
    assert len(results) >= 1
    assert "pasta" in results[0].ai_description.lower()

    # Verify code.py was excluded
    from cidre.db import get_item_by_path
    assert get_item_by_path(conn, str(docs / "code.py")) is None

    conn.close()
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_integration.py -v
```

Expected: 1 passed

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test for index and search flow"
```

---

## Task 14: Final Polish — README, CI, and First Tag

**Agent:** core-architect
**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md` (add badges, expand usage)

- [ ] **Step 1: Create GitHub Actions CI**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: macos-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint
        run: ruff check src/ tests/

      - name: Type check
        run: mypy src/cidre/ --ignore-missing-imports

      - name: Test
        run: pytest tests/ -v --tb=short
```

- [ ] **Step 2: Commit CI**

```bash
mkdir -p .github/workflows
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for lint, type check, and tests"
```

- [ ] **Step 3: Run full test suite one final time**

```bash
ruff check src/ tests/
pytest tests/ -v
```

Expected: All pass, no lint errors

- [ ] **Step 4: Tag v0.1.0**

```bash
git tag -a v0.1.0 -m "v0.1.0 — initial release"
```

---

## Agent Team Summary

| Task | Agent | Dependencies |
|------|-------|-------------|
| 1. Project Scaffolding | core-architect | None |
| 2. Config System | core-architect | Task 1 |
| 3. Database Schema | core-architect | Task 1 |
| 4. Embedding Providers | core-architect | Task 2 |
| 5. File Scanner | indexer-agent | Task 1 |
| 6. Vision Module | vision-agent | Task 1 |
| 7. Indexing Pipeline | indexer-agent | Tasks 3, 4, 5, 6 |
| 8. Apple Photos Importer | indexer-agent | Task 1 |
| 9. Search Engine | search-agent | Tasks 3, 4 |
| 10. Full CLI | core-architect | Tasks 2, 3, 4, 5, 7, 8, 9 |
| 11. File Watcher Daemon | indexer-agent | Tasks 5, 7 |
| 12. Landing Page | site-agent | None |
| 13. Integration Test | core-architect | Tasks 7, 9 |
| 14. CI + Final Polish | core-architect | All |

### Parallelization

These tasks can run in parallel after Task 1:
- **Parallel batch 1:** Tasks 2, 3, 5, 6, 8, 12 (no cross-dependencies)
- **Parallel batch 2:** Tasks 4, 9 (after 2, 3)
- **Parallel batch 3:** Task 7 (after 3, 4, 5, 6)
- **Sequential:** Tasks 10, 11, 13, 14 (depend on most prior work)
