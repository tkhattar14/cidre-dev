# Cidre — Design Spec

**Date:** 2026-04-09
**Status:** Approved
**Domain:** cidre.dev
**License:** MIT
**Tagline:** Local-first search intelligence for macOS

---

## 1. Problem

macOS Spotlight indexes everything indiscriminately — source code, caches, system files — while offering no semantic understanding of content. Photos are siloed in Apple Photos with limited search. There's no unified, privacy-first way to search across documents, photos, videos, and notes using natural language on a Mac.

## 2. Solution

Cidre is a local-only CLI tool that indexes documents, photos, videos, and notes on macOS, generates AI descriptions and categories using Gemma 4 via Ollama, and provides instant semantic search via embeddings. Nothing leaves the machine.

## 3. Target Hardware

- Apple Silicon Mac (M1+)
- Tested on: MacBook Pro M4 Pro, 12 cores, 24GB unified memory
- Minimum: 16GB unified memory (E4B fallback), recommended: 24GB+ (26B MoE)

## 4. Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  CLI (typer)  │────>│  Search Core │────>│  SQLite + sqlite-vec│
└──────────────┘     └──────┬───────┘     └──────────────────┘
                            │
                     ┌──────┴───────┐
                     │   Indexer     │
                     │  ┌──────────┐│
                     │  │ Scanner  ││──> watchdog daemon
                     │  └──────────┘│
                     │  ┌──────────┐│
                     │  │ Ollama   ││──> Gemma 4 26B MoE (vision + text)
                     │  │ Bridge   ││──> Embedding provider (configurable)
                     │  └──────────┘│
                     │  ┌──────────┐│
                     │  │ Photos   ││──> Apple Photos via osxphotos
                     │  │ Importer ││
                     │  └──────────┘│
                     └──────────────┘
```

### Tech Stack

- Python 3.12+
- Ollama for model serving
- `typer` for CLI
- `sqlite-vec` for vector storage (single-file SQLite)
- `osxphotos` for Apple Photos read-only access
- `watchdog` for filesystem monitoring
- `pdfplumber` for PDF text extraction
- `launchd` plist for daemon management

## 5. Models

### LLM (Description + Categorization)

| Model | Memory (Q4) | Active Params | Use Case |
|-------|-------------|---------------|----------|
| Gemma 4 26B MoE | ~7 GB | 4B per token | Default — best quality/memory ratio |
| Gemma 4 E4B | ~2.4 GB | 4.5B | Fallback for 16GB machines |
| Gemma 4 31B Dense | ~11 GB | 31B | Power users with 32GB+ |

All Gemma 4 models include a multimodal vision encoder for image/video understanding. User selects LLM at `cidre init` based on their available memory. Cidre auto-detects unified memory and recommends the best fit.

### Embedding (Configurable at Setup)

| Provider | Model | Params | RAM | Dimensions | Local |
|----------|-------|--------|-----|------------|-------|
| Ollama (default) | EmbeddingGemma | 308M | <200MB | 768 | Yes |
| Ollama | bge-m3 | 568M | ~1.2GB | 1024 | Yes |
| Ollama | qwen3-embedding:4b | 4B | ~2.5GB | 4096 | Yes |
| OpenAI | text-embedding-3-large | API | 0 | 3072 | No |

User selects at `cidre init`. Switching models requires re-indexing (different vector spaces).

### Embedding Provider Abstraction

```python
class EmbeddingProvider(Protocol):
    name: str
    dimensions: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

Implementations: `OllamaEmbedding`, `OpenAIEmbedding`. Stored in config, injected into indexer and search core.

## 6. Content Scope

### Indexed

- Photos and screenshots (JPEG, PNG, HEIC, WebP)
- Videos (MP4, MOV — key frame extraction)
- Documents (PDF)
- Markdown files (.md)
- Obsidian vaults
- Apple Photos library (read-only via osxphotos)

### Excluded

- Source code files (.py, .js, .ts, .swift, .go, .rs, .java, .c, .cpp, .h, etc.)
- Package directories (node_modules, .venv, __pycache__)
- Git internals (.git)
- System directories (Library/, .Trash/)
- User-configurable exclude patterns

## 7. Indexing Pipeline

```
File detected (new or modified)
    │
    ├─ Photo/Video?
    │   └─> Gemma 4 26B MoE (vision)
    │       ├─ Description: "A sunset over mountains with orange sky"
    │       ├─ Categories: [travel, landscape, sunset]
    │       └─ EXIF metadata: {date, location, dominant colors}
    │
    ├─ PDF?
    │   └─> pdfplumber (text extraction) ─> Gemma 4 (summarize + categorize)
    │
    ├─ Markdown/Notes?
    │   └─> Read content ─> Gemma 4 (summarize + categorize)
    │
    └─ All paths converge:
        Text description ─> EmbeddingProvider.embed() ─> vector
        │
        Store in SQLite:
          file_path, file_hash, file_type, size, modified_at,
          ai_description, categories[], summary,
          embedding vector (sqlite-vec),
          source ("filesystem" | "apple_photos")
```

### Indexing Behavior

- **Incremental:** File hash stored. Unchanged files are skipped.
- **Batching:** Photos queued and processed in batches. Gemma 4 loaded during batch, unloaded after.
- **Apple Photos:** Read-only via osxphotos. No export — reads originals in-place.
- **Video:** Extract 1 frame per 10 seconds, describe key frames, combine into video-level description.

### Indexing Modes

- **On-demand:** `cidre index` — full or filtered run
- **Daemon:** `cidre watch start` — background watchdog process for watched directories, managed via launchd
- **Hybrid (default):** Daemon watches configured directories; on-demand for everything else

## 8. Search

### Query Flow

```
Query: "receipts from last year"
    │
    ├─ Embed query via provider (~5ms)
    ├─ sqlite-vec: top-K nearest neighbors (K=20, ~5ms)
    ├─ Post-filter by metadata (type, date, category)
    └─ Rank and return results
```

### Smart Query Features

- **Date parsing:** "last year", "Q1 2025", "this month" → date range filter
- **Type inference:** "photos of..." → auto-filter to photo type
- **Category boost:** query mentions known category → boost matching results

### Output Format

```
$ cidre search "sunset photos"

1. IMG_7202.jpg                          [travel, landscape]
   "Golden sunset over the ocean from a beach"
   ~/Documents/IMG_7202.jpg · 2.4 MB · 2024-06-15
   Score: 0.92

2. vacation_23.png                       [landscape]
   "Mountain sunset with orange and purple sky"
   ~/Pictures/vacation_23.png · 1.8 MB · 2023-08-20
   Score: 0.87
```

## 9. CLI Interface

```bash
# Setup
cidre init                              # First-time setup, checks Ollama
cidre add ~/Documents                   # Add directory to index
cidre add --photos                      # Add Apple Photos library
cidre exclude "*.swift" "node_modules" ".git"

# Indexing
cidre index                             # Full index run
cidre index --only photos               # Index only photos
cidre index --only docs                 # Index only documents
cidre watch start                       # Start background daemon
cidre watch stop
cidre status                            # Show index stats

# Search
cidre search "sunset photos"
cidre search "invoices from 2025"
cidre search --type photo "mountains"
cidre search --type doc "health insurance"
cidre search --category receipts

# Categorization
cidre categories                        # List auto-detected categories
cidre categorize ~/Downloads            # Categorize a directory on-demand
```

## 10. Storage

```
~/.cidre/
├── config.toml          # embedding provider, watched dirs, excludes
├── cidre.db             # SQLite + sqlite-vec (single file)
└── logs/
    └── daemon.log
```

### Config Schema

```toml
[general]
llm_model = "gemma4:26b-a4b"

[embedding]
provider = "ollama"
model = "embeddinggemma"
dimensions = 768

[sources]
watched = ["~/Documents", "~/Desktop", "~/Downloads"]
photos = true

[exclude]
patterns = ["*.swift", "*.py", "*.js", "*.ts", "node_modules", ".git", "Library"]
```

## 11. Project Structure

```
cidre/
├── pyproject.toml
├── README.md
├── LICENSE (MIT)
├── src/
│   └── cidre/
│       ├── __init__.py
│       ├── cli.py              # typer CLI entry point
│       ├── config.py           # config loading/saving
│       ├── db.py               # SQLite + sqlite-vec schema & queries
│       ├── indexer/
│       │   ├── __init__.py
│       │   ├── scanner.py      # file discovery, hashing, filtering
│       │   ├── photos.py       # Apple Photos via osxphotos
│       │   ├── pipeline.py     # orchestrates description + embedding
│       │   └── daemon.py       # watchdog-based file watcher
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py         # EmbeddingProvider protocol
│       │   ├── ollama.py       # Ollama embedding + LLM bridge
│       │   └── openai.py       # OpenAI embedding (optional)
│       ├── search/
│       │   ├── __init__.py
│       │   ├── engine.py       # vector search + metadata filtering
│       │   └── query.py        # query parsing (dates, types, categories)
│       └── vision/
│           ├── __init__.py
│           ├── describe.py     # image/video description via Gemma 4
│           └── categorize.py   # auto-categorization logic
├── tests/
├── site/                       # cidre.dev landing page
│   ├── index.html
│   └── style.css
└── docs/
```

## 12. Landing Page (cidre.dev)

Single-page, minimalist, dark theme. Pure HTML/CSS, no framework.

### Sections

1. **Hero:** Name, tagline, animated terminal demo showing search
2. **How it works:** 3-step visual (Index → Embed → Search)
3. **Features:** Bullet grid (photos, docs, Apple Photos, daemon, model choice, 100% local)
4. **Install:** `brew install cidre` (future) + GitHub link
5. **Footer:** MIT License, Built for macOS

### Hosting

Static site — GitHub Pages or Cloudflare Pages from the `site/` directory.

## 13. Open Source Considerations

- **License:** MIT
- **CI:** GitHub Actions — lint, type check, tests (macOS runner for osxphotos tests)
- **Distribution:** PyPI (`pip install cidre`) + Homebrew formula (future)
- **Docs:** README covers setup, usage, configuration. Landing page for marketing.
- **Contributing:** CONTRIBUTING.md with setup instructions, code style, PR process
- **Platform:** macOS only (Apple Photos integration, launchd). Linux support could be added later without Apple Photos/launchd features.

## 14. Agent Team Structure

For building Cidre, the implementation will use an agent team with specialized teammates:

| Agent | Responsibility |
|-------|---------------|
| **core-architect** | DB schema, config system, provider abstraction, project scaffolding |
| **indexer-agent** | File scanner, indexing pipeline, Apple Photos importer, daemon |
| **search-agent** | Search engine, query parsing, result ranking |
| **vision-agent** | Ollama bridge, image/video description, categorization prompts |
| **site-agent** | Landing page HTML/CSS |

Agents work in isolation on their modules, sharing only the interfaces defined in this spec (EmbeddingProvider protocol, DB schema, config schema).
