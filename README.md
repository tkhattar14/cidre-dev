# Cidre

**Local-first search intelligence for macOS.**

Cidre indexes your documents, photos, videos, and notes using on-device AI, then lets you search them instantly with natural language. Everything runs locally — nothing leaves your machine.

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

## How It Works

Cidre uses a two-phase approach to deliver fast, accurate search:

1. **Index** — Gemma 4 (via Ollama) analyzes each file and generates a natural language description + categories. Photos are described by what's in them. PDFs are summarized. Markdown is categorized.

2. **Embed** — Each description is converted into a vector using a configurable embedding model and stored in a local SQLite database.

3. **Search** — Your query is embedded and matched against the index using vector similarity. Results are instant (~5ms) with no LLM call required at search time.

## Features

- **Semantic search** — Find files by meaning, not just filename. "receipts from last year" or "photos of mountains" just work.
- **Auto-categorization** — Every indexed file is automatically tagged with categories (travel, receipt, work, food, etc.).
- **Apple Photos integration** — Search your Photos library without exporting. Read-only access via [osxphotos](https://github.com/RhetTbull/osxphotos).
- **Multi-format support** — Photos (JPEG, PNG, HEIC, WebP), videos (MP4, MOV), PDFs, and Markdown files.
- **Incremental indexing** — Only new or changed files are processed. File hashes prevent redundant work.
- **Background daemon** — Watches your directories for changes and indexes automatically via macOS launchd.
- **Choose your embedding model** — EmbeddingGemma (default), bge-m3, qwen3-embedding, or OpenAI.
- **Smart query parsing** — Date expressions ("from 2025", "last year", "Q1 2025"), type hints ("photos of..."), and category filters are parsed automatically.
- **100% local** — Powered by Ollama. No API keys required. No data sent anywhere.

## Requirements

- macOS (Apple Silicon recommended)
- Python 3.12+
- [Ollama](https://ollama.com) installed and running

### Hardware

| RAM | Recommended Model | Notes |
|-----|------------------|-------|
| 16GB | `gemma4` (E4B, 9.6GB) | Works well, leave headroom for apps |
| 24GB | `gemma4` (E4B, 9.6GB) | Comfortable with room to spare |
| 32GB+ | `gemma4:26b` (26B MoE, 17GB) | Higher quality descriptions |

## Installation

```bash
# Install Ollama
brew install ollama
brew services start ollama

# Pull models
ollama pull gemma4
ollama pull embeddinggemma

# Install Cidre
pip install cidre
```

Or install from source:

```bash
git clone https://github.com/tkhattar14/cidre-dev.git
cd cidre-dev
pip install -e ".[dev]"
```

## Quick Start

```bash
# Initialize — choose your LLM and embedding model
cidre init

# Add directories to index
cidre add ~/Documents
cidre add ~/Desktop
cidre add ~/Downloads

# Add Apple Photos library
cidre add --photos

# Run indexing
cidre index

# Search
cidre search "sunset photos"
```

## Usage

### Search

```bash
# Basic search
cidre search "quarterly budget report"

# Filter by type
cidre search --type photo "mountains"
cidre search --type doc "insurance policy"

# Filter by category
cidre search --category receipt "2025"

# Limit results
cidre search -n 5 "meeting notes"
```

Cidre automatically parses natural language queries:
- `"photos of mountains"` → filters to photos, searches for "mountains"
- `"invoices from 2025"` → filters to year 2025, searches for "invoices"
- `"receipts from last year"` → filters to previous year
- `"docs from Q1 2025"` → filters to Jan-Mar 2025

### Indexing

```bash
# Index everything
cidre index

# Index only specific types
cidre index --only photos
cidre index --only docs
cidre index --only markdown
```

### Background Watching

```bash
# Start daemon (watches configured directories for changes)
cidre watch start

# Stop daemon
cidre watch stop
```

The daemon uses macOS launchd — it starts automatically on login and indexes new/changed files in real time.

### Managing Sources

```bash
# Add a directory
cidre add ~/Pictures

# Add Apple Photos
cidre add --photos

# Add exclude patterns
cidre exclude "*.log" "tmp"

# Check status
cidre status

# List categories
cidre categories
```

## Configuration

Cidre stores its config at `~/.cidre/config.toml`:

```toml
[general]
llm_model = "gemma4"

[embedding]
provider = "ollama"
model = "embeddinggemma"
dimensions = 768

[sources]
watched = ["/Users/you/Documents", "/Users/you/Desktop"]
photos = true

[exclude]
patterns = ["*.py", "*.js", "*.ts", "node_modules", ".git", "Library"]
```

### Embedding Models

Choose at setup (`cidre init`) based on your needs:

| Model | Size | Quality | Local | Best For |
|-------|------|---------|-------|----------|
| **EmbeddingGemma** (default) | <200MB | Good | Yes | General use, low memory |
| **bge-m3** | 1.2GB | Good | Yes | Hybrid dense+sparse search |
| **qwen3-embedding:4b** | 2.5GB | Best | Yes | Maximum quality |
| **OpenAI text-embedding-3-large** | API | Best | No | Cloud-based (data leaves machine) |

Switching embedding models requires re-indexing (`cidre index`).

## Architecture

```
~/.cidre/
├── config.toml          # Models, sources, exclude patterns
├── cidre.db             # SQLite + sqlite-vec (single file)
└── logs/
    └── daemon.log       # Watcher daemon logs
```

```
src/cidre/
├── cli.py               # Typer CLI (init, add, index, search, watch, status)
├── config.py            # Config dataclass + TOML read/write
├── db.py                # SQLite + sqlite-vec schema and queries
├── indexer/
│   ├── scanner.py       # File discovery, hashing, type classification
│   ├── pipeline.py      # Orchestrates: describe → embed → store
│   ├── photos.py        # Apple Photos read-only import
│   └── daemon.py        # Watchdog file watcher + launchd
├── providers/
│   ├── base.py          # EmbeddingProvider protocol
│   ├── ollama.py        # Ollama embedding + LLM client
│   └── openai.py        # OpenAI embedding client
├── search/
│   ├── engine.py        # Vector search + metadata filtering
│   └── query.py         # Natural language query parsing
└── vision/
    ├── describe.py      # Image/video/doc/markdown description via Gemma 4
    └── categorize.py    # Category extraction and parsing
```

## What Gets Indexed

### Included
- Photos and screenshots — JPEG, PNG, HEIC, WebP
- Videos — MP4, MOV (key frame extraction)
- Documents — PDF (text extracted via pdfplumber)
- Notes — Markdown files, Obsidian vaults
- Apple Photos library (read-only, no export)

### Excluded by Default
- Source code files (`.py`, `.js`, `.ts`, `.swift`, `.go`, `.rs`, `.java`, `.c`, `.cpp`)
- Package directories (`node_modules`, `.venv`, `__pycache__`)
- System directories (`Library/`, `.Trash/`, `.git/`)
- Configurable via `cidre exclude`

## Development

```bash
# Clone and install
git clone https://github.com/tkhattar14/cidre-dev.git
cd cidre-dev
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Type check
mypy src/cidre/ --ignore-missing-imports
```

## Privacy

Cidre is designed to be completely local:

- All AI inference runs on your Mac via Ollama — no cloud API calls (unless you explicitly choose OpenAI embeddings)
- The index database lives at `~/.cidre/cidre.db` — a single SQLite file on your disk
- Apple Photos are accessed read-only — no files are copied or exported
- No telemetry, no analytics, no network requests

## License

MIT — see [LICENSE](LICENSE) for details.
