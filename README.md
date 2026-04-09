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
