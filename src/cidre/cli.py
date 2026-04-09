from __future__ import annotations

import json
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

    try:
        subprocess.check_output(["ollama", "--version"], stderr=subprocess.STDOUT)
        console.print("[green]Ollama found.[/green]")
    except FileNotFoundError:
        console.print("[red]Ollama not found. Install it from https://ollama.com[/red]")
        raise typer.Exit(1)

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

    console.print("\n[bold]Choose embedding model:[/bold]")
    emb_choices = list(EMBEDDING_MODELS.keys())
    for i, model in enumerate(emb_choices, 1):
        info = EMBEDDING_MODELS[model]
        local = "local" if info["provider"] == "ollama" else "[yellow]cloud — data leaves machine[/yellow]"
        console.print(f"  {i}. {model} — {info['dimensions']}d, {local}")

    emb_idx = typer.prompt("Select", type=int, default=1) - 1
    emb_model = emb_choices[max(0, min(emb_idx, len(emb_choices) - 1))]
    emb_info = EMBEDDING_MODELS[emb_model]

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
        photos_list = list_apple_photos()
        photo_files = [(Path(p.original_path), "photo") for p in photos_list if p.original_path]
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

    db_path = CIDRE_HOME / "cidre.db"
    if not db_path.exists():
        console.print("[red]No index found. Run: cidre index[/red]")
        raise typer.Exit(1)

    from cidre.db import init_db
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
