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


@app.command()
def init():
    """Initialize Cidre in the current directory."""
    typer.echo("Cidre initialized.")
