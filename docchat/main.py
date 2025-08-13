
import sys
from pathlib import Path
import typer
from typing_extensions import Annotated

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from docchat.app import DocChatApp
from docchat.config import Config

cli = typer.Typer(
    help="DocChat CLI – Ingest documents and chat locally with RAG",
    invoke_without_command=True,
    add_completion=False,  # hide completion install/show options
)


def build_config(docs_folder: Path | None) -> Config:
    cfg = Config.load("config.yaml")
    if docs_folder is not None:
        cfg.docs_path = docs_folder
    return cfg


@cli.command()
def ingest(
    docs_folder: Annotated[Path, typer.Argument(help="Folder with documents", exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True)] = Path("./docs"),
):
    """Ingest documents then start chat."""
    cfg = build_config(docs_folder)
    app = DocChatApp(cfg)
    app.process_documents()
    # Separator instead of pause
    if sys.stdin.isatty():
        typer.echo("\n--- Ingestion complete ---\n")
    else:
        typer.echo("(Non-interactive mode: skipping chat after ingestion)")
    # After ingestion always go to chat if interactive
    if sys.stdin.isatty():
        app.run()


@cli.command()
def chat(
    docs_folder: Annotated[Path, typer.Argument(help="Folder with documents", exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True)] = Path("./docs"),
):
    """Start an interactive chat session using local LLM over your docs."""
    cfg = build_config(docs_folder)
    app = DocChatApp(cfg)
    app.run()


@cli.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context, docs_folder: Annotated[Path, typer.Option(help="Folder with documents", exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True)] = Path("./docs")):
    """Default behavior: show help (implicitly) then launch chat if no subcommand supplied."""
    if ctx.invoked_subcommand is not None:
        return  # Another command will run
    # Show help manually (since we removed no_args_is_help)
    typer.echo(ctx.get_help())
    if not sys.stdin.isatty():
        typer.echo("(Non-interactive mode: chat skipped)")
        return
    # Interactive: no pause, just a blank line separator then chat
    typer.echo("\n")
    cfg = build_config(docs_folder)
    app = DocChatApp(cfg)
    app.run()

if __name__ == "__main__":
    cli()
