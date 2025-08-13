
import sys
from pathlib import Path
import typer
from typing_extensions import Annotated

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from docchat.app import DocChatApp
from docchat.config import Config

cli = typer.Typer(help="DocChat CLI – Ingest documents and chat locally with RAG", invoke_without_command=True, no_args_is_help=True)


def find_model_path() -> Path | None:
    """Find the first .gguf model file in the 'model' directory."""
    model_dir = Path("model")
    if not model_dir.exists() or not model_dir.is_dir():
        return None
    for file_path in model_dir.glob("*.gguf"):
        return file_path
    return None


def build_config(docs_folder: Path | None) -> Config:
    cfg = Config.load("config.yaml")
    if docs_folder is not None:
        cfg.docs_path = docs_folder
    if not cfg.llm_model_path:
        auto = find_model_path()
        if auto:
            cfg.llm_model_path = auto
    return cfg


@cli.command()
def ingest(
    docs_folder: Annotated[Path, typer.Argument(help="Folder with documents", exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True)] = Path("./docs"),
):
    """Ingest documents then start chat."""
    cfg = build_config(docs_folder)
    app = DocChatApp(cfg)
    app.process_documents()
    # After ingestion always go to chat
    if not cfg.llm_model_path:
        typer.secho("No GGUF model found. Set llm_model_path in config.yaml or put a .gguf in ./model", fg=typer.colors.RED)
        raise typer.Exit(1)
    app.run()


@cli.command()
def chat(
    docs_folder: Annotated[Path, typer.Argument(help="Folder with documents", exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True)] = Path("./docs"),
):
    """Start an interactive chat session using local LLM over your docs."""
    cfg = build_config(docs_folder)
    if not cfg.llm_model_path:
        typer.secho("No GGUF model found. Set llm_model_path in config.yaml or put a .gguf in ./model", fg=typer.colors.RED)
        raise typer.Exit(1)
    app = DocChatApp(cfg)
    app.run()


@cli.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context, docs_folder: Annotated[Path, typer.Option(help="Folder with documents", exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True)] = Path("./docs")):
    """Default behavior: show help (implicitly) then launch chat if no subcommand supplied."""
    if ctx.invoked_subcommand is not None:
        return
    # Show help intentionally (since invoke_without_command + no_args_is_help) then proceed
    typer.echo(ctx.get_help())
    cfg = build_config(docs_folder)
    if not cfg.llm_model_path:
        typer.secho("No GGUF model found. Set llm_model_path in config.yaml or put a .gguf in ./model", fg=typer.colors.RED)
        raise typer.Exit(1)
    app = DocChatApp(cfg)
    app.run()

if __name__ == "__main__":
    cli()
