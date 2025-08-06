
import sys
from pathlib import Path
import typer
from typing_extensions import Annotated

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from docchat.app import DocChatApp

# Create a Typer app instance
cli = typer.Typer()

def find_model_path() -> str:
    """Find the first .gguf model file in the 'model' directory."""
    model_dir = Path("model")
    if not model_dir.exists() or not model_dir.is_dir():
        return None

    for file_path in model_dir.glob("*.gguf"):
        return str(file_path)
    
    return None

@cli.command()
def main(
    docs_folder: Annotated[Path, typer.Argument(
        help="The path to the folder containing your documents.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    )] = Path("./docs"),
):
    """
    A CLI application for chatting with your local documents.
    """
    # Find the model path
    model_path = find_model_path()
    
    if not model_path:
        typer.secho("Error: No .gguf model file found in the 'model' directory.", fg=typer.colors.RED, err=True)
        typer.secho("Please place your GGUF model file in a folder named 'model' in the project root.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    
    typer.secho(f"Found model: {model_path}", fg=typer.colors.GREEN)
    
    # Instantiate and run the application
    try:
        app = DocChatApp(str(docs_folder), model_path)
        app.run()
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    cli()
