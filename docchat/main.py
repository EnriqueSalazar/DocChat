import os
import sys
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from docchat.app import DocChatApp


def find_model_path() -> str:
    """Find the first .gguf model file in the 'model' directory."""
    model_dir = Path("model")
    if not model_dir.exists() or not model_dir.is_dir():
        return None

    for file_path in model_dir.glob("*.gguf"):
        return str(file_path)
    
    return None

def main():
    """Main entry point for the DocChat CLI application."""
    # Get document folder from command line argument or use default
    if len(sys.argv) > 1:
        docs_folder = sys.argv[1]
    else:
        docs_folder = "./docs"
        
    # Find the model path
    model_path = find_model_path()
    
    if not model_path:
        print("Error: No .gguf model file found in the 'model' directory.")
        print("Please place your GGUF model file in a folder named 'model' in the project root.")
        sys.exit(1)
    
    print(f"Found model: {model_path}")
    
    app = DocChatApp(docs_folder, model_path)
    app.run()


if __name__ == "__main__":
    main()