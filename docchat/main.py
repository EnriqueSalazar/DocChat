
import os
import sys
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from docchat.app import DocChatApp


def main():
    """Main entry point for the DocChat CLI application."""
    # Get document folder from command line argument or use default
    if len(sys.argv) > 1:
        docs_folder = sys.argv[1]
    else:
        docs_folder = "./docs"
        
    # Get model path from environment variable or a default path
    model_path = os.environ.get("LLM_MODEL_PATH", "path/to/your/llm_model.gguf")
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please set the LLM_MODEL_PATH environment variable or place the model at the default path.")
        sys.exit(1)
    
    app = DocChatApp(docs_folder, model_path)
    app.run()


if __name__ == "__main__":
    main()
