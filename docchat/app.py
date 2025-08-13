import logging
from pathlib import Path
from datetime import datetime

from docchat.database import DocumentDatabase
from docchat.document_loader import DocumentLoader
from docchat.chunking import TextChunker
from docchat.embeddings import EmbeddingGenerator, ChromaDBManager
from docchat.rag_pipeline import RAGPipeline
from docchat.model_downloader import ensure_local_model
from docchat.config import Config
from docchat.logging_setup import setup_logging
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown


class DocChatApp:
    """Main application class for DocChat."""

    def __init__(self, config: Config, verbose: bool = False, stream_mode: bool = False):
        self.config = config
        # Configure logging early; force reconfigure to adjust verbosity
        setup_logging(Path("./logs"), level=logging.DEBUG if verbose else logging.INFO, force=True)
        self.logger = logging.getLogger("DocChat")
        self.verbose = verbose
        self.console = Console()
        self.stream_mode = stream_mode

        # Core components
        self.docs_folder = config.docs_path
        self.db = DocumentDatabase()
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(model_name=config.embedding_model)
        self.chroma_manager = ChromaDBManager(persist_directory=str(config.vectorstore_path))

        # Determine model identifier (HF repo or local downloaded directory)
        model_id = getattr(config, 'llm_model_name', "togethercomputer/RedPajama-INCITE-7B-Instruct")

        # Allow environment override for testing with smaller models
        import os as _os
        env_override = _os.getenv("DOCCHAT_MODEL_NAME")
        if env_override:
            self.logger.info("Overriding model name via DOCCHAT_MODEL_NAME=%s", env_override)
            model_id = env_override

        # Optional explicit download to local folder to avoid repeated cache fetch; optimize for faster generation
        if getattr(config, 'llm_auto_download', True):
            local_dir = Path('./model/redpajama')
            try:
                model_path = ensure_local_model(model_id, local_dir)
                model_id = str(model_path)
            except Exception as e:
                self.logger.warning("Falling back to remote model due to snapshot error: %s", e)

        if self.verbose:
            self.logger.debug(
                "Initializing RAG pipeline with model_id=%s embedding_model=%s", model_id, config.embedding_model
            )

        self.rag_pipeline = RAGPipeline(
            embedding_model=config.embedding_model,
            chroma_persist_dir=str(config.vectorstore_path),
            llm_model_path=model_id,
            llm_max_new_tokens=config.llm_max_new_tokens,
            cpu_threads=config.cpu_threads,
            enable_gpu=config.enable_gpu,
            context_max_chars=config.context_max_chars,
            dynamic_quantization=config.cpu_dynamic_quantization,
            llm_compile=False,
        )

        # Wire adaptive configuration if enabled
        if config.adaptive_enable:
            adaptive_cfg = {
                'enable': True,
                'latency_target': config.adaptive_latency_target,
                'min_new_tokens': config.adaptive_min_new_tokens,
                'max_new_tokens': config.adaptive_max_new_tokens,
                'growth_factor': config.adaptive_growth_factor,
                'shrink_factor': config.adaptive_shrink_factor,
                'ema_alpha': config.adaptive_ema_alpha,
            }
            self.rag_pipeline.configure_adaptive(adaptive_cfg)

        # Ensure the document and history folders exist
        self.docs_folder.mkdir(parents=True, exist_ok=True)
        config.history_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Start chat loop (no ingestion here)."""
        self.logger.info("Starting DocChat chat session...")
        self.logger.info("Welcome to DocChat! Type 'exit' or 'quit' to end.")
        self.chat_loop()

    def process_documents(self):
        """Process all documents in the folder."""
        self.logger.info(f"Scanning for documents in: {self.docs_folder}")

        # Get current and processed files
        current_files = {p.resolve() for p in self.docs_folder.rglob("*") if p.is_file()}
        processed_files = {Path(f[0]).resolve() for f in self.db.get_processed_files()}

        # Find new, modified, and deleted files
        new_files = current_files - processed_files
        deleted_files = processed_files - current_files

        # Process modified files
        modified_files = set()
        for file_path in processed_files.intersection(current_files):
            content_hash = self.db.calculate_file_hash(str(file_path))
            last_hash = self.db.get_file_hash(str(file_path))
            if content_hash != last_hash:
                modified_files.add(file_path)

        if new_files:
            self.logger.info(f"Found {len(new_files)} new documents to process.")
            self.add_documents(new_files)

        if modified_files:
            self.logger.info(f"Found {len(modified_files)} modified documents to re-process.")
            self.update_documents(modified_files)

        if deleted_files:
            self.logger.info(f"Found {len(deleted_files)} deleted documents to remove.")
            self.remove_documents(deleted_files)

        if not any([new_files, modified_files, deleted_files]):
            self.logger.info("All documents are up to date.")

    def add_documents(self, file_paths: set):
        """Add new documents to the database and vector store."""
        for file_path in file_paths:
            content = self.loader.load_document(str(file_path))
            if content:
                chunks = self.chunker.chunk_text(content)
                embeddings = self.embedding_generator.generate_embeddings(chunks)

                # Create metadata and IDs for ChromaDB
                metadatas = [{"file_path": str(file_path)} for _ in chunks]
                ids = [f"{str(file_path)}::{i}" for i in range(len(chunks))]

                self.chroma_manager.add_embeddings(chunks, embeddings, metadatas, ids)

                # Add to SQLite DB
                content_hash = self.db.calculate_file_hash(str(file_path))
                last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                self.db.add_file(str(file_path), content_hash, last_modified)

        self.logger.info(f"Successfully added {len(file_paths)} documents.")

    def update_documents(self, file_paths: set):
        """Update modified documents."""
        self.remove_documents(file_paths, is_update=True)
        self.add_documents(file_paths)
        self.logger.info(f"Successfully updated {len(file_paths)} documents.")

    def remove_documents(self, file_paths: set, is_update: bool = False):
        """Remove documents from the database and vector store."""
        for file_path in file_paths:
            # Remove from ChromaDB using metadata filter
            self.chroma_manager.delete_where_file(str(file_path))

            # Remove from SQLite DB
            if not is_update:
                self.db.remove_file(str(file_path))

        self.logger.info(f"Successfully removed {len(file_paths)} documents.")

    def chat_loop(self):
        """Handle the interactive chat session."""
        while True:
            self.console.print("\n[bold cyan]Ask a question[/]: ", end="")
            query = input("").strip()
            if query.lower() in ["exit", "quit"]:
                self.logger.info("Exiting DocChat. Goodbye!")
                break

            if not query:
                continue
            if self.stream_mode:
                # Streaming answer
                self.console.print("[yellow]Generating (stream)...[/]")
                pieces = []
                sources_final = []
                for piece, sources in self.rag_pipeline.ask_stream(query, top_k=self.config.top_k):
                    if sources is None:
                        # partial text
                        pieces.append(piece)
                        self.console.print(piece, end="", style="green")
                    else:
                        # final result
                        final_answer = piece
                        sources_final = sources
                        if final_answer:
                            self.console.print("")
                            self.console.print(f"[bold green]{final_answer}[/]")
                        break
                if sources_final:
                    sources_text = "\n".join(f"• {s}" for s in sources_final)
                    self.console.print(Panel.fit(sources_text, title="[magenta]Sources[/]", border_style="magenta"))
                self.save_conversation(query, ''.join(pieces))
            else:
                # Non-stream mode: show quick status line before possibly long generation
                self.console.print("[yellow]Generating...[/]", end="", style="yellow")
                answer, sources = self.rag_pipeline.ask(query, top_k=self.config.top_k)
                self.console.print("")  # newline after generation
                self.console.print(f"\n[bold green]{answer}[/]")
                if sources:
                    sources_text = "\n".join(f"• {s}" for s in sources)
                    self.console.print(Panel.fit(sources_text, title="[magenta]Sources[/]", border_style="magenta"))
                self.save_conversation(query, answer)

    def save_conversation(self, query: str, answer: str):
        """Save the conversation to a daily log file."""
        log_dir = self.config.history_dir
        log_file = log_dir / f"chat_history_{datetime.now().strftime('%Y-%m-%d')}.txt"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"User: {query}\n")
            f.write(f"AI: {answer}\n\n")
            f.write("-" * 40 + "\n")