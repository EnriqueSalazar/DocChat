import os
from pathlib import Path
from datetime import datetime
from docchat.database import DocumentDatabase
from docchat.document_loader import DocumentLoader
from docchat.chunking import TextChunker
from docchat.embeddings import EmbeddingGenerator, ChromaDBManager
from docchat.rag_pipeline import RAGPipeline

class DocChatApp:
    """Main application class for DocChat."""
    
    def __init__(self, docs_folder: str, model_path: str):
        self.docs_folder = Path(docs_folder)
        self.db = DocumentDatabase()
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.chroma_manager = ChromaDBManager()
        self.rag_pipeline = RAGPipeline(llm_model_path=model_path)
        
        # Ensure the document folder exists
        if not self.docs_folder.exists():
            self.docs_folder.mkdir(parents=True)
            print(f"Created document folder: {self.docs_folder}")
            
    def run(self):
        """Run the main application loop."""
        print("Starting DocChat...")
        self.process_documents()
        
        print("Welcome to DocChat! Type 'exit' or 'quit' to end.")
        self.chat_loop()
        
    def process_documents(self):
        """Process all documents in the folder."""
        print(f"Scanning for documents in: {self.docs_folder}")
        
        # Get current and processed files
        current_files = {p.resolve() for p in self.docs_folder.rglob('*') if p.is_file()}
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
            print(f"Found {len(new_files)} new documents to process.")
            self.add_documents(new_files)
            
        if modified_files:
            print(f"Found {len(modified_files)} modified documents to re-process.")
            self.update_documents(modified_files)
            
        if deleted_files:
            print(f"Found {len(deleted_files)} deleted documents to remove.")
            self.remove_documents(deleted_files)
            
        if not any([new_files, modified_files, deleted_files]):
            print("All documents are up to date.")
            
    def add_documents(self, file_paths: set):
        """Add new documents to the database and vector store."""
        for file_path in file_paths:
            content = self.loader.load_document(str(file_path))
            if content:
                chunks = self.chunker.chunk_text(content)
                embeddings = self.embedding_generator.generate_embeddings(chunks)
                
                # Create metadata and IDs for ChromaDB
                metadatas = [{'file_path': str(file_path)} for _ in chunks]
                ids = [f"{file_path}_{i}" for i in range(len(chunks))]
                
                self.chroma_manager.add_embeddings(chunks, embeddings, metadatas, ids)
                
                # Add to SQLite DB
                content_hash = self.db.calculate_file_hash(str(file_path))
                last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                self.db.add_file(str(file_path), content_hash, last_modified)
                
        print(f"Successfully added {len(file_paths)} documents.")
        
    def update_documents(self, file_paths: set):
        """Update modified documents."""
        self.remove_documents(file_paths, is_update=True)
        self.add_documents(file_paths)
        print(f"Successfully updated {len(file_paths)} documents.")
        
    def remove_documents(self, file_paths: set, is_update: bool = False):
        """Remove documents from the database and vector store."""
        for file_path in file_paths:
            # Remove from ChromaDB
            ids_to_delete = [f"{file_path}_{i}" for i in range(1000)] # Placeholder
            self.chroma_manager.delete_by_ids(ids_to_delete)
            
            # Remove from SQLite DB
            if not is_update:
                self.db.remove_file(str(file_path))
            
        print(f"Successfully removed {len(file_paths)} documents.")
        
    def chat_loop(self):
        """Handle the interactive chat session."""
        while True:
            query = input("\nAsk a question: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("Exiting DocChat. Goodbye!")
                break
                
            if not query:
                continue
                
            answer, sources = self.rag_pipeline.ask(query)
            
            print("\nAnswer:")
            print(answer)
            
            if sources:
                print("\nSources:")
                for source in sources:
                    print(f"- {source}")
            
            self.save_conversation(query, answer)
            
    def save_conversation(self, query: str, answer: str):
        """Save the conversation to a daily log file."""
        log_dir = Path("chat_history")
        if not log_dir.exists():
            log_dir.mkdir()
            
        log_file = log_dir / f"chat_history_{datetime.now().strftime('%Y-%m-%d')}.txt"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"User: {query}\n")
            f.write(f"AI: {answer}\n\n")
            f.write("-" * 40 + "\n")