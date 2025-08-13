from typing import List, Tuple
from .embeddings import EmbeddingGenerator, ChromaDBManager
from .chunking import TextChunker
from .llm import LocalLLM

class RAGPipeline:
    """Handles the Retrieval Augmented Generation pipeline."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chroma_persist_dir: str = "./chromadb",
        llm_model_path: str = "togethercomputer/RedPajama-INCITE-7B-Instruct",
        llm_max_new_tokens: int = 512,
        cpu_threads: int | None = None,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
            embedding_model: Sentence transformer model name.
            chroma_persist_dir: Directory for ChromaDB persistence.
            llm_model_path: HF model repo or local path for RedPajama.
            llm_max_new_tokens: Max new tokens for generation.
            cpu_threads: Optional override for torch CPU threads.
        """
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.chroma_manager = ChromaDBManager(chroma_persist_dir)
        self.chunker = TextChunker()
        self.llm = LocalLLM(
            llm_model_path,
            max_new_tokens=llm_max_new_tokens,
            cpu_threads=cpu_threads,
        )
        
    def ask(self, query: str, top_k: int = 4) -> Tuple[str, List[str]]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query (str): User's question
            
        Returns:
            A tuple containing the answer and a list of sources.
        """
        # Compute query embedding and query ChromaDB
        q_emb = self.embedding_generator.model.encode([query])[0].tolist()
        results = self.chroma_manager.query_by_embedding(q_emb, n_results=top_k)

        # Extract documents and metadata from results
        if not results.get('documents') or not results['documents'][0]:
            return "I couldn't find any relevant information in your documents.", []

        # Combine retrieved chunks into context
        context_chunks = results['documents'][0]
        sources = set()

        if 'metadatas' in results and results['metadatas']:
            for metadata in results['metadatas'][0]:
                if 'file_path' in metadata:
                    sources.add(metadata['file_path'])

        context_text = "\n\n".join(context_chunks)

        # Generate a response from the LLM
        answer = self.llm.generate_response(query, context_text)

        # Check for the specific [NO_ANSWER] token
        if answer == "[NO_ANSWER]":
            return "I could not find an answer to that in the documents.", []
        else:
            return answer, list(sources)

    def ask_stream(self, query: str, top_k: int = 4):
        """Stream answer tokens. Yields (partial_text, sources or None). Final yield returns full answer & sources."""
        q_emb = self.embedding_generator.model.encode([query])[0].tolist()
        results = self.chroma_manager.query_by_embedding(q_emb, n_results=top_k)
        if not results.get('documents') or not results['documents'][0]:
            yield "I couldn't find any relevant information in your documents.", []
            return
        context_chunks = results['documents'][0]
        sources = set()
        if 'metadatas' in results and results['metadatas']:
            for metadata in results['metadatas'][0]:
                if 'file_path' in metadata:
                    sources.add(metadata['file_path'])
        context_text = "\n\n".join(context_chunks)
        streamer = self.llm.stream_response(query, context_text)
        final_answer = ""
        try:
            for piece in streamer:
                final_answer += piece
                yield piece, None
            # After generator finishes, streamer returns final processed answer via return value
            # Python generators raise StopIteration with value prop; we can't capture here, so post-process again
            processed = self.llm._post_process(final_answer)
            if processed == "[NO_ANSWER]":
                yield "[NO_ANSWER]", []
            else:
                yield processed, list(sources)
        except Exception:
            yield "Error streaming response.", []
