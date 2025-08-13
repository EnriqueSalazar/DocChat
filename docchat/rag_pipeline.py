from typing import List, Tuple
from .embeddings import EmbeddingGenerator, ChromaDBManager
from .chunking import TextChunker
from .llm import LocalLLM

class RAGPipeline:
    """Handles the Retrieval Augmented Generation pipeline."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2",
                 chroma_persist_dir: str = "./chromadb",
                 llm_model_path: str = "togethercomputer/RedPajama-INCITE-7B-Instruct"):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model (str): Sentence transformer model name
            chroma_persist_dir (str): Directory for ChromaDB persistence
            llm_model_path (str): HF model repo or local path for RedPajama
        """
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.chroma_manager = ChromaDBManager(chroma_persist_dir)
        self.chunker = TextChunker()
        self.llm = LocalLLM(llm_model_path)
        
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
