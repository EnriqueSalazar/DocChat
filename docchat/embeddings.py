from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any
import os

class EmbeddingGenerator:
    """Handles generation of embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self.model.encode(texts).tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of the embeddings."""
        # Get a sample embedding to determine dimension
        sample = self.model.encode(["test"])
        return len(sample[0])

class ChromaDBManager:
    """Handles interaction with ChromaDB vector database."""
    
    def __init__(self, persist_directory: str = "./chromadb"):
        """
        Initialize the ChromaDB manager.
        
        Args:
            persist_directory (str): Directory to store the database
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "documents"
        self.collection = None
        
    def get_or_create_collection(self):
        """Get or create a collection in ChromaDB."""
        if not self.collection:
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                # Collection doesn't exist, create it
                dimension = 384  # Default for all-MiniLM-L6-v2 model
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None  # We'll handle embeddings separately
                )
        return self.collection
    
    def add_embeddings(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """
        Add text embeddings to the ChromaDB collection.
        
        Args:
            texts (List[str]): Texts to embed and store
            metadatas (List[Dict[str, Any]]): Metadata for each text
            ids (List[str]): Unique IDs for each embedding
        """
        # Get or create collection
        collection = self.get_or_create_collection()
        
        # Add documents with metadata
        collection.add(
            embeddings=[],
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_text: str, n_results: int = 4) -> List[Dict]:
        """
        Query the ChromaDB for similar documents.
        
        Args:
            query_text (str): Text to search for
            n_results (int): Number of results to return
            
        Returns:
            List[Dict]: Results from the database
        """
        # Get or create collection
        collection = self.get_or_create_collection()
        
        # Query with similarity search
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        
        return results
    
    def delete_by_ids(self, ids: List[str]):
        """
        Delete documents by their IDs.
        
        Args:
            ids (List[str]): List of document IDs to delete
        """
        if self.collection:
            try:
                self.collection.delete(ids=ids)
            except Exception as e:
                print(f"Error deleting from ChromaDB: {e}")