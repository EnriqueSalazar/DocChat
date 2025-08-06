from typing import List

class TextChunker:
    """Handles splitting text into meaningful chunks."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        """
        Initialize the chunker.
        
        Args:
            chunk_size (int): Maximum size of each chunk in characters
            overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Text to be chunked
            
        Returns:
            List[str]: List of text chunks
        """
        if not text:
            return []
            
        # Simple character-based chunking with overlap
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            
            # Ensure we don't cut words in half if possible
            if end < len(text) and not chunk[-1].isspace():
                # Find the last space to avoid cutting words
                last_space = chunk.rfind(' ')
                if last_space > 0:
                    chunk = chunk[:last_space]
            
            chunks.append(chunk)
            
            # Move start position with overlap
            start += self.chunk_size - self.overlap
            
        return chunks