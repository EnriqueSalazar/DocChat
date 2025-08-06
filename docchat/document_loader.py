import os
from pathlib import Path
from typing import List, Tuple
import csv
import pypdf
import markdown

class DocumentLoader:
    """Handles loading and parsing of different document types."""
    
    def __init__(self):
        pass
    
    def load_document(self, file_path: str) -> str:
        """
        Load a document based on its extension.
        
        Args:
            file_path (str): Path to the document
            
        Returns:
            str: Content of the document
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")
            
            suffix = path.suffix.lower()
            
            if suffix == '.txt':
                return self._load_text_file(path)
            elif suffix == '.md':
                return self._load_markdown_file(path)
            elif suffix == '.pdf':
                return self._load_pdf_file(path)
            elif suffix == '.csv':
                return self._load_csv_file(path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
        except Exception as e:
            print(f"Could not parse document {file_path}: {str(e)}")
            return ""
    
    def _load_text_file(self, path: Path) -> str:
        """Load a plain text file."""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_markdown_file(self, path: Path) -> str:
        """Load and convert markdown to plain text."""
        with open(path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to plain text
        html = markdown.markdown(md_content)
        # Simple conversion - in a real app you might want more sophisticated HTML-to-text conversion
        return html
    
    def _load_pdf_file(self, path: Path) -> str:
        """Load content from a PDF file."""
        try:
            with open(path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        except Exception as e:
            raise ValueError(f"Could not parse PDF file: {str(e)}")
    
    def _load_csv_file(self, path: Path) -> str:
        """Load content from a CSV file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Convert to simple text format
                rows = []
                for row in reader:
                    rows.append(', '.join(row))
                return '\n'.join(rows)
        except Exception as e:
            raise ValueError(f"Could not parse CSV file: {str(e)}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported document extensions."""
        return ['.txt', '.md', '.pdf', '.csv']
