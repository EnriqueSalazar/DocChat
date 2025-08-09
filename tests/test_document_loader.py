
import unittest
from pathlib import Path
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from docchat.document_loader import DocumentLoader

class TestDocumentLoader(unittest.TestCase):
    """Tests for the DocumentLoader class."""

    def setUp(self):
        """Set up a DocumentLoader instance and create dummy files."""
        self.loader = DocumentLoader()
        self.docs_dir = Path("docs")
        
        # Ensure dummy files exist
        if not self.docs_dir.exists():
            self.docs_dir.mkdir()
            
        with open(self.docs_dir / "test_doc.txt", "w") as f:
            f.write("This is a test txt file.")
        with open(self.docs_dir / "test_doc.md", "w") as f:
            f.write("# This is a test markdown file")
        with open(self.docs_dir / "test_doc.csv", "w") as f:
            f.write("col1,col2\nval1,val2")
        # A dummy PDF needs to be handled differently, we'll just check for error handling
        with open(self.docs_dir / "corrupted.pdf", "w") as f:
            f.write("this is not a pdf")

    def test_load_txt_file(self):
        """Test loading a .txt file."""
        content = self.loader.load_document(str(self.docs_dir / "test_doc.txt"))
        self.assertEqual(content, "This is a test txt file.")

    def test_load_md_file(self):
        """Test loading a .md file."""
        content = self.loader.load_document(str(self.docs_dir / "test_doc.md"))
        # markdown library converts it to html
        self.assertEqual(content, "<h1>This is a test markdown file</h1>")

    def test_load_csv_file(self):
        """Test loading a .csv file."""
        content = self.loader.load_document(str(self.docs_dir / "test_doc.csv"))
        self.assertEqual(content, "col1, col2\nval1, val2")

    def test_unsupported_file_type(self):
        """Test that an unsupported file type returns an empty string."""
        with open(self.docs_dir / "test.unsupported", "w") as f:
            f.write("test")
        content = self.loader.load_document(str(self.docs_dir / "test.unsupported"))
        self.assertEqual(content, "")

    def test_corrupted_pdf_file(self):
        """Test that a corrupted PDF file is handled gracefully."""
        content = self.loader.load_document(str(self.docs_dir / "corrupted.pdf"))
        self.assertEqual(content, "")

    def test_file_not_found(self):
        """Test that a non-existent file is handled gracefully."""
        content = self.loader.load_document("non_existent_file.txt")
        self.assertEqual(content, "")

if __name__ == "__main__":
    unittest.main()
