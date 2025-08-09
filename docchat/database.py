import sqlite3
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

class DocumentDatabase:
    """Handles SQLite database operations for tracking processed documents."""
    
    def __init__(self, db_path: str = "docchat.db"):
        self.db_path = db_path
        self.init_database()
    
    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection with sane defaults for concurrency."""
        # Increase timeout to wait for locks, allow cross-thread usage
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        try:
            # Reduce lock contention and improve concurrency
            conn.execute("PRAGMA busy_timeout=5000")
            # WAL can improve write/read concurrency; ignore if not supported
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            # If PRAGMAs fail, continue with defaults
            pass
        return conn
    
    def _retry_locked(self, func, *args, **kwargs):
        """Retry wrapper for handling 'database is locked' errors on writes."""
        max_attempts = 5
        delay = 0.2
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_attempts:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        
    def init_database(self):
        """Initialize the database with required tables."""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Create table for tracking processed files
        def _create():
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    content_hash TEXT NOT NULL,
                    last_modified TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

        try:
            self._retry_locked(_create)
        finally:
            conn.close()
    
    def get_processed_files(self) -> List[Tuple]:
        """Get all processed files from the database."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT file_path, content_hash, last_modified FROM documents")
        files = cursor.fetchall()
        conn.close()
        return files
    
    def add_file(self, file_path: str, content_hash: str, last_modified: datetime):
        """Add a new processed file to the database."""
        conn = self._connect()
        cursor = conn.cursor()
        def _insert():
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (file_path, content_hash, last_modified) 
                VALUES (?, ?, ?)
            ''', (file_path, content_hash, last_modified))
            conn.commit()
        try:
            self._retry_locked(_insert)
        except sqlite3.Error as e:
            print(f"Error adding file to database: {e}")
        finally:
            conn.close()
    
    def remove_file(self, file_path: str):
        """Remove a file from the database."""
        conn = self._connect()
        cursor = conn.cursor()
        def _delete():
            cursor.execute("DELETE FROM documents WHERE file_path = ?", (file_path,))
            conn.commit()
        try:
            self._retry_locked(_delete)
        except sqlite3.Error as e:
            print(f"Error removing file from database: {e}")
        finally:
            conn.close()

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get the content hash of a specific file."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT content_hash FROM documents WHERE file_path = ?", (file_path,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def get_file_last_modified(self, file_path: str) -> Optional[datetime]:
        """Get the last modified timestamp of a specific file."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT last_modified FROM documents WHERE file_path = ?", (file_path,))
        result = cursor.fetchone()
        conn.close()
        return datetime.fromisoformat(result[0]) if result else None

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file's content."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            return ""