# Development Task Checklist – Local CLI Document Chat Application

## Setup & Environment
- [x] Install Python 3.11+
- [ ] Install Poetry
- [x] Create `pyproject.toml` with dependencies:
  - typer
  - pyyaml
  - sentence-transformers
  - chromadb
  - pypdf
  - llama-cpp-python
- [x] Configure `.gitignore` for `__pycache__`, `.venv`, `history/`, `vectorstore/`

## Configuration
- [x] Create `config.yaml` with:
  - docs_path
  - chunk_size / chunk_overlap
  - embedding_model
  - llm_model_path
  - top_k
- [x] Implement config loader utility.

## Database
- [x] Create SQLite schema for:
  - file path
  - hash
  - mtime
- [x] Implement DB helper functions.

## Document Processing
- [x] Implement file scanner.
- [x] Implement change detection (new, modified, deleted).
- [x] Implement per-format parsers:
  - txt
  - md
  - pdf
  - csv
- [x] Implement chunking function (RecursiveCharacterTextSplitter).

## Embedding & Storage
- [x] Load embedding model (`sentence-transformers`).
- [x] Initialize ChromaDB (persistent mode).
- [x] Store chunks with metadata.

## LLM Interface
- [x] Load local Mistral model via `llama-cpp-python`.
- [x] Implement prompt builder.
- [x] Implement query function to:
  - embed question
  - search vector DB
  - format prompt
  - get LLM response
  - return sources + answer.

## CLI Application
- [x] Implement CLI with Typer:
  - ingest command
  - chat command
- [x] Chat loop:
  - take input
  - process query
  - display response & sources
  - save to history file.

## Logging
- [x] Configure `logging` module.
- [x] Log INFO and ERROR messages.
- [x] Save to rotating log file.

## Testing
- [ ] Unit tests for:
  - file parsing
  - hashing
  - chunking
  - embedding & retrieval
- [ ] Manual test with mixed document set.

## Packaging & Delivery
- [ ] Test on Windows/macOS/Linux.
- [ ] Create README with usage instructions.
- [ ] Package for pip install.
