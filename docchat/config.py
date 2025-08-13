from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    docs_path: Path = Path("./docs")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model_path: Optional[Path] = None
    llm_model_name: str = "togethercomputer/RedPajama-INCITE-7B-Instruct"
    llm_auto_download: bool = True
    llm_max_new_tokens: int = 384
    cpu_threads: int | None = None  # allow override of torch thread count
    top_k: int = 4
    vectorstore_path: Path = Path("./chromadb")
    history_dir: Path = Path("./history")

    @staticmethod
    def load(config_file: Path | str = "config.yaml") -> "Config":
        path = Path(config_file)
        cfg = Config()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Map YAML keys to dataclass fields if present
            if "docs_path" in data:
                cfg.docs_path = Path(data["docs_path"]).expanduser().resolve()
            if "chunk_size" in data:
                cfg.chunk_size = int(data["chunk_size"])
            if "chunk_overlap" in data:
                cfg.chunk_overlap = int(data["chunk_overlap"])
            if "embedding_model" in data:
                cfg.embedding_model = str(data["embedding_model"])  
            if "llm_model_path" in data and data["llm_model_path"]:
                cfg.llm_model_path = Path(data["llm_model_path"]).expanduser().resolve()
            if "llm_model_name" in data and data["llm_model_name"]:
                cfg.llm_model_name = str(data["llm_model_name"])  # HF repo id
            if "llm_auto_download" in data:
                cfg.llm_auto_download = bool(data["llm_auto_download"])
            if "llm_max_new_tokens" in data:
                cfg.llm_max_new_tokens = int(data["llm_max_new_tokens"])
            if "cpu_threads" in data and data["cpu_threads"] is not None:
                cfg.cpu_threads = int(data["cpu_threads"])
            if "top_k" in data:
                cfg.top_k = int(data["top_k"])
            if "vectorstore_path" in data:
                cfg.vectorstore_path = Path(data["vectorstore_path"]).expanduser().resolve()
            if "history_dir" in data:
                cfg.history_dir = Path(data["history_dir"]).expanduser().resolve()

        return cfg
