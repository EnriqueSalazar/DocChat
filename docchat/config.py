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
    enable_gpu: bool = False  # allow optional GPU usage
    context_max_chars: int = 6000  # trim combined retrieved context to this many characters
    cpu_dynamic_quantization: bool = False  # apply torch dynamic quantization on CPU linear layers
    # Adaptive generation settings
    adaptive_enable: bool = False
    adaptive_latency_target: float = 4.0  # seconds target end-to-end answer latency
    adaptive_min_new_tokens: int = 64
    adaptive_max_new_tokens: int = 512
    adaptive_growth_factor: float = 1.2
    adaptive_shrink_factor: float = 0.7
    adaptive_ema_alpha: float = 0.4
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
            if "enable_gpu" in data:
                cfg.enable_gpu = bool(data["enable_gpu"])
            if "context_max_chars" in data:
                cfg.context_max_chars = int(data["context_max_chars"])
            if "cpu_dynamic_quantization" in data:
                cfg.cpu_dynamic_quantization = bool(data["cpu_dynamic_quantization"])
            if "adaptive_enable" in data:
                cfg.adaptive_enable = bool(data["adaptive_enable"])
            if "adaptive_latency_target" in data:
                cfg.adaptive_latency_target = float(data["adaptive_latency_target"])
            if "adaptive_min_new_tokens" in data:
                cfg.adaptive_min_new_tokens = int(data["adaptive_min_new_tokens"])
            if "adaptive_max_new_tokens" in data:
                cfg.adaptive_max_new_tokens = int(data["adaptive_max_new_tokens"])
            if "adaptive_growth_factor" in data:
                cfg.adaptive_growth_factor = float(data["adaptive_growth_factor"])
            if "adaptive_shrink_factor" in data:
                cfg.adaptive_shrink_factor = float(data["adaptive_shrink_factor"])
            if "adaptive_ema_alpha" in data:
                cfg.adaptive_ema_alpha = float(data["adaptive_ema_alpha"])
            if "top_k" in data:
                cfg.top_k = int(data["top_k"])
            if "vectorstore_path" in data:
                cfg.vectorstore_path = Path(data["vectorstore_path"]).expanduser().resolve()
            if "history_dir" in data:
                cfg.history_dir = Path(data["history_dir"]).expanduser().resolve()

        return cfg
