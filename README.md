# DocChat

Local RAG (Retrieval Augmented Generation) CLI that ingests your documents and lets you chat with them using the RedPajama INCITE 7B Instruct model.

## Features
- Document ingestion with automatic change detection (new / modified / deleted)
- Persistent local vector store (ChromaDB)
- SentenceTransformer embeddings
- RedPajama model (Hugging Face Transformers) with structured JSON answers
- Automatic first-time model download (no HF auth required for public files)
- CPU-first optimizations (focus on fast generation after initial load)

## Quick Start
```
python -m docchat.main ingest  # ingest docs then chat
python -m docchat.main chat    # chat (after prior ingestion)
```

Default documents folder: `./docs`

## Configuration (`config.yaml`)
```yaml
docs_path: ./docs
chunk_size: 1000
chunk_overlap: 200
embedding_model: all-MiniLM-L6-v2
llm_model_name: togethercomputer/RedPajama-INCITE-7B-Instruct
llm_auto_download: true  # set false to rely on HF cache manually
top_k: 4
vectorstore_path: ./chromadb
history_dir: ./history
```

If `llm_auto_download` is true and `./model/redpajama` lacks shards, they are downloaded directly from:
https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Instruct

Files fetched:
```
config.json
generation_config.json
pytorch_model-00001-of-00003.bin
pytorch_model-00002-of-00003.bin
pytorch_model-00003-of-00003.bin
tokenizer.json
tokenizer_config.json
special_tokens_map.json
vocab.json
merges.txt
```

## Structured Answers
The model is prompted to emit JSON with:
```
{"answer": "...", "no_answer": false}
```
If `no_answer` is true (or `[NO_ANSWER]` appears), the UI shows that no answer was found in your documents.

## Performance
- Prioritizes chat speed over slightly longer initial load.
- CPU: attempts torch.compile (PyTorch 2+) for faster subsequent generations.
- GPU (if available): moves model to CUDA automatically.

## Roadmap
- Optional quantization (4/8-bit)
- Streaming tokens
- Web UI

## License
See `LICENSE`.