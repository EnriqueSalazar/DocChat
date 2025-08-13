import logging
from typing import List, Tuple
from .embeddings import EmbeddingGenerator, ChromaDBManager
from .chunking import TextChunker
from .llm import LocalLLM


class RAGPipeline:
    """Handles the Retrieval Augmented Generation pipeline."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chroma_persist_dir: str = "./chromadb",
        llm_model_path: str = "togethercomputer/RedPajama-INCITE-7B-Instruct",
        llm_max_new_tokens: int = 512,
        cpu_threads: int | None = None,
        enable_gpu: bool = False,
        context_max_chars: int | None = None,
        dynamic_quantization: bool = False,
        llm_compile: bool = False,
    ) -> None:
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.chroma_manager = ChromaDBManager(chroma_persist_dir)
        self.chunker = TextChunker()
        self.llm = LocalLLM(
            llm_model_path,
            max_new_tokens=llm_max_new_tokens,
            cpu_threads=cpu_threads,
            enable_gpu=enable_gpu,
            dynamic_quantization=dynamic_quantization,
            compile_model=llm_compile,
        )
        self.context_max_chars = context_max_chars
        self.adaptive_cfg = None
        self.adaptive_state = None

    def ask(self, query: str, top_k: int = 4) -> Tuple[str, List[str]]:
        import time
        log = logging.getLogger("DocChat.RAG")
        start_time = time.time()
        log.debug("ask() query='%s' top_k=%d", query, top_k)
        t0 = time.time()
        q_emb = self.embedding_generator.model.encode([query])[0].tolist()
        log.debug("Embedding computed in %.3fs", time.time() - t0)
        t1 = time.time()
        results = self.chroma_manager.query_by_embedding(q_emb, n_results=top_k)
        log.debug("Vector store query in %.3fs", time.time() - t1)
        if not results.get('documents') or not results['documents'][0]:
            return "I couldn't find any relevant information in your documents.", []
        context_chunks = results['documents'][0]
        log.debug("Retrieved %d context chunks", len(context_chunks))
        sources = set()
        if 'metadatas' in results and results['metadatas']:
            for metadata in results['metadatas'][0]:
                if 'file_path' in metadata:
                    sources.add(metadata['file_path'])
        context_text = "\n\n".join(context_chunks)
        if self.context_max_chars and len(context_text) > self.context_max_chars:
            context_text = context_text[: self.context_max_chars]
        log.debug("Context length chars=%d", len(context_text))
        t2 = time.time()
        result = self.llm.generate_response(query, context_text)
        gen_time = time.time() - t2
        latency = time.time() - start_time
        log.debug("Generation time %.3fs total latency %.3fs", gen_time, latency)
        if self.adaptive_cfg and self.adaptive_cfg.get('enable'):
            self._adaptive_update(latency)
        if result.get("no_answer"):
            return "I could not find an answer to that in the documents.", []
        return result.get("answer", ""), list(sources)

    def ask_stream(self, query: str, top_k: int = 4):
        """Stream an answer; yields (partial_text_or_final, sources|None)."""
        import time
        log = logging.getLogger("DocChat.RAG")
        log.debug("ask_stream() query='%s' top_k=%d", query, top_k)
        t0 = time.time()
        q_emb = self.embedding_generator.model.encode([query])[0].tolist()
        log.debug("Embedding computed in %.3fs", time.time() - t0)
        t1 = time.time()
        results = self.chroma_manager.query_by_embedding(q_emb, n_results=top_k)
        log.debug("Vector store query in %.3fs", time.time() - t1)
        if not results.get('documents') or not results['documents'][0]:
            yield "I couldn't find any relevant information in your documents.", []
            return
        context_chunks = results['documents'][0]
        log.debug("Retrieved %d context chunks", len(context_chunks))
        sources = set()
        if 'metadatas' in results and results['metadatas']:
            for metadata in results['metadatas'][0]:
                if 'file_path' in metadata:
                    sources.add(metadata['file_path'])
        context_text = "\n\n".join(context_chunks)
        if self.context_max_chars and len(context_text) > self.context_max_chars:
            context_text = context_text[: self.context_max_chars]
        log.debug("Context length chars=%d", len(context_text))
        streamer = self.llm.stream_response(query, context_text)
        t2 = time.time()
        try:
            for piece in streamer:
                if isinstance(piece, str):
                    yield piece, None
                else:
                    result = piece
                    log.debug("Generation streamed in %.3fs", time.time() - t2)
                    if result.get("no_answer"):
                        yield "", []
                    else:
                        yield result.get("answer", ""), list(sources)
        except Exception:
            yield "Error streaming response.", []

    def configure_adaptive(self, cfg: dict):
        self.adaptive_cfg = cfg
        self.adaptive_state = {"ema_latency": None}

    def _adaptive_update(self, latency: float):
        st = self.adaptive_state
        cfg = self.adaptive_cfg
        alpha = cfg['ema_alpha']
        st['ema_latency'] = latency if st['ema_latency'] is None else alpha * latency + (1 - alpha) * st['ema_latency']
        target = cfg['latency_target']
        current = self.llm.max_new_tokens
        changed = False
        if st['ema_latency'] > target * 1.1 and current > cfg['min_new_tokens']:
            new_val = max(cfg['min_new_tokens'], int(current * cfg['shrink_factor']))
            if new_val != current:
                self.llm.max_new_tokens = new_val
                changed = True
        elif st['ema_latency'] < target * 0.5 and current < cfg['max_new_tokens']:
            new_val = min(cfg['max_new_tokens'], int(current * cfg['growth_factor']))
            if new_val != current:
                self.llm.max_new_tokens = new_val
                changed = True
        if changed:
            logging.getLogger("DocChat.Adaptive").info(
                "Adaptive adjust max_new_tokens=%s (ema_latency=%.2fs target=%.2fs)",
                self.llm.max_new_tokens,
                st['ema_latency'],
                target,
            )
