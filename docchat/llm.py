import logging
import os
import re
from llama_cpp import Llama
try:
    # Capability flags exposed by llama-cpp-python if compiled with GPU backends
    from llama_cpp import LLAMA_CUBLAS, LLAMA_METAL, LLAMA_ROCM
except Exception:  # pragma: no cover
    LLAMA_CUBLAS = False
    LLAMA_METAL = False
    LLAMA_ROCM = False

class LocalLLM:
    """Wrapper for a local GGUF-based Large Language Model."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int | None = None,
        n_threads: int | None = None,
        n_gpu_layers: int | None = None,
    ):
        """
        Initialize the local LLM. 
        
        Args:
            model_path (str): Path to the GGUF model file
            n_ctx (int): Context size for the model
            n_threads (int): Number of threads to use for generation
        """
        logger = logging.getLogger("DocChat.LLM")

        # Auto-tune defaults if not provided
        auto_threads = max(1, min((os.cpu_count() or 4), 8))
        threads = n_threads if n_threads is not None else auto_threads
        ctx = n_ctx if n_ctx is not None else 2048

        gpu_supported = bool(LLAMA_CUBLAS or LLAMA_ROCM or LLAMA_METAL)
        # Offload all layers if GPU is available, otherwise CPU-only (0)
        offload_layers = n_gpu_layers if n_gpu_layers is not None else (-1 if gpu_supported else 0)

        logger.info(
            "llama.cpp backends: CUBLAS=%s, ROCm=%s, METAL=%s", LLAMA_CUBLAS, LLAMA_ROCM, LLAMA_METAL
        )
        try:
            logger.info(
                "Loading local LLM (llama-cpp): model_path=%s, n_ctx=%s, n_threads=%s, n_gpu_layers=%s, use_mmap=%s, verbose=%s",
                model_path,
                ctx,
                threads,
                offload_layers,
                True,
                False,
            )
            self.llm = Llama(
                model_path=model_path,
                n_ctx=ctx,
                n_threads=threads,
                n_gpu_layers=offload_layers,
                use_mmap=True,
                verbose=False,
            )
            logger.info("Local LLM loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load local LLM.")
            raise RuntimeError(f"Failed to load LLM model from {model_path}: {e}")
            
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response from the LLM based on a query and context.
        
        Args:
            query (str): The user's question
            context (str): The context retrieved from documents
            
        Returns:
            str: The generated answer
        """
        prompt = f"""
You are a helpful assistant that answers questions using ONLY the context.

Rules:
1. If (and only if) the context does NOT contain the information needed, output exactly:
[NO_ANSWER]
2. Otherwise output ONLY the answer (no preamble, no mention of the rules, do NOT repeat [NO_ANSWER]).
3. Be concise but complete.

Context:
---
{context}
---

Question: {query}

Answer:
"""

        try:
            response = self.llm(
                prompt,
                max_tokens=512,
                stop=["Question:", "User:"],
                echo=False,
            )
            answer = response['choices'][0]['text'].strip()

            # Normalize common misspelling variants of the no-answer token produced by model
            upper = answer.upper().strip()
            variants = {"[NO_ANSWER]", "[NO_ANSWAN]", "[NO_ANSWE]"}
            if upper in variants:
                return "[NO_ANSWER]"

            # If a variant token appears alongside other content, strip it out
            token_pattern = re.compile(r"\[NO_[A-Z]+\]")
            if token_pattern.search(upper):
                # Remove only exact tokens, leave surrounding explanatory text
                cleaned = token_pattern.sub("", answer).strip()
                if cleaned:
                    answer = cleaned
            # Also handle the canonical [NO_ANSWER] embedded in longer text
            if "[NO_ANSWER]" in answer and len(answer) > len("[NO_ANSWER]"):
                cleaned2 = answer.replace("[NO_ANSWER]", "").strip()
                if cleaned2:
                    answer = cleaned2
            return answer
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return "I encountered an error while generating a response."

