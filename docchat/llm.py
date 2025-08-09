import logging
import os
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
        Instructions:
        - You are a helpful assistant that answers questions based ONLY on the provided context.
        - Your answers should be concise and directly address the user's question.
        - If the answer to the question is not present in the context, you MUST return the exact string "[NO_ANSWER]" and nothing else.
        - Do not use any knowledge outside of the provided context.

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
                # Avoid stopping on every newline; only stop on clear section markers if they appear
                stop=["Question:", "User:"],
                echo=False
            )
            
            # Extract the text from the response
            answer = response['choices'][0]['text'].strip()
            return answer
            
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return "I encountered an error while generating a response."

