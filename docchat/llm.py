import logging
import re
import json
import threading
import warnings
from typing import Optional, Generator, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

class LocalLLM:
    """Wrapper for a RedPajama (HF Transformers) causal LM with structured JSON answer extraction.

    Supports optional GPU (enable_gpu=True) else defaults to CPU-only optimized path.
    """

    def __init__(self, model_repo_or_path: str, max_new_tokens: int = 512, cpu_threads: Optional[int] = None, enable_gpu: bool = False, dynamic_quantization: bool = False, compile_model: bool = False):
        """Initialize the local RedPajama model.

        Args:
            model_repo_or_path: HF repo id or local directory containing model files.
            max_new_tokens: Generation token budget.
            cpu_threads: Optional thread override for torch intra/inter-op.
            enable_gpu: Whether to attempt GPU usage if available.
            dynamic_quantization: Apply torch dynamic quantization to Linear layers (CPU only) for faster inference.
            compile_model: Use torch.compile (if available) for potential speedup.
        """
        self.logger = logging.getLogger("DocChat.LLM")
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if enable_gpu and torch.cuda.is_available() else "cpu"
        warnings.filterwarnings("ignore", message=".*not compatible with the current PyTorch installation.*", category=UserWarning)

        if cpu_threads and cpu_threads > 0 and self.device == 'cpu':
            try:
                torch.set_num_threads(cpu_threads)
                if hasattr(torch, "set_num_interop_threads"):
                    torch.set_num_interop_threads(max(1, cpu_threads // 2))
                self.logger.info("Set torch CPU threads=%s interop=%s", cpu_threads, max(1, cpu_threads // 2))
            except Exception as e:
                self.logger.warning("Could not set CPU threads (%s)", e)

        self.logger.info("Loading RedPajama model repo=%s device=%s", model_repo_or_path, self.device)

        def _do_load():
            tok = AutoTokenizer.from_pretrained(model_repo_or_path, use_fast=True)
            torch_dtype = torch.float32
            if self.device == "cuda":
                if torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float16
            mdl = AutoModelForCausalLM.from_pretrained(
                model_repo_or_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=(self.device == 'cpu'),
            )
            if self.device == 'cuda':
                mdl.to('cuda')
            else:
                if dynamic_quantization:
                    try:
                        mdl = torch.quantization.quantize_dynamic(mdl, {torch.nn.Linear}, dtype=torch.qint8)
                        self.logger.info('Applied dynamic quantization (int8) to Linear layers.')
                    except Exception as e:
                        self.logger.warning('Dynamic quantization failed: %s', e)
                if compile_model and hasattr(torch, 'compile'):
                    try:
                        mdl = torch.compile(mdl, mode='reduce-overhead')
                        self.logger.info('Applied torch.compile for CPU optimization.')
                    except Exception as e:
                        self.logger.warning('torch.compile failed: %s', e)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok, mdl

        try:
            self.tokenizer, self.model = _do_load()
            self.logger.info("Model loaded (device=%s).", self.device)
        except Exception as e:
            self.logger.exception("Failed to load RedPajama model")
            raise RuntimeError(f"Failed to load model {model_repo_or_path}: {e}")
            
    def generate_response(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate a response from the LLM based on a query and context.
        
        Args:
            query (str): The user's question
            context (str): The context retrieved from documents
            
        Returns:
            str: The generated answer
        """
        prompt = (
            "You are a STRICT extractive QA assistant. You MUST use only the facts found in the provided context CHUNKs. "
            "If the answer is not explicitly stated or directly inferable from the context, respond with JSON {\"answer\": \"\", \"no_answer\": true}. "
            "Do NOT add outside knowledge, do NOT guess, do NOT fabricate numbers, dates, or names. Paraphrasing is allowed but must preserve meaning. "
            "Return ONLY a single line of compact JSON with exactly the keys: answer (string) and no_answer (bool). No prose before or after.\n\n"
            f"Context:\n{context}\nQuestion: {query}\nJSON:"
        )

        try:
            import time
            t0 = time.time()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            self.logger.debug("Tokenized prompt len=%d device=%s", inputs["input_ids"].shape[1], self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            self.logger.debug("Generation produced %d tokens in %.3fs", output_ids.shape[1] - inputs["input_ids"].shape[1], time.time() - t0)
            generated = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            data = self._post_process(generated)
            if not data["no_answer"] and not self._validate_against_context(data["answer"], context):
                data["no_answer"] = True
                data["answer"] = ""
            return data
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return {"answer": "", "no_answer": True, "error": str(e)}

    def stream_response(self, query: str, context: str) -> Generator[str, None, Dict[str, Any]]:
        """Stream tokens for a response; yields incremental text and returns final processed answer."""
        prompt = (
            "You are a STRICT extractive QA assistant. You MUST use only the facts found in the provided context CHUNKs. "
            "If the answer is not explicitly stated or directly inferable from the context, respond with JSON {\"answer\": \"\", \"no_answer\": true}. "
            "Do NOT add outside knowledge, do NOT guess, do NOT fabricate numbers, dates, or names. Paraphrasing is allowed but must preserve meaning. "
            "Return ONLY a single line of compact JSON with exactly the keys: answer (string) and no_answer (bool). No prose before or after.\n\n"
            f"Context:\n{context}\nQuestion: {query}\nJSON:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.logger.debug("[stream] Tokenized prompt len=%d", inputs["input_ids"].shape[1])
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, streamer=streamer)

        def _generate():
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        import time
        start_time = time.time()
        thread = threading.Thread(target=_generate)
        thread.start()
        collected = []
        for token_text in streamer:
            collected.append(token_text)
            yield token_text
        thread.join()
        self.logger.debug("[stream] Generation finished in %.3fs; produced %d tokens", time.time() - start_time, len(collected))
        full_text = "".join(collected)
        data = self._post_process(full_text)
        if not data["no_answer"] and not self._validate_against_context(data["answer"], context):
            data["no_answer"] = True
            data["answer"] = ""
        # Yield final structured result as a dict (ask_stream detects non-str and treats as final)
        yield data

    def _post_process(self, generated: str) -> Dict[str, Any]:
        """Extract JSON; return dict {answer, no_answer}."""
        match = re.search(r"\{.*\}", generated, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict) and "no_answer" in data and "answer" in data:
                    return {"answer": str(data.get("answer", "")).strip(), "no_answer": bool(data.get("no_answer"))}
            except Exception:
                pass
        # Fallback heuristic: if model didn't return JSON, treat as answer text
        cleaned = generated.strip()
        if not cleaned or len(cleaned) < 2:
            return {"answer": "", "no_answer": True}
        # If model leaked sentinel
        if "[NO_ANSWER]" in cleaned.upper():
            return {"answer": "", "no_answer": True}
        return {"answer": cleaned, "no_answer": False}

    def _validate_against_context(self, answer: str, context: str) -> bool:
        """Heuristic validation: ensure majority of content words appear in context to reduce hallucinations."""
        if not answer.strip():
            return False
        # Normalize
        def norm(txt: str) -> str:
            return re.sub(r"[^a-z0-9 ]", " ", txt.lower())
        a = norm(answer)
        c = norm(context)
        ctx_tokens = set(c.split())
        ans_tokens = [t for t in a.split() if len(t) > 3]
        if not ans_tokens:
            return True
        hits = sum(1 for t in ans_tokens if t in ctx_tokens)
        ratio = hits / max(1, len(ans_tokens))
        return ratio >= 0.55

