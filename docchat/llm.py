import logging
import os
import re
import json
import threading
import warnings
from typing import Optional, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

class LocalLLM:
    """Wrapper for a RedPajama (HF Transformers) causal LM with structured JSON answer extraction (CPU-only)."""

    def __init__(
        self,
        model_repo_or_path: str,
        max_new_tokens: int = 512,
        cpu_threads: Optional[int] = None,
    ):
        """Initialize the local RedPajama model (forced CPU-only).

        Args:
            model_repo_or_path: HF repo id or local directory containing model files.
            max_new_tokens: Generation token budget.
            cpu_threads: Optional thread override for torch intra/inter-op.
        """
        self.logger = logging.getLogger("DocChat.LLM")
        self.max_new_tokens = max_new_tokens
        self.device = "cpu"
        # Suppress noisy build capability warnings irrelevant to CPU-only mode
        warnings.filterwarnings(
            "ignore",
            message=".*not compatible with the current PyTorch installation.*",
            category=UserWarning,
        )

        if cpu_threads and cpu_threads > 0:
            try:
                torch.set_num_threads(cpu_threads)
                if hasattr(torch, "set_num_interop_threads"):
                    torch.set_num_interop_threads(max(1, cpu_threads // 2))
                self.logger.info("Set torch CPU threads=%s interop=%s", cpu_threads, max(1, cpu_threads // 2))
            except Exception as e:
                self.logger.warning("Could not set CPU threads (%s)", e)

        self.logger.info("Loading RedPajama model (CPU-only) repo=%s", model_repo_or_path)

        def _do_load():
            tok = AutoTokenizer.from_pretrained(model_repo_or_path, use_fast=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_repo_or_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            if hasattr(torch, 'compile'):
                try:
                    mdl = torch.compile(mdl, mode='reduce-overhead')
                    self.logger.info('Applied torch.compile for CPU optimization.')
                except Exception:
                    pass
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok, mdl

        try:
            self.tokenizer, self.model = _do_load()
            self.logger.info("Model loaded (CPU).")
        except Exception as e:
            self.logger.exception("Failed to load RedPajama model on CPU")
            raise RuntimeError(f"Failed to load model {model_repo_or_path}: {e}")
            
    def generate_response(self, query: str, context: str) -> str:
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
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            answer = self._post_process(generated)
            if answer != "[NO_ANSWER]" and not self._validate_against_context(answer, context):
                return "[NO_ANSWER]"
            return answer
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return "I encountered an error while generating a response."

    def stream_response(self, query: str, context: str) -> Generator[str, None, str]:
        """Stream tokens for a response; yields incremental text and returns final processed answer."""
        prompt = (
            "You are a STRICT extractive QA assistant. You MUST use only the facts found in the provided context CHUNKs. "
            "If the answer is not explicitly stated or directly inferable from the context, respond with JSON {\"answer\": \"\", \"no_answer\": true}. "
            "Do NOT add outside knowledge, do NOT guess, do NOT fabricate numbers, dates, or names. Paraphrasing is allowed but must preserve meaning. "
            "Return ONLY a single line of compact JSON with exactly the keys: answer (string) and no_answer (bool). No prose before or after.\n\n"
            f"Context:\n{context}\nQuestion: {query}\nJSON:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, streamer=streamer)

        def _generate():
            # Single attempt generation (CPU-only path eliminates CUDA fallback complexity)
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        thread = threading.Thread(target=_generate)
        thread.start()
        collected = []
        for token_text in streamer:
            collected.append(token_text)
            yield token_text
        thread.join()
        full_text = "".join(collected)
        answer = self._post_process(full_text)
        if answer != "[NO_ANSWER]" and not self._validate_against_context(answer, context):
            return "[NO_ANSWER]"
        return answer

    def _post_process(self, generated: str) -> str:
        """Convert raw generated text (possibly with JSON) into final answer or [NO_ANSWER]."""
        match = re.search(r"\{.*\}", generated, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict) and "no_answer" in data and "answer" in data:
                    if data.get("no_answer"):
                        return "[NO_ANSWER]"
                    return str(data.get("answer", "")).strip()
            except Exception:
                pass
        if "[NO_ANSWER]" in generated.upper():
            return "[NO_ANSWER]"
        return generated.strip()

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

