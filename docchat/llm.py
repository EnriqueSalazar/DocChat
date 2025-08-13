import logging
import os
import re
import json
import threading
from typing import Optional, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

class LocalLLM:
    """Wrapper for a RedPajama (HF Transformers) causal LM with structured JSON answer extraction."""
    
    def __init__(
        self,
        model_repo_or_path: str,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 512,
    ):
        """Initialize the local RedPajama model.

        Args:
            model_repo_or_path: HF repo id or local directory containing model files.
            device: 'cuda' or 'cpu'; auto-detected if None.
            dtype: Preferred dtype (bfloat16|float16|float32) depending on hardware.
            max_new_tokens: Generation token budget.
        """
        logger = logging.getLogger("DocChat.LLM")
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" and torch.cuda.is_available() else torch.float16 if dtype == "float16" and torch.cuda.is_available() else torch.float32
        logger.info("Loading RedPajama model repo=%s device=%s dtype=%s", model_repo_or_path, self.device, torch_dtype)

        def _do_load(target_device: str):
            tok = AutoTokenizer.from_pretrained(model_repo_or_path, use_fast=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_repo_or_path,
                torch_dtype=torch_dtype if target_device != 'cpu' else torch.float32,
                low_cpu_mem_usage=True,
            )
            if target_device == 'cuda':
                mdl.to('cuda')
            if target_device == 'cpu' and hasattr(torch, 'compile'):
                try:
                    mdl = torch.compile(mdl, mode='reduce-overhead')
                    logger.info('Applied torch.compile for CPU optimization.')
                except Exception:
                    pass
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok, mdl

        try:
            # Attempt GPU load if selected
            self.tokenizer, self.model = _do_load(self.device)
            # Quick capability check: attempt tiny forward pass; if fails fallback
            if self.device == 'cuda':
                try:
                    test_ids = self.tokenizer("hello", return_tensors="pt").to('cuda')
                    with torch.no_grad():
                        _ = self.model(**test_ids)
                except Exception as gpu_err:
                    logger.warning("CUDA model test failed (%s). Falling back to CPU.", gpu_err)
                    self.device = 'cpu'
                    self.tokenizer, self.model = _do_load('cpu')
            logger.info("Model loaded (device=%s).", self.device)
        except Exception as e:
            logger.exception("Failed to load RedPajama model; last attempt on device=%s", self.device)
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
        prompt = f"""You are a QA assistant. Use ONLY the provided context. Return JSON with keys: answer (string) and no_answer (bool).
If the context lacks the information, set no_answer true and answer be an empty string. Otherwise no_answer false and answer contains only the answer.
Context:\n{context}\nQuestion: {query}\nJSON:"""

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
            return self._post_process(generated)
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return "I encountered an error while generating a response."

    def stream_response(self, query: str, context: str) -> Generator[str, None, str]:
        """Stream tokens for a response; yields incremental text and returns final processed answer."""
        prompt = f"""You are a QA assistant. Use ONLY the provided context. Return JSON with keys: answer (string) and no_answer (bool).
If the context lacks the information, set no_answer true and answer be an empty string. Otherwise no_answer false and answer contains only the answer.
Context:\n{context}\nQuestion: {query}\nJSON:"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, streamer=streamer)

        def _generate():
            tried_fallback = False
            while True:
                try:
                    with torch.no_grad():
                        self.model.generate(**gen_kwargs)
                    break
                except RuntimeError as rte:
                    # Detect CUDA kernel errors and fallback to CPU once
                    if ("CUDA error" in str(rte) or "no kernel image" in str(rte)) and self.device == 'cuda' and not tried_fallback:
                        print("[LLM] CUDA generation error; falling back to CPU reload.")
                        self.device = 'cpu'
                        try:
                            # Reload on CPU
                            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer.name_or_path, use_fast=True)
                            self.model = AutoModelForCausalLM.from_pretrained(self.tokenizer.name_or_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
                            if hasattr(torch, 'compile'):
                                try:
                                    self.model = torch.compile(self.model, mode='reduce-overhead')
                                except Exception:
                                    pass
                            gen_inputs = self.tokenizer(prompt, return_tensors="pt").to('cpu')
                            gen_kwargs.update(gen_inputs)
                            gen_kwargs['streamer'] = streamer
                            tried_fallback = True
                            continue
                        except Exception as reload_err:
                            print(f"[LLM] CPU fallback reload failed: {reload_err}")
                            break
                    else:
                        break

        thread = threading.Thread(target=_generate)
        thread.start()
        collected = []
        for token_text in streamer:
            collected.append(token_text)
            yield token_text
        thread.join()
        full_text = "".join(collected)
        return self._post_process(full_text)

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

