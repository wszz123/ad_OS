"""Small local causal-model adapter for LangGraph agent nodes."""

import json
import re
import time
from typing import Dict, List, Optional

import torch


class QwenCausalAdapter:
    """Lazy Qwen3 adapter with deterministic generation and usage accounting."""

    def __init__(self, model_path: str, *, device: str = "cuda:0",
                 max_new_tokens: int = 128, dtype: Optional[torch.dtype] = torch.bfloat16,
                 model=None, tokenizer=None) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.model = model
        self.tokenizer = tokenizer
        self.loaded_at = None
        self.call_stats: List[Dict] = []

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, use_fast=True
        )
        kwargs = {"local_files_only": True}
        if self.device.type == "cpu":
            kwargs["dtype"] = torch.float32
        elif self.dtype is not None:
            kwargs["dtype"] = self.dtype
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.loaded_at = time.perf_counter() - started

    def generate(self, messages: List[Dict[str, str]], *, max_new_tokens: Optional[int] = None) -> Dict:
        self.load()
        started = time.perf_counter()
        kwargs = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        if hasattr(encoded, "input_ids") or isinstance(encoded, dict):
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        else:
            input_ids = encoded.to(self.device)
            attention_mask = torch.ones_like(input_ids)
        prompt_tokens = int(attention_mask.sum().item())
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        generated_ids = output[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        generated_tokens = int(generated_ids.numel())
        stats = {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
        self.call_stats.append(stats)
        return {"text": text, **stats}

    def generate_json(self, messages: List[Dict[str, str]], *,
                      max_new_tokens: Optional[int] = None) -> Dict:
        result = self.generate(messages, max_new_tokens=max_new_tokens)
        text = result["text"]
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError(f"model did not return a JSON object: {text!r}")
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("model output must be a JSON object")
        return {**result, "value": value}
