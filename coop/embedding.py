"""Local Qwen embedding adapter with no dependency on sentence-transformers."""

from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool Qwen3 embeddings at the final non-padding token."""
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    sequence_lengths = positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(dim=1).values
    if (sequence_lengths < 0).any():
        raise ValueError("cannot pool an empty sequence")
    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device),
        sequence_lengths,
    ]


class LocalQwenEmbedding:
    """Lazy local Transformers adapter for Qwen3-Embedding.

    The model is loaded only on the first call, so protocol and memory unit tests
    remain fast and do not require GPU memory.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = torch.bfloat16,
        max_length: int = 2048,
        output_dim: Optional[int] = 768,
        instruction: str = "Represent the query for retrieving relevant shared memories",
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_length = max_length
        self.output_dim = output_dim
        self.instruction = instruction
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, padding_side="left"
        )
        kwargs = {"local_files_only": True}
        if self.device.type == "cpu":
            kwargs["dtype"] = torch.float32
        elif self.dtype is not None:
            kwargs["dtype"] = self.dtype
        self._model = AutoModel.from_pretrained(self.model_path, **kwargs)
        self._model.to(self.device)
        self._model.eval()

    def _prepare(self, texts: Iterable[str], query: bool) -> List[str]:
        values = list(texts)
        if query:
            return [f"Instruct: {self.instruction}\nQuery: {text}" for text in values]
        return values

    @torch.inference_mode()
    def encode(self, texts: Iterable[str], *, query: bool = False) -> torch.Tensor:
        self._load()
        values = self._prepare(texts, query)
        if not values:
            dimension = self.output_dim or self._model.config.hidden_size
            return torch.empty((0, dimension), dtype=torch.float32)
        encoded = self._tokenizer(
            values,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self._model(**encoded)
        vectors = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
        vectors = F.normalize(vectors.float(), p=2, dim=1)
        if self.output_dim is not None:
            if not 32 <= self.output_dim <= vectors.shape[1]:
                raise ValueError(f"output_dim must be in [32, {vectors.shape[1]}]")
            vectors = vectors[:, : self.output_dim]
            vectors = F.normalize(vectors, p=2, dim=1)
        return vectors.cpu()
