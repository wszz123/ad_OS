import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from coop.embedding import LocalQwenEmbedding
from coop.qwen_agent import QwenCausalAdapter


class CpuModelLoadingTests(unittest.TestCase):
    def test_bfloat16_checkpoint_loads_and_runs_in_float32_on_cpu(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=32)
        with tempfile.TemporaryDirectory() as directory:
            tiny = Qwen3ForCausalLM(config).to(torch.bfloat16)
            tiny.save_pretrained(directory)
            del tiny
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=SimpleNamespace(pad_token_id=0)):
                causal = QwenCausalAdapter(directory, device="cpu")
                causal.load()
                embedding = LocalQwenEmbedding(directory, device="cpu", output_dim=32)
                embedding._load()
            tokens = torch.tensor([[1, 2]])
            with torch.inference_mode():
                logits = causal.model(input_ids=tokens, use_cache=False).logits
                hidden = embedding._model(input_ids=tokens, use_cache=False).last_hidden_state
            self.assertEqual(next(causal.model.parameters()).dtype, torch.float32)
            self.assertEqual(next(embedding._model.parameters()).dtype, torch.float32)
            self.assertEqual(logits.dtype, torch.float32)
            self.assertEqual(hidden.dtype, torch.float32)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(hidden).all())


if __name__ == "__main__":
    unittest.main()
