import tempfile
import unittest

import numpy as np
import torch

from coop.embedding import last_token_pool
from coop.memory import MemoryAdapter, MemoryItem
from coop.state_exchange import StateExchange


class AdapterTests(unittest.TestCase):
    def test_pooling_handles_left_and_right_padding(self):
        states = torch.arange(24).reshape(2, 4, 3)
        masks = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
        pooled = last_token_pool(states, masks)
        self.assertTrue(torch.equal(pooled[0], states[0, 3]))
        self.assertTrue(torch.equal(pooled[1], states[1, 1]))

    def test_keyword_search_filters_and_misses(self):
        store = MemoryAdapter()
        store.put(MemoryItem("m1", "retriever", "", "cache", "GPU cache", tags=["performance"]))
        self.assertEqual(store.search("unrelated"), [])
        self.assertEqual(store.search("GPU", tags=["other"]), [])
        self.assertEqual(store.search("GPU", tags=["performance"])[0].item.memory_id, "m1")
        store.close()

    def test_persistent_memory_keyword_and_semantic_search(self):
        def embed(texts):
            vectors = []
            for text in texts:
                lower = text.lower()
                vectors.append([float("gpu" in lower), float("cache" in lower), float("memory" in lower)])
            return vectors

        with tempfile.NamedTemporaryFile(suffix=".sqlite") as file:
            store = MemoryAdapter(file.name, embedder=embed)
            store.put(MemoryItem("m1", "retriever", "2026-01-01", "GPU cache", "KV cache reduced latency", tags=["performance"]))
            store.put(MemoryItem("m2", "planner", "2026-01-01", "memory", "Shared memory supports reuse", tags=["memory"]))
            results = store.search("GPU cache reuse", limit=1)
            self.assertEqual(results[0].item.memory_id, "m1")
            self.assertEqual(results[0].match_type, "semantic")
            store.close()
            reopened = MemoryAdapter(file.name, embedder=embed)
            self.assertEqual(reopened.search("GPU cache", limit=1)[0].item.memory_id, "m1")
            reopened.close()

    def test_state_exchange_round_trip_and_size(self):
        vector = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
        exchange = StateExchange()
        for encoding in ("float32", "float16", "int8"):
            packet = exchange.encode(vector, state_id="s1", source_agent="a", target_agent="b", encoding=encoding)
            restored = exchange.decode(packet)
            tolerance = 1e-3 if encoding != "int8" else 1.0 / 80
            self.assertTrue(np.allclose(vector, restored, atol=tolerance))
        self.assertLess(
            exchange.encode(vector, state_id="s1", source_agent="a", target_agent="b", encoding="int8").payload_bytes,
            exchange.encode(vector, state_id="s1", source_agent="a", target_agent="b", encoding="float32").payload_bytes,
        )


if __name__ == "__main__":
    unittest.main()
