"""Run the real local embedding model once.

Usage: conda run -n latentmas python -m tools.smoke_qwen_embedding
"""

import json
import time
import tempfile
from pathlib import Path

import numpy as np

from coop.embedding import LocalQwenEmbedding
from coop.memory import MemoryAdapter, MemoryItem
from coop.state_exchange import StateExchange


MODEL = "/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B"


def main() -> None:
    started = time.perf_counter()
    embedder = LocalQwenEmbedding(MODEL, device="cuda", output_dim=768)
    documents = [
        "GPU KV cache stores attention keys and values to avoid repeated computation during autoregressive inference.",
        "Shared agent memory stores evidence and strategies so later tasks can reuse previous results.",
    ]
    vectors = embedder.encode(documents)
    queries = ["How can decoding avoid recomputing attention?",
               "How can another agent reuse previous evidence?"]
    query_vectors = embedder.encode(queries, query=True).numpy()
    exchange = StateExchange()
    records = []
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "memories.sqlite")
        writer = MemoryAdapter(path)
        for index, document in enumerate(documents):
            writer.put(MemoryItem(
                f"m{index}", "retriever", "", f"group-{index}", document,
                embedding=vectors[index].tolist(),
            ))
        writer.close()
        for index in range(10):
            group = index % 2
            packet = exchange.encode(
                query_vectors[group], state_id=f"s{index}", source_agent="planner",
                target_agent="retriever", encoding="float16",
            )
            restored = exchange.decode(packet)
            # A different agent reopens durable memory and uses the received vector.
            reader = MemoryAdapter(path)
            matches = reader.search_vector(restored, limit=1)
            reader.close()
            baseline = int(np.argmax(vectors.numpy() @ query_vectors[group]))
            records.append({
                "round": index + 1, "group": group, "expected_memory": f"m{group}",
                "actual_memory": matches[0].item.memory_id,
                "correct": matches[0].item.memory_id == f"m{group}",
                "ranking_preserved": matches[0].item.memory_id == f"m{baseline}",
                "state_bytes": packet.payload_bytes,
                "max_restore_error": float(np.max(np.abs(restored - query_vectors[group]))),
            })
    print(json.dumps({
        "model": MODEL, "shape": list(vectors.shape), "rounds": records,
        "retrieval_accuracy": sum(row["correct"] for row in records) / len(records),
        "state_transfers": len(records),
        "state_payload_bytes": sum(row["state_bytes"] for row in records),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "scope": "Two-document smoke check; not a full multi-agent benchmark",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
