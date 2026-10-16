"""Run one three-role LangGraph task with the local Qwen3-8B model."""

import argparse
import json
import time

from langgraph.checkpoint.memory import InMemorySaver

from coop.memory import MemoryAdapter, MemoryItem
from coop.embedding import LocalQwenEmbedding
from coop.protocol import ProtocolAdapter
from coop.qwen_agent import QwenCausalAdapter
from coop.qwen_graph import build_qwen_graph


MODEL = "/home/Users/SharedFile/Models/Qwen/Qwen3-8B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-device", default="cuda:1")
    parser.add_argument("--without-state-exchange", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--query", default="Explain how persistent shared memory helps multiple agents avoid repeating research.")
    args = parser.parse_args()

    whole_smoke_started = time.perf_counter()
    query_encoder = None
    state_packets = {}
    if not args.without_state_exchange:
        query_encoder = LocalQwenEmbedding(
            "/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B",
            device=args.embedding_device, output_dim=768,
        ).encode
        seed_vector = query_encoder([
            "Persistent shared memory stores evidence and summaries so different agents can reuse prior work across tasks."
        ])[0].tolist()
    else:
        seed_vector = None
    memory = MemoryAdapter()
    memory.put(MemoryItem(
        "seed-memory", "retriever", "2026-01-01T00:00:00+00:00", "shared memory",
        "Persistent shared memory stores evidence and summaries so different agents can reuse prior work across tasks.",
        tags=["agents", "memory"],
        embedding=seed_vector,
    ))
    llm = QwenCausalAdapter(MODEL, device=args.device, max_new_tokens=args.max_new_tokens)
    protocol = ProtocolAdapter()
    graph = build_qwen_graph(llm, memory, protocol, max_new_tokens=args.max_new_tokens,
                             query_encoder=query_encoder, state_packets=state_packets,
                             checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "qwen-smoke-1"}}
    started = time.perf_counter()
    result = graph.invoke({"task_id": "qwen-smoke-1", "query": args.query}, config)
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "model": MODEL,
        "query": args.query,
        "plan": result["plan"],
        "evidence_id": result["evidence_id"],
        "answer": result["answer"],
        "agent_calls": llm.call_stats,
        "prompt_tokens": sum(call["prompt_tokens"] for call in llm.call_stats),
        "generated_tokens": sum(call["generated_tokens"] for call in llm.call_stats),
        "protocol_frames": len(protocol.frames),
        "protocol_chars": sum(len(frame) for frame in protocol.frames),
        "state_transfer_count": len(state_packets),
        "state_payload_bytes": sum(packet.payload_bytes for packet in state_packets.values()),
        "state_encoding": "float16" if state_packets else None,
        "graph_elapsed_sec_including_qwen_load": round(elapsed, 3),
        "total_smoke_sec_including_both_model_loads": round(time.perf_counter() - whole_smoke_started, 3),
        "model_load_sec": round(llm.loaded_at or 0, 3),
        "checkpoint_has_answer": bool(graph.get_state(config).values.get("answer")),
    }, ensure_ascii=False, indent=2))
    memory.close()


if __name__ == "__main__":
    main()
