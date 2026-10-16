"""One process of the checkpoint/state recovery experiment."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import threading
import time

from langgraph.checkpoint.sqlite import SqliteSaver

from coop.continuous_graph import build_continuous_graph
from coop.embedding import LocalQwenEmbedding
from coop.memory import MemoryAdapter, MemoryItem
from coop.protocol import ProtocolAdapter
from coop.qwen_agent import QwenCausalAdapter
from coop.state_store import SQLiteStateStore


QWEN = "/home/Users/SharedFile/Models/Qwen/Qwen3-8B"
EMBEDDING = "/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B"
QUERY = "Explain how persistent shared memory helps agents avoid repeating research."
EVIDENCE = "Persistent shared memory stores evidence and summaries so different agents can reuse prior work across tasks."
NAMESPACE = "recovery-v1-qwen3-embedding-768"
TASK_ID = "recovery-task"


class RecoveryLLM:
    """Calls are audited by role; any planner call after restart is an error."""

    def __init__(self, *, fake, phase, device, max_new_tokens, model_path=QWEN):
        self.fake, self.phase = fake, phase
        self.calls = []
        self.adapter = None if fake else QwenCausalAdapter(model_path, device=device, max_new_tokens=max_new_tokens)

    def generate_json(self, messages, *, max_new_tokens):
        if self.phase == "resume":
            raise AssertionError("planner was rerun after checkpoint recovery")
        self.calls.append("planner")
        if self.fake:
            return {"value": {"topic": "shared memory", "search_query": "persistent memory"}}
        return self.adapter.generate_json(messages, max_new_tokens=max_new_tokens)

    def generate(self, messages, *, max_new_tokens):
        self.calls.append("summarizer")
        if self.fake:
            return {"text": "Shared memory lets agents reuse prior evidence."}
        return self.adapter.generate(messages, max_new_tokens=max_new_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--phase", choices=("prepare", "resume"), required=True)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--wait-for-kill", action="store_true")
    parser.add_argument("--model-path", default=QWEN)
    parser.add_argument("--embedding-path", default=EMBEDDING)
    args = parser.parse_args()
    directory = Path(args.directory)
    started = time.perf_counter()
    llm = RecoveryLLM(fake=args.fake, phase=args.phase, device=args.device,
                      max_new_tokens=args.max_new_tokens, model_path=args.model_path)
    embedding_calls = []
    encode = None
    if args.phase == "prepare":
        encode = (lambda texts, query=False: [[1.0, 0.0] for _ in texts]) if args.fake else LocalQwenEmbedding(
            args.embedding_path, device=args.embedding_device, output_dim=768,
        ).encode

    def query_encoder(texts, *, query):
        if args.phase == "resume":
            raise AssertionError("query embedding was recomputed after recovery")
        embedding_calls.append("query")
        return encode(texts, query=query)

    with ExitStack() as stack:
        corpus = MemoryAdapter(str(directory / "corpus.sqlite"))
        stack.callback(corpus.close)
        memory = MemoryAdapter(str(directory / "memory.sqlite"))
        stack.callback(memory.close)
        states = stack.enter_context(SQLiteStateStore(str(directory / "states.sqlite"), namespace=NAMESPACE))
        saver = stack.enter_context(SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")))
        if args.phase == "prepare":
            vector = encode([EVIDENCE], query=False)[0]
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            corpus.put(MemoryItem("evidence-memory", "corpus", "", "shared memory", EVIDENCE,
                                  embedding=vector))
        transport = ProtocolAdapter()
        graph, _, events = build_continuous_graph(
            llm, corpus, memory, transport, encoder=query_encoder, reuse=True,
            namespace=NAMESPACE, state_store=states, checkpointer=saver,
            interrupt_after=["planner"] if args.phase == "prepare" else None,
            max_new_tokens=args.max_new_tokens,
        )
        config = {"configurable": {"thread_id": TASK_ID}}
        result = graph.invoke({"task_id": TASK_ID, "query": QUERY} if args.phase == "prepare" else None,
                              config, durability="sync")
        snapshot = graph.get_state(config)
        packet = states[result["state_id"]]
        info = {
            "phase": args.phase, "pid": os.getpid(), "fake": args.fake,
            "next_nodes": list(snapshot.next), "state_id": packet.state_id,
            "state_checksum": packet.checksum, "state_payload_bytes": packet.payload_bytes,
            "state_count": len(states), "calls": llm.calls, "query_embedding_calls": embedding_calls,
            "llm_usage": llm.adapter.call_stats if llm.adapter else [],
            "task_metrics": snapshot.values.get("task_metrics"),
            "memory_records": memory.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
            "protocol_frames": transport.frames,
            "evidence_id": result.get("evidence_id"), "answer": result.get("answer"),
            "elapsed_sec_including_load": time.perf_counter() - started,
        }
        if args.phase == "prepare" and snapshot.next != ("retriever",):
            raise AssertionError(f"unexpected checkpoint boundary: {snapshot.next}")
        if args.phase == "resume":
            if snapshot.next or not result.get("answer") or result.get("evidence_id") != "evidence-memory":
                raise AssertionError("recovery did not complete a grounded answer")
            count_before = len(llm.calls)
            graph.invoke(None, config, durability="sync")
            info["completed_reinvoke_calls"] = len(llm.calls) - count_before
        # Commit the audit marker before the controller terminates this process.
        marker = directory / (args.phase + ".json")
        with marker.open("w", encoding="utf-8") as file:
            json.dump(info, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        if args.wait_for_kill:
            # Keep live DB/model objects; controller sends SIGKILL, no graceful close.
            threading.Event().wait()
    print(json.dumps({"phase": args.phase, "pid": os.getpid(), "report": str(marker)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
