"""Real-model four-mode paired ablation with task-created persistent memory."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from coop.continuous_graph import build_continuous_graph, state_sizes
from coop.continuous_tasks import continuous_tasks, evidence_items
from coop.embedding import LocalQwenEmbedding
from coop.memory import MemoryAdapter, MemoryItem
from coop.protocol import ProtocolAdapter
from coop.qwen_agent import QwenCausalAdapter
from coop.state_store import SQLiteStateStore
from coop.text_transport import TextAdapter


QWEN = "/home/Users/SharedFile/Models/Qwen/Qwen3-8B"
EMBEDDING = "/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B"
MODES = ("text", "protocol", "protocol_vector", "protocol_vector_memory")


def experiment_evidence():
    """Known references plus distractors; expected IDs are evaluator-only labels."""
    return evidence_items() + [
        MemoryItem("evidence-kv", "corpus", "", "KV cache",
                   "KV caches retain attention keys and values during autoregressive decoding to avoid recomputing previous tokens."),
        MemoryItem("evidence-checkpoint", "corpus", "", "checkpoints",
                   "A checkpoint saves a single task thread's graph state for resumption; it is distinct from cross-thread shared knowledge."),
        MemoryItem("evidence-protocol", "corpus", "", "communication protocol",
                   "Agent protocols contain actions, input parameters, returned results, capabilities and a version for discovery and interoperability."),
        MemoryItem("evidence-sandbox", "corpus", "", "code execution",
                   "Code execution tools run bounded Python tasks with timeouts and process isolation; tool output is returned to the requesting agent."),
    ]


def run_mode(llm, encoder, items, tasks, directory, mode, *, namespace,
             threshold, margin, max_new_tokens, durable=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    corpus = MemoryAdapter(str(directory / "corpus.sqlite"))
    memory = MemoryAdapter(str(directory / "task_memory.sqlite"))
    rows, traces = [], []
    resources = ExitStack()
    try:
        for item in items:
            corpus.put(item)
        transport = TextAdapter() if mode == "text" else ProtocolAdapter()
        states = None
        saver = InMemorySaver()
        if durable:
            from langgraph.checkpoint.sqlite import SqliteSaver

            states = resources.enter_context(SQLiteStateStore(str(directory / "states.sqlite"), namespace=namespace))
            saver = resources.enter_context(SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")))
        graph, packets, events = build_continuous_graph(
            llm, corpus, memory, transport,
            encoder=encoder if "vector" in mode else None,
            reuse=mode == "protocol_vector_memory", namespace=namespace,
            threshold=threshold, margin=margin, max_new_tokens=max_new_tokens,
            checkpointer=saver, state_store=states,
        )
        for task in tasks:
            if isinstance(transport, TextAdapter):
                transport.begin_task()
            calls_before, frames_before = len(llm.call_stats), len(transport.frames)
            started = time.perf_counter()
            config = {"configurable": {"thread_id": task.task_id}}
            row = {"task_id": task.task_id, "group": task.group, "query": task.query,
                   "expected_evidence_id": task.expected_memory_id}
            try:
                result = graph.invoke({"task_id": task.task_id, "query": task.query}, config, durability="sync")
                row.update({"status": "ok", "plan": {"topic": result["topic"], "search_query": result["search_query"]},
                            "evidence_id": result["evidence_id"], "evidence": result["evidence"],
                            "answer": result["answer"], "reused_memory_id": result["reused_memory_id"],
                            "evidence_correct": result["evidence_id"] == task.expected_memory_id,
                            "checkpoint_has_answer": bool(graph.get_state(config).values.get("answer"))})
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}", evidence_correct=False)
            # Includes graph, tokenization, online embeddings, DB reads/writes and generation.
            row["elapsed_sec"] = time.perf_counter() - started
            calls = llm.call_stats[calls_before:]
            frames = transport.frames[frames_before:]
            packet = packets.get(task.task_id + ":query")
            payload_bytes, metadata_bytes = state_sizes(packet) if packet else (0, 0)
            row.update({"model_calls": len(calls),
                        "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
                        "generated_tokens": sum(call["generated_tokens"] for call in calls),
                        "message_count": len(frames), "text_chars": sum(len(frame) for frame in frames),
                        "message_utf8_bytes": sum(len(frame.encode("utf-8")) for frame in frames),
                        "message_tokens": sum(len(llm.tokenizer.encode(frame, add_special_tokens=False)) for frame in frames),
                        "state_transfers": int(packet is not None), "state_payload_bytes": payload_bytes,
                        "state_metadata_bytes": metadata_bytes, **events.get(task.task_id, {})})
            row["total_transfer_bytes"] = row["message_utf8_bytes"] + payload_bytes + metadata_bytes
            rows.append(row)
            traces.append({"task_id": task.task_id, "frames": frames, "calls": calls})
            print(f"{mode} {task.task_id}: {row['status']}, calls={len(calls)}, reuse={row.get('accepted_reuse', 0)}", flush=True)
        memory_count = memory.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        resources.close()
        memory.close()
        corpus.close()
    # Audit both traces and the memory database; no temporary-directory destruction.
    (directory / "trace.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {key: sum(row.get(key, 0) for row in rows) for key in (
        "model_calls", "prompt_tokens", "generated_tokens", "message_count", "text_chars",
        "message_utf8_bytes", "message_tokens", "state_transfers", "state_payload_bytes",
        "state_metadata_bytes", "total_transfer_bytes", "memory_queries", "accepted_reuse",
        "corpus_searches", "embedding_sec", "memory_write_sec", "elapsed_sec",
    )}
    metrics.update({"task_count": len(tasks), "completed": sum(row["status"] == "ok" for row in rows),
                    "evidence_correct": sum(row["evidence_correct"] for row in rows),
                    "evidence_accuracy_proxy": sum(row["evidence_correct"] for row in rows) / len(tasks),
                    "new_memory_records": memory_count,
                    "accepted_reuse_rate": metrics["accepted_reuse"] / metrics["memory_queries"] if metrics["memory_queries"] else 0,
                    "state_backend": "sqlite" if durable else "memory",
                    "checkpoint_backend": "sqlite" if durable else "memory",
                    "tasks": rows})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--repeat", type=int, default=1, help="Independent trials with empty memory each time")
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--durable", action="store_true", help="SQLite state packets and synchronous graph checkpoints")
    parser.add_argument("--report", default="results/qwen_continuous_v2.json")
    args = parser.parse_args()
    if args.repeat < 1 or args.max_new_tokens < 1 or not -1 <= args.threshold <= 1 or args.margin < 0:
        parser.error("invalid repeat, generation budget or reuse gate")
    target = Path(args.report)
    # Each invocation retains a unique audit directory; existing results are preserved.
    run_directory = target.parent / (target.stem + "_" + uuid4().hex[:8])
    run_directory.mkdir(parents=True)
    whole_started = time.perf_counter()
    tasks, items = continuous_tasks(), experiment_evidence()
    llm = QwenCausalAdapter(QWEN, device=args.device, max_new_tokens=args.max_new_tokens)
    encoder_adapter = LocalQwenEmbedding(EMBEDDING, device=args.embedding_device, output_dim=768)
    llm.load()
    preparation_started = time.perf_counter()
    vectors = encoder_adapter.encode([item.summary for item in items])
    for item, vector in zip(items, vectors):
        item.embedding = vector.tolist()
    # Shared warmup is excluded from paired task time, and explicitly timed.
    encoder_adapter.encode(["Warmup retrieval"], query=True)
    llm.generate([{"role": "user", "content": "Say ready."}], max_new_tokens=4)
    preparation_sec = time.perf_counter() - preparation_started
    namespace = hashlib.sha256(json.dumps({
        "corpus": [asdict(item) for item in items], "embedding": EMBEDDING,
        "dimension": 768, "instruction": encoder_adapter.instruction, "generator": QWEN,
    }, sort_keys=True).encode()).hexdigest()
    report = {"experiment": "real_qwen_continuity_v2", "config": vars(args),
              "qwen_model": QWEN, "embedding_model": EMBEDDING,
              "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "langgraph")},
              "corpus": [asdict(item) for item in items], "namespace": namespace,
              "qwen_load_sec": llm.loaded_at, "shared_preparation_and_warmup_sec": preparation_sec,
              "artifact_directory": str(run_directory), "trials": [],
              "limitations": ["Evidence ID accuracy is not answer correctness; read task answers and sources.",
                              "Text mode is a lossless labelled-text codec with real handoff history, not an LLM natural-language parser.",
                              "Protocol improvements measure actual wire encoding; identical LLM prompts isolate transport.",
                               ("Binary packets and checkpoints use SQLite; this continuous run does not inject failures."
                                if args.durable else "Binary transfer uses an in-process registry; crash recovery is tested separately."),
                              "Threshold/margin are fixed configuration, not tuned with evaluator labels."]}
    modes = args.mode or list(MODES)
    for trial in range(args.repeat):
        # Reverse mode order every other trial to expose ordering bias.
        order = modes if trial % 2 == 0 else list(reversed(modes))
        measurements = {}
        for mode in order:
            measurements[mode] = run_mode(
                llm, encoder_adapter.encode, items, tasks, run_directory / f"trial-{trial}-{mode}", mode,
                namespace=namespace, threshold=args.threshold, margin=args.margin,
                max_new_tokens=args.max_new_tokens,
                durable=args.durable,
            )
        if "protocol_vector" in measurements and "protocol_vector_memory" in measurements:
            baseline = measurements["protocol_vector"]
            reuse = measurements["protocol_vector_memory"]
            reuse["model_calls_saved_vs_vector"] = baseline["model_calls"] - reuse["model_calls"]
            reuse["corpus_searches_saved_vs_vector"] = baseline["corpus_searches"] - reuse["corpus_searches"]
            reuse["prompt_token_reduction_vs_vector_pct"] = 100 * (1 - reuse["prompt_tokens"] / baseline["prompt_tokens"]) if baseline["prompt_tokens"] else None
        report["trials"].append({"order": order, "modes": measurements})
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["total_sec_including_preparation"] = time.perf_counter() - whole_started
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for index, trial in enumerate(report["trials"]):
        print(json.dumps({"trial": index, "modes": {
            mode: {key: value for key, value in data.items() if key != "tasks"}
            for mode, data in trial["modes"].items()}}, ensure_ascii=False, indent=2))
    print(f"report={target}")


if __name__ == "__main__":
    main()
