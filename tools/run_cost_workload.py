"""Evaluate four-agent, multi-source tasks with numerical and citation grading."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from uuid import uuid4

import torch

from langgraph.checkpoint.memory import InMemorySaver

from coop.answer_validation import validate_cost_answer
from coop.continuous_graph import state_sizes
from coop.cost_graph import PLAN_PROMPT, REVIEW_PROMPT, SUMMARY_PROMPT, build_cost_graph, public_request
from coop.embedding import LocalQwenEmbedding
from coop.memory import MemoryAdapter
from coop.memory_backends import MeasuredMemoryBackend, create_memory_backend
from coop.protocol import ProtocolAdapter
from coop.qwen_agent import QwenCausalAdapter
from coop.state_store import SQLiteStateStore
from coop.sqlite_indexed_memory import IndexedSQLiteMemoryAdapter
from coop.repair_ledger import RepairLedger
from coop.text_transport import TextAdapter
from coop.workload import cost_documents, cost_tasks


QWEN = "/home/Users/SharedFile/Models/Qwen/Qwen3-8B"
EMBEDDING = "/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B"
MODES = ("text", "protocol", "protocol_vector", "protocol_vector_memory")
STORAGE_PROFILES = {
    "reference": {"tag_pushdown": False, "metadata_projection": False, "deduplicate_validation": False},
    "tag_pushdown": {"tag_pushdown": True, "metadata_projection": False, "deduplicate_validation": False},
    "metadata_projection": {"tag_pushdown": True, "metadata_projection": True, "deduplicate_validation": False},
    "optimized": {"tag_pushdown": True, "metadata_projection": True, "deduplicate_validation": True},
}


def run_cost_mode(llm, encoder, docs, tasks, directory, mode, *, namespace,
                  threshold=0.55, max_new_tokens=192, durable=False, review_plan=False,
                  repair_plan=False, max_plan_repairs=2, memory_backend="sqlite",
                  storage_profile="reference", ingest_strategy="per_item"):
    if storage_profile not in STORAGE_PROFILES:
        raise ValueError(f"unknown storage profile: {storage_profile}")
    if ingest_strategy not in ("per_item", "batch"):
        raise ValueError(f"unknown ingest strategy: {ingest_strategy}")
    profile = STORAGE_PROFILES[storage_profile]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rows, traces = [], []
    with ExitStack() as resources:
        corpus_class = IndexedSQLiteMemoryAdapter if profile["tag_pushdown"] else MemoryAdapter
        corpus = corpus_class(str(directory / "corpus.sqlite"))
        raw_memory = create_memory_backend(memory_backend, path=directory / "memory.sqlite",
                                           namespace=("latentmas", namespace))
        memory = MeasuredMemoryBackend(raw_memory)
        repair_ledger = resources.enter_context(RepairLedger(str(directory / "repair_ledger.sqlite"))) if repair_plan else None
        resources.callback(corpus.close)
        resources.callback(memory.close)
        ingest_started = time.perf_counter()
        if ingest_strategy == "batch":
            corpus.put_many(docs)
        else:
            for item in docs:
                corpus.put(item)
        corpus_ingest_sec = time.perf_counter() - ingest_started
        corpus_write_stats = corpus.write_stats()
        saver, states = InMemorySaver(), None
        if durable:
            from langgraph.checkpoint.sqlite import SqliteSaver

            saver = resources.enter_context(SqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")))
            states = resources.enter_context(SQLiteStateStore(str(directory / "states.sqlite"), namespace=namespace))
        transport = TextAdapter() if mode == "text" else ProtocolAdapter()
        graph, packets, events = build_cost_graph(
            llm, corpus, memory, transport, namespace=namespace,
            encoder=encoder if "vector" in mode else None,
            reuse=mode == "protocol_vector_memory", threshold=threshold,
            max_new_tokens=max_new_tokens, state_store=states, checkpointer=saver,
            review_plan=review_plan,
            repair_plan=repair_plan, max_plan_repairs=max_plan_repairs,
            metadata_projection=profile["metadata_projection"], deduplicate_validation=profile["deduplicate_validation"],
            repair_ledger=repair_ledger,
        )
        for task in tasks:
            if mode == "text":
                transport.begin_task()
            first_call, first_frame = len(llm.call_stats), len(transport.frames)
            scans_before = corpus.scan_stats()
            row = {"task_id": task.task_id, "entity": task.entity, "question": task.question,
                   "inputs": task.inputs, "expected_value": task.expected_value,
                   "required_citations": list(task.required_citations)}
            started = time.perf_counter()
            try:
                result = graph.invoke(public_request(task), {"configurable": {"thread_id": task.task_id}}, durability="sync")
                answer, tool = json.loads(result["answer_json"]), json.loads(result["tool_json"])
                row.update(status="ok", expression=result["expression"], tool_result=tool, answer=answer,
                           documents=json.loads(result["documents_json"]), reused_memory_id=result["reused_memory_id"],
                           quality=validate_cost_answer(task, answer, tool))
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}",
                           quality={"passed": False, "numeric_correct": False, "unit_correct": False,
                                    "citations_correct": False, "tool_numeric_correct": False})
            row["elapsed_sec"] = time.perf_counter() - started
            calls, frames = llm.call_stats[first_call:], transport.frames[first_frame:]
            packet = packets.get(task.task_id + ":facts-query")
            payload, metadata = state_sizes(packet) if packet else (0, 0)
            row.update(model_calls=len(calls), prompt_tokens=sum(call["prompt_tokens"] for call in calls),
                       generated_tokens=sum(call["generated_tokens"] for call in calls), message_count=len(frames),
                       message_tokens=sum(len(llm.tokenizer.encode(frame, add_special_tokens=False)) for frame in frames),
                       text_chars=sum(len(frame) for frame in frames),
                       message_bytes=sum(len(frame.encode("utf-8")) for frame in frames),
                       state_transfers=int(packet is not None), state_payload_bytes=payload,
                       state_metadata_bytes=metadata, **events.get(task.task_id, {}))
            row["total_transfer_bytes"] = row["message_bytes"] + payload + metadata
            row["corpus_scan_stats"] = {key: value - scans_before[key] for key, value in corpus.scan_stats().items()}
            rows.append(row)
            traces.append({"task_id": task.task_id, "calls": calls, "frames": frames})
            print(f"{mode} {task.task_id}: {row['status']}, quality={row['quality']['passed']}, reuse={row.get('accepted_reuse', 0)}", flush=True)
        backend_stats = memory.stats()
        corpus_scans = corpus.scan_stats()
        records = [asdict(item) for item in raw_memory.items()]
        count = len(records)
        # Snapshot exports are audit artifacts, not an InMemoryStore durability claim.
        (directory / "memory_snapshot.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "trace.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    totals = {key: sum(row.get(key, 0) for row in rows) for key in (
        "model_calls", "prompt_tokens", "generated_tokens", "message_count", "message_tokens", "text_chars",
        "message_bytes", "state_transfers", "state_payload_bytes", "state_metadata_bytes", "total_transfer_bytes",
        "memory_queries", "accepted_reuse", "corpus_searches", "source_parses", "cache_validation_parses",
        "tool_calls", "plan_reviews", "plan_repairs", "plan_attempt_count", "plan_repair_exhausted",
        "embedding_sec", "corpus_search_sec", "memory_lookup_sec", "elapsed_sec",
    )}
    totals.update(task_count=len(tasks), completed=sum(row["status"] == "ok" for row in rows),
                  new_memory_records=count, roles=["planner", "retriever", "executor", "summarizer"],
                  state_backend="sqlite" if durable else "memory", memory_backend=memory_backend,
                  memory_persistent=memory_backend == "sqlite", memory_backend_operations=backend_stats,
                  corpus_scan_stats=corpus_scans,
                  storage_profile=storage_profile, storage_settings=profile,
                  corpus_index_setup_sec=getattr(corpus, "index_setup_sec", 0.0),
                  corpus_ingest_sec=corpus_ingest_sec,
                  corpus_ingest_strategy=ingest_strategy, corpus_write_stats=corpus_write_stats,
                  tasks=rows)
    for key in ("passed", "numeric_correct", "unit_correct", "citations_correct", "tool_numeric_correct"):
        totals[key + "_count"] = sum(row["quality"][key] for row in rows)
    totals["quality_pass_rate"] = totals["passed_count"] / len(tasks)
    totals["accepted_reuse_rate"] = totals["accepted_reuse"] / totals["memory_queries"] if totals["memory_queries"] else 0
    return totals


def prepare_cost_runtime(*, device, embedding_device, max_new_tokens,
                         review_plan=False, repair_plan=False, max_plan_repairs=2,
                         docs=None, tasks=None, embedding_batch_size=None,
                         llm=None, embedding=None):
    """Shared model/corpus preparation for mode and backend paired experiments."""
    started = time.perf_counter()
    llm = QwenCausalAdapter(QWEN, device=device, max_new_tokens=max_new_tokens) if llm is None else llm
    embedding = LocalQwenEmbedding(EMBEDDING, device=embedding_device, output_dim=768) if embedding is None else embedding
    docs = cost_documents() if docs is None else docs
    tasks = cost_tasks() if tasks is None else tasks
    if not docs or not tasks or (embedding_batch_size is not None and embedding_batch_size < 1):
        raise ValueError("nonempty corpus/tasks and positive embedding batch size are required")
    llm.load()
    prep = time.perf_counter()
    populate_embeddings(docs, embedding.encode, batch_size=embedding_batch_size or len(docs))
    embedding.encode(["Orion cost"], query=True)
    llm.generate([{"role": "user", "content": "Say ready."}], max_new_tokens=4)
    namespace = cost_namespace(docs, instruction=embedding.instruction, review_plan=review_plan,
                               repair_plan=repair_plan, max_plan_repairs=max_plan_repairs,
                               model_path=llm.model_path, embedding_path=embedding.model_path)
    metadata = {
        "model": llm.model_path, "embedding_model": embedding.model_path, "namespace": namespace,
        "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "langgraph")},
        "runtime": {"generator_dtype": str(llm.model.dtype), "embedding_dtype": str(embedding._model.dtype),
                    "cpu_threads": torch.get_num_threads(), "decoding": "greedy", "thinking": False},
        "qwen_load_sec": llm.loaded_at, "shared_preparation_sec": time.perf_counter() - prep,
        "initialization_sec": time.perf_counter() - started,
        "embedding_batch_size": embedding_batch_size or len(docs),
        "tasks": [asdict(task) for task in tasks], "documents": [
            {key: value for key, value in asdict(item).items() if key != "embedding"} for item in docs],
    }
    return llm, embedding, docs, tasks, metadata


def cost_namespace(docs, *, instruction, review_plan=False, repair_plan=False, max_plan_repairs=2,
                   model_path=QWEN, embedding_path=EMBEDDING):
    """Stable policy/source/vector fingerprint, also checked by experiment resume."""
    return hashlib.sha256(json.dumps({"sources": [asdict(item) for item in docs],
        "embedding": embedding_path, "model": model_path, "instruction": instruction,
        "planner_prompt": PLAN_PROMPT, "summary_prompt": SUMMARY_PROMPT,
        "review_prompt": REVIEW_PROMPT if review_plan or repair_plan else None,
        "repair_policy": {"max_repairs": max_plan_repairs, "version": "quantity-contract-v1"} if repair_plan else None,
        "tool_version": "decimal-v1"}, sort_keys=True).encode()).hexdigest()


def populate_embeddings(docs, encoder, *, batch_size):
    if batch_size < 1:
        raise ValueError("embedding batch size must be positive")
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        vectors = encoder([item.summary for item in batch])
        if len(vectors) != len(batch):
            raise ValueError("embedding count does not match document batch")
        for item, vector in zip(batch, vectors):
            item.embedding = vector.tolist() if hasattr(vector, "tolist") else list(vector)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--durable", action="store_true")
    parser.add_argument("--review-plan", action="store_true", help="Extra planner review with unit/input checks; every call is counted")
    parser.add_argument("--repair-plan", action="store_true", help="Bounded plan corrections using public unit and quantity contracts")
    parser.add_argument("--max-plan-repairs", type=int, default=2, help="Maximum correction calls after the draft, in [1,5]")
    parser.add_argument("--memory-backend", choices=("sqlite", "langgraph"), default="sqlite")
    parser.add_argument("--storage-profile", choices=tuple(STORAGE_PROFILES), default="reference")
    parser.add_argument("--ingest-strategy", choices=("per_item", "batch"), default="per_item")
    parser.add_argument("--report", default="results/cost_workload.json")
    args = parser.parse_args()
    if args.repeat < 1 or args.max_new_tokens < 1 or not -1 <= args.threshold <= 1 or not 1 <= args.max_plan_repairs <= 5:
        parser.error("invalid repeat, generation budget or reuse threshold")
    if args.review_plan and args.repair_plan:
        parser.error("--review-plan and --repair-plan are mutually exclusive")
    target = Path(args.report)
    directory = target.parent / (target.stem + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    started = time.perf_counter()
    llm, embedding, docs, tasks, prepared = prepare_cost_runtime(
        device=args.device, embedding_device=args.embedding_device, max_new_tokens=args.max_new_tokens,
        review_plan=args.review_plan, repair_plan=args.repair_plan, max_plan_repairs=args.max_plan_repairs,
    )
    namespace = prepared["namespace"]
    report = {"experiment": "four_role_sourced_cost_workload", "synthetic": True, "config": vars(args),
              **prepared,
              "artifact_directory": str(directory), "trials": [],
              "limitations": ["Synthetic cost-analysis fixtures, not an external benchmark.",
                              "The entity and input parameters are part of the user request; gold never enters agents.",
                              "Text baseline is labelled text with actual task handoff history, not an LLM parser.",
                              "Grading covers numeric value, unit, tool correctness and source IDs; explanation prose is not graded.",
                              "Cached facts are revalidated against current sources; parsing overhead is counted.",
                              "Bounded repair uses public domain contracts, no gold amounts or citation labels; all attempts are retained.",
                              "Changed task parameters require fresh planning and execution; memory does not bypass these calls."]}
    if args.memory_backend == "langgraph":
        report["limitations"].append("LangGraph InMemoryStore is process-local; durable state/checkpoints do not make task memory durable.")
    modes = args.mode or list(MODES)
    for trial in range(args.repeat):
        order = modes if trial % 2 == 0 else list(reversed(modes))
        data = {mode: run_cost_mode(llm, embedding.encode, docs, tasks, directory / f"trial-{trial}-{mode}", mode,
                                   namespace=namespace, threshold=args.threshold, max_new_tokens=args.max_new_tokens,
                                   durable=args.durable, review_plan=args.review_plan,
                                   repair_plan=args.repair_plan, max_plan_repairs=args.max_plan_repairs,
                                    memory_backend=args.memory_backend, storage_profile=args.storage_profile,
                                    ingest_strategy=args.ingest_strategy) for mode in order}
        report["trials"].append({"order": order, "modes": data})
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["total_sec_including_preparation"] = time.perf_counter() - started
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for index, trial in enumerate(report["trials"]):
        print(json.dumps({"trial": index, "modes": {mode: {key: value for key, value in row.items() if key != "tasks"}
                                                    for mode, row in trial["modes"].items()}}, ensure_ascii=False, indent=2))
    print(f"report={target}")


if __name__ == "__main__":
    main()
