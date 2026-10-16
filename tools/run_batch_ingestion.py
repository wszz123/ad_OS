"""Paired ingestion benchmark with unchanged SQLite durability settings."""

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
import sqlite3
import time
from uuid import uuid4

import torch

from coop.embedding import LocalQwenEmbedding
from coop.heldout_workload import heldout_tasks
from coop.qwen_agent import QwenCausalAdapter
from coop.sqlite_indexed_memory import IndexedSQLiteMemoryAdapter
from tools.run_cost_workload import EMBEDDING, QWEN, cost_namespace, run_cost_mode
from tools.run_heldout_workload import load_cached_documents, save_progress


STRATEGIES = ("per_item", "batch")


def digest_records(items):
    values = sorted((asdict(item) for item in items), key=lambda value: value["memory_id"])
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def measure_ingestion(docs, directory, strategy):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "corpus.sqlite"
    store = IndexedSQLiteMemoryAdapter(str(path))
    try:
        durability = {"journal_mode": store.connection.execute("PRAGMA journal_mode").fetchone()[0],
                      "synchronous": store.connection.execute("PRAGMA synchronous").fetchone()[0]}
        started = time.perf_counter()
        if strategy == "per_item":
            ids = [store.put(replace(item)) for item in docs]
        elif strategy == "batch":
            ids = store.put_many(replace(item) for item in docs)
        else:
            raise ValueError(f"unknown ingestion strategy: {strategy}")
        elapsed = time.perf_counter() - started
        counts = store.write_stats()
        actual_digest = digest_records(store.items())
    finally:
        store.close()
    reopened = IndexedSQLiteMemoryAdapter(str(path))
    try:
        reopened_digest = digest_records(reopened.items())
        entity_doc = next(item for item in docs if item.memory_id == "orion-price")
        queries = {entity: [result.item.memory_id for result in reopened.search_vector(entity_doc.embedding, tags=["entity:" + entity], limit=10)]
                   for entity in ("Orion", "Lyra")}
        count = reopened.count()
    finally:
        reopened.close()
    checks = {"source_records_preserved": actual_digest == digest_records(docs),
              "reopen_records_preserved": actual_digest == reopened_digest,
              "ordered_ids_preserved": ids == [item.memory_id for item in docs],
              "expected_record_count": count == len(docs),
              "expected_commit_count": counts["commits"] == (len(docs) if strategy == "per_item" else 1)}
    return {"elapsed_sec": elapsed, "write_stats": counts, "durability": durability,
            "record_digest": actual_digest, "reopened_queries": queries, "checks": checks,
            "artifact_directory": str(directory)}


def workflow_checks(results):
    a, b = results["per_item"], results["batch"]
    task_keys = ("task_id", "status", "expression", "tool_result", "answer", "quality",
                 "reused_memory_id", "accepted_reuse", "plan_attempts")
    metric_keys = ("model_calls", "prompt_tokens", "generated_tokens", "message_tokens", "total_transfer_bytes", "accepted_reuse")
    return {"tasks_equal": [{key: row.get(key) for key in task_keys} for row in a["tasks"]]
                           == [{key: row.get(key) for key in task_keys} for row in b["tasks"]],
            "invariant_metrics_equal": all(a[key] == b[key] for key in metric_keys),
            "both_quality_passed": all(result["passed_count"] == result["task_count"] for result in results.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", default="results/heldout_workload.json")
    parser.add_argument("--corpus-size", type=int, default=137)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--workflow", action="store_true", help="Also run paired real-model task trials")
    parser.add_argument("--device")
    parser.add_argument("--embedding-device")
    parser.add_argument("--report", default="results/batch_ingestion.json")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    started = time.perf_counter()
    source = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
    cells = [cell for cell in source["cells"] if cell["corpus_size"] == args.corpus_size]
    if len(cells) != 1:
        parser.error("source report needs one matching corpus")
    cell = cells[0]
    docs = load_cached_documents(source, cell)
    target = Path(args.report)
    directory = target.parent / (target.stem + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    report = {"experiment": "sqlite_atomic_batch_ingestion", "config": vars(args), "corpus_size": len(docs),
              "source_config": source["config"], "namespace": cell["metadata"]["namespace"],
              "sqlite_version": sqlite3.sqlite_version,
              "artifact_directory": str(directory), "ingestion_trials": [], "workflow_trials": [],
              "limitations": ["Write timings use cached document vectors and include row preparation/serialization, transaction and commit.",
                              "Schema/index setup and reopen checks are outside the timed ingestion interval.",
                              "No relaxed durability, deleted journal or disabled fsync is used.",
                              "Database rollback does not undo Python item mutations or external embedding callbacks.",
                              "Batching only initial corpus ingestion; per-task memories remain individually persisted."]}
    for trial in range(args.repeat):
        order = list(STRATEGIES) if trial % 2 == 0 else list(reversed(STRATEGIES))
        values = {}
        for strategy in order:
            values[strategy] = measure_ingestion(docs, directory / f"ingest-{trial}-{strategy}", strategy)
            report["pending_ingestion"] = {"index": trial, "strategies": values}
            save_progress(target, report)
        checks = {"same_durability": values["per_item"]["durability"] == values["batch"]["durability"],
                  "same_reopened_queries": values["per_item"]["reopened_queries"] == values["batch"]["reopened_queries"],
                  "all_record_checks_passed": all(all(value["checks"].values()) for value in values.values())}
        report["ingestion_trials"].append({"order": order, "strategies": values, "checks": checks})
        del report["pending_ingestion"]
        save_progress(target, report)
    summary = {strategy: {"elapsed_sec": mean(trial["strategies"][strategy]["elapsed_sec"] for trial in report["ingestion_trials"]),
                           "commits": mean(trial["strategies"][strategy]["write_stats"]["commits"] for trial in report["ingestion_trials"])}
               for strategy in STRATEGIES}
    report["ingestion_summary"] = summary
    print(json.dumps({"ingestion_summary": summary}, ensure_ascii=False, indent=2), flush=True)
    if args.workflow:
        config = source["config"]
        tasks = heldout_tasks(config["seed"], config["rounds"])
        if json.loads(json.dumps([asdict(task) for task in tasks])) != cell["metadata"]["tasks"]:
            raise ValueError("frozen task manifest mismatch")
        llm = QwenCausalAdapter(QWEN, device=args.device or config["device"], max_new_tokens=config["max_new_tokens"])
        encoder = LocalQwenEmbedding(EMBEDDING, device=args.embedding_device or config["embedding_device"], output_dim=768)
        namespace = cost_namespace(docs, instruction=encoder.instruction, repair_plan=True, max_plan_repairs=config["max_plan_repairs"])
        if namespace != cell["metadata"]["namespace"]:
            raise ValueError("source/model/policy fingerprint mismatch")
        warmup = time.perf_counter()
        llm.load()
        encoder.encode(["Warmup ingestion comparison"], query=True)
        llm.generate([{"role": "user", "content": "Say ready."}], max_new_tokens=4)
        report["model_initialization_sec"] = time.perf_counter() - warmup
        report["environment"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "langgraph")}
        report["runtime"] = {"model": QWEN, "embedding_model": EMBEDDING,
                              "device": str(llm.device), "embedding_device": str(encoder.device),
                              "generator_dtype": str(llm.model.dtype), "embedding_dtype": str(encoder._model.dtype),
                              "cpu_threads": torch.get_num_threads()}
        for trial in range(args.repeat):
            order = list(STRATEGIES) if trial % 2 == 0 else list(reversed(STRATEGIES))
            values = {}
            for strategy in order:
                values[strategy] = run_cost_mode(llm, encoder.encode, docs, tasks,
                    directory / f"workflow-{trial}-{strategy}", "protocol_vector_memory", namespace=namespace,
                    threshold=config["threshold"], max_new_tokens=config["max_new_tokens"], durable=True,
                    repair_plan=True, max_plan_repairs=config["max_plan_repairs"],
                    storage_profile="optimized", ingest_strategy=strategy)
                report["pending_workflow"] = {"index": trial, "strategies": values}
                save_progress(target, report)
            report["workflow_trials"].append({"order": order, "strategies": values, "checks": workflow_checks(values)})
            del report["pending_workflow"]
            save_progress(target, report)
    report["all_checks_passed"] = all(all(trial["checks"].values()) for trial in report["ingestion_trials"] + report["workflow_trials"])
    report["total_sec"] = time.perf_counter() - started
    save_progress(target, report)
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "ingestion_summary": summary,
                      "workflow_trials": len(report["workflow_trials"]), "report": str(target)}, ensure_ascii=False, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit("batch ingestion comparison failed; see report")


if __name__ == "__main__":
    main()
