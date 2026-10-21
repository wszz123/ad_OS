"""Run two related five-task chains against a local executable service exercise."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from uuid import uuid4

from coop.devops_graph import QwenDevOps, ScriptedDevOps, build_devops_graph
from coop.devops_world import DevOpsWorld, tasks
from coop.memory import MemoryAdapter, MemoryItem
from coop.protocol import ProtocolAdapter
from coop.text_transport import TextAdapter


MODES = ("text", "protocol", "protocol_vector", "protocol_vector_group_memory",
         "protocol_vector_memory")


def fixture_embedding(texts, *, query=False):
    """Fixed topic vectors solely for transport/retrieval wiring, not semantic evidence."""
    return [[1.0, 0.5, 0.25, 0.125] for _ in texts]


def run_sequence(directory, *, mode="protocol_vector_memory", decision=None,
                 encoder=fixture_embedding, max_steps=12, backend="fixture", on_update=None,
                 workload="service", task_limit=10):
    if mode not in MODES:
        raise ValueError("unknown mode")
    if type(task_limit) is not int or not 1 <= task_limit <= 10:
        raise ValueError("task_limit must be an integer between 1 and 10")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    # Refuse to mix business states, memories and reports from different experiments.
    if any(root.iterdir()):
        raise ValueError("sequence directory must be empty; choose a fresh output directory")
    run_id = uuid4().hex
    if workload == "project":
        from coop.catalog_world import CatalogProjectWorld, ScriptedProject
        world = CatalogProjectWorld(root / "world")
        task_factory = world.task_factory
        decision = decision or ScriptedProject()
    elif workload == "service":
        world = DevOpsWorld(root / "world")
        task_factory = tasks
    else:
        raise ValueError("workload must be service or project")
    memory = MemoryAdapter(str(root / "memory.sqlite"))
    corpus = MemoryAdapter()
    guide = getattr(world, "guide", "Service search entry point is search(records, request) in candidate.py. "
             "Operations probes deployed.py; development probes candidate.py. "
             "Deploy copies the candidate to the deployment. Search returns record IDs. "
             "Preserve earlier requirements when implementing a new repair.")
    corpus.put(MemoryItem("project-guide", "runtime", "", "catalog-service", guide,
                          embedding=list(map(float, encoder([guide])[0])) if "vector" in mode else None))
    decision = decision or ScriptedDevOps()
    scope = ("shared" if mode == "protocol_vector_memory" else
             "group" if mode == "protocol_vector_group_memory" else "none")
    report = {"experiment": "related_devops_sequences", "run_id": run_id, "mode": mode,
              "backend": backend, "workload": getattr(world, "workload_id", "constructed_catalog_service_v1"),
              "workload_option": workload, "snapshot_extension": getattr(world, "snapshot_extension", ".py"),
              "qwen_token_budgets": ({"planner": 768, "executor": 2048 if workload == "project" else 768, "verifier": 768}
                                     if backend == "qwen" else None),
              "runtime_profile": {"public_evidence_projection": workload == "project" and backend == "qwen",
                                  "typed_nonpatch_binding": workload == "project" and backend == "qwen"},
              "requested_tasks": task_limit, "full_sequence": task_limit == 10,
              "evidence_scope": ("scripted decisions and fixed vectors; not LLM performance evidence"
                                 if backend == "fixture" else "local Qwen decisions; constructed tasks; "
                                 + ("Qwen embeddings" if "vector" in mode else "no vector retrieval")),
              "status": "running", "tasks": [], "max_steps": max_steps, "active_task": None}

    def publish():
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if on_update:
            on_update(json.loads(json.dumps(report)))

    try:
        publish()
        for task in task_factory()[:task_limit]:
            report["active_task"] = task["task_id"]
            publish()
            if any(not row["passed"] for row in report["tasks"]):
                report["tasks"].append({"task_id": task["task_id"], "group": task["group"],
                                        "status": "blocked", "passed": False,
                                        "reason": "prerequisite failed; no reference repair injected"})
                publish()
                continue
            if hasattr(world, "prepare_task"):
                world.prepare_task(task)
            transport = TextAdapter() if mode == "text" else ProtocolAdapter()
            graph, packets, metrics = build_devops_graph(
                world, memory, transport, decision, encoder, run_id=run_id,
                memory_scope=scope, use_vector="vector" in mode, max_steps=max_steps, corpus=corpus)
            before_reads, before_tools = world.source_reads, world.tool_calls
            before_patch_files = getattr(world, "patch_files", 0)
            before_operations = getattr(world, "probe_operations", 0)
            before_calls = len(decision.call_stats)
            before_prompts = len(getattr(decision, "prompt_records", []))
            started = time.perf_counter()
            try:
                result = graph.invoke({"task": task}, {"recursion_limit": max_steps * 5 + 20})
                task_elapsed = time.perf_counter() - started
                evaluation_started = time.perf_counter()
                grade = world.evaluate(task, result["observations"])
                evaluation_sec = time.perf_counter() - evaluation_started
                passed = result["status"] == "verified" and grade["passed"]
                commit_started = time.perf_counter()
                if passed and result["memory_candidate"]:
                    memory.put(MemoryItem(**result["memory_candidate"]))
                task_elapsed += time.perf_counter() - commit_started
                row = {"task_id": task["task_id"], "group": task["group"], "status": result["status"],
                       "passed": passed, "steps": result["steps"], "summary": result["summary"],
                       "observations": result["observations"], "grade": grade, "error": result["error"],
                       "evaluation_sec": evaluation_sec}
            except Exception as exc:
                task_elapsed = time.perf_counter() - started
                row = {"task_id": task["task_id"], "group": task["group"], "status": "error",
                       "passed": False, "error": f"{type(exc).__name__}: {exc}"}
            calls = decision.call_stats[before_calls:]
            row.update(metrics.get(task["task_id"], {}))
            row.update(elapsed_sec=task_elapsed,
                       source_context_reads=world.source_reads - before_reads,
                       tool_calls=world.tool_calls - before_tools,
                       patch_files=getattr(world, "patch_files", 0) - before_patch_files,
                       probe_operations=getattr(world, "probe_operations", 0) - before_operations,
                       llm_calls=len(calls), prompt_tokens=sum(c["prompt_tokens"] for c in calls),
                       generated_tokens=sum(c["generated_tokens"] for c in calls),
                       message_count=len(transport.frames),
                       message_chars=sum(len(f) for f in transport.frames),
                       message_bytes=sum(len(f.encode()) for f in transport.frames))
            row["effective_reuse"] = row.get("accepted_reuse", 0) if row["passed"] else 0
            row["effective_cross_group_reuse"] = row.get("cross_group_reuse", 0) if row["passed"] else 0
            row["total_transfer_bytes"] = (row["message_bytes"] + row.get("state_payload_bytes", 0)
                                           + row.get("state_metadata_bytes", 0))
            (root / f"{task['task_id']}-trace.json").write_text(json.dumps(transport.frames, indent=2))
            if hasattr(decision, "prompt_records"):
                (root / f"{task['task_id']}-model-prompts.json").write_text(
                    json.dumps(decision.prompt_records[before_prompts:], ensure_ascii=False, indent=2))
            # State bytes are persisted separately from JSON control messages.
            for index, packet in enumerate(packets.values()):
                (root / f"{task['task_id']}-state-{index}.bin").write_bytes(packet.payload)
                metadata = asdict(packet)
                metadata.pop("payload")
                (root / f"{task['task_id']}-state-{index}.json").write_text(json.dumps(metadata, indent=2))
            extension = getattr(world, "snapshot_extension", ".py")
            (root / f"{task['task_id']}-candidate{extension}").write_text(world.source.read_text())
            (root / f"{task['task_id']}-deployed{extension}").write_text(world.deployed.read_text())
            report["tasks"].append(row)
            publish()
        report["status"] = ("complete" if task_limit == 10 else "partial") if all(r["passed"] for r in report["tasks"]) else "failed"
        report["active_task"] = None
        report["totals"] = {key: sum(r.get(key, 0) for r in report["tasks"]) for key in (
            "message_count", "message_chars", "message_bytes", "state_transfers", "state_payload_bytes",
            "state_metadata_bytes", "total_transfer_bytes",
            "source_context_reads", "tool_calls", "patch_files", "probe_operations", "context_filtered_reuse",
            "llm_calls", "prompt_tokens", "generated_tokens",
            "memory_queries", "accepted_reuse", "cross_group_reuse", "effective_reuse",
            "effective_cross_group_reuse", "avoided_source_reads", "elapsed_sec", "evaluation_sec")}
        report["totals"]["passed_tasks"] = sum(r["passed"] for r in report["tasks"])
        report["totals"]["accepted_query_rate"] = (report["totals"]["accepted_reuse"] /
                                                   max(report["totals"]["memory_queries"], 1))
        publish()
        return report
    finally:
        memory.close()
        corpus.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fixture", "qwen"), default="fixture")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--workload", choices=("service", "project"), default="project")
    parser.add_argument("--output-root", default="results/devops")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--task-limit", type=int, default=10, help="Run an ordered prefix for diagnostics; fewer than 10 is labelled partial")
    parser.add_argument("--model-path", default="/home/Users/SharedFile/Models/Qwen/Qwen3-8B")
    parser.add_argument("--embedding-path", default="/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--embedding-device", default="cpu")
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    if not 1 <= args.task_limit <= 10:
        parser.error("task-limit must be between 1 and 10")
    decision, encoder = None, fixture_embedding
    if args.backend == "qwen":
        from coop.qwen_agent import QwenCausalAdapter
        from coop.embedding import LocalQwenEmbedding
        decision = QwenDevOps(QwenCausalAdapter(args.model_path, device=args.device, max_new_tokens=768))
        embedding = LocalQwenEmbedding(args.embedding_path, device=args.embedding_device, output_dim=768)
        encoder = embedding.encode
    root = Path(args.output_root) / (args.backend + "-" + uuid4().hex[:8])
    modes = MODES if args.mode == "all" else (args.mode,)
    reports = [run_sequence(root / mode, mode=mode, decision=decision, encoder=encoder,
                            max_steps=args.max_steps, backend=args.backend, workload=args.workload,
                            task_limit=args.task_limit) for mode in modes]
    print(json.dumps({"directory": str(root), "modes": {
        report["mode"]: {"status": report["status"], **report["totals"]} for report in reports}}, indent=2))
    if any(report["status"] not in ("complete", "partial") for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
