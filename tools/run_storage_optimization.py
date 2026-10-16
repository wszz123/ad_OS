"""Four-profile storage ablation using a frozen, already-indexed heldout corpus."""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
import time
from uuid import uuid4

import torch

from coop.embedding import LocalQwenEmbedding
from coop.heldout_workload import heldout_tasks
from coop.qwen_agent import QwenCausalAdapter
from tools.run_cost_workload import EMBEDDING, QWEN, STORAGE_PROFILES, cost_namespace, run_cost_mode
from tools.run_heldout_workload import load_cached_documents, save_progress


def total_vector_bytes(result):
    stats = result["corpus_scan_stats"]
    return sum(stats.get(key, 0) for key in (
        "vector_json_bytes_loaded", "filtered_vector_json_bytes_loaded", "point_vector_json_bytes_loaded"))


def compare_storage_results(results):
    reference = results["reference"]
    task_keys = ("task_id", "status", "expression", "tool_result", "answer", "quality",
                 "reused_memory_id", "accepted_reuse", "reuse_source_task", "plan_attempts")
    invariants = ("completed", "passed_count", "model_calls", "prompt_tokens", "generated_tokens",
                  "message_tokens", "total_transfer_bytes", "accepted_reuse", "corpus_searches", "plan_repairs")
    checks = {}
    for profile, result in results.items():
        checks[profile] = {
            "task_results_equal": [{key: row.get(key) for key in task_keys} for row in reference["tasks"]]
                                  == [{key: row.get(key) for key in task_keys} for row in result["tasks"]],
            "invariant_metrics_equal": all(reference[key] == result[key] for key in invariants),
            "quality_passed": result["passed_count"] == result["task_count"],
        }
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", default="results/heldout_workload.json")
    parser.add_argument("--corpus-size", type=int, default=137)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--embedding-device")
    parser.add_argument("--report", default="results/storage_optimization.json")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    whole_started = time.perf_counter()
    source = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
    cells = [cell for cell in source["cells"] if cell["corpus_size"] == args.corpus_size]
    if len(cells) != 1:
        parser.error("source report needs exactly one matching corpus cell")
    cell = cells[0]
    docs = load_cached_documents(source, cell)
    config = source["config"]
    tasks = heldout_tasks(config["seed"], config["rounds"])
    if json.loads(json.dumps([asdict(task) for task in tasks])) != cell["metadata"]["tasks"]:
        raise ValueError("frozen task manifest no longer matches the current generator")
    device, embedding_device = args.device or config["device"], args.embedding_device or config["embedding_device"]
    llm = QwenCausalAdapter(QWEN, device=device, max_new_tokens=config["max_new_tokens"])
    encoder = LocalQwenEmbedding(EMBEDDING, device=embedding_device, output_dim=768)
    namespace = cost_namespace(docs, instruction=encoder.instruction, repair_plan=True,
                               max_plan_repairs=config["max_plan_repairs"])
    if namespace != cell["metadata"]["namespace"]:
        raise ValueError("frozen source/model/policy fingerprint does not match")
    source_cache_load_sec = time.perf_counter() - whole_started
    target = Path(args.report)
    directory = target.parent / (target.stem + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    started = time.perf_counter()
    llm.load()
    encoder.encode(["Warmup storage ablation"], query=True)
    llm.generate([{"role": "user", "content": "Say ready."}], max_new_tokens=4)
    report = {"experiment": "sqlite_storage_pushdown_projection_dedup_ablation", "config": vars(args),
              "source_config": config, "namespace": namespace, "corpus_size": len(docs),
              "profiles": STORAGE_PROFILES, "artifact_directory": str(directory), "trials": [],
              "fixed_mode": "protocol_vector_memory", "initialization_sec": time.perf_counter() - started,
              "source_cache_load_sec": source_cache_load_sec,
              "runtime": {"device": device, "embedding_device": embedding_device,
                          "generator_dtype": str(llm.model.dtype), "embedding_dtype": str(encoder._model.dtype),
                          "cpu_threads": torch.get_num_threads()},
              "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "langgraph")},
              "limitations": ["Logical materialized vector JSON bytes are not physical disk I/O.",
                              "Public entity/tag filtering, not native ANN; exact ranking remains unchanged.",
                              "Original corpus vectors are reused; index setup and fresh row ingestion are reported separately.",
                              "Cold-source load, seeding, model/embedding and task timings are separate; small runs do not prove general end-to-end speedup."]}
    for trial in range(args.repeat):
        order = list(STORAGE_PROFILES) if trial % 2 == 0 else list(reversed(STORAGE_PROFILES))
        data, artifacts = {}, {}
        report["pending_trial"] = {"index": trial, "order": order, "profiles": data, "artifacts": artifacts}
        for profile in order:
            artifact = directory / f"trial-{trial}-{profile}"
            data[profile] = run_cost_mode(llm, encoder.encode, docs, tasks, artifact, "protocol_vector_memory",
                namespace=namespace, threshold=config["threshold"], max_new_tokens=config["max_new_tokens"],
                durable=True, repair_plan=True, max_plan_repairs=config["max_plan_repairs"], storage_profile=profile)
            artifacts[profile] = str(artifact)
            save_progress(target, report)
        report["trials"].append({"order": order, "profiles": data, "artifacts": artifacts,
                                 "checks": compare_storage_results(data)})
        del report["pending_trial"]
        save_progress(target, report)
    summary = {}
    for profile in STORAGE_PROFILES:
        results = [trial["profiles"][profile] for trial in report["trials"]]
        summary[profile] = {key: mean(result[key] for result in results) for key in (
            "completed", "passed_count", "model_calls", "accepted_reuse", "corpus_searches",
            "corpus_search_sec", "memory_lookup_sec", "corpus_index_setup_sec", "corpus_ingest_sec", "elapsed_sec")}
        summary[profile]["scan_stats"] = {
            key: mean(result["corpus_scan_stats"][key] for result in results) for key in results[0]["corpus_scan_stats"]}
        summary[profile]["total_vector_json_bytes_loaded"] = mean(total_vector_bytes(result) for result in results)
    report["summary"] = summary
    report["all_checks_passed"] = all(all(check.values()) for trial in report["trials"] for check in trial["checks"].values())
    report["total_sec_including_initialization_and_ingestion"] = time.perf_counter() - whole_started
    save_progress(target, report)
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "summary": summary,
                      "report": str(target)}, ensure_ascii=False, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit("storage ablation failed consistency/quality checks; see report")


if __name__ == "__main__":
    main()
