"""Pair memory backends while holding graph, mode, models and tasks fixed."""

import argparse
import json
from pathlib import Path
from statistics import mean
import time
from uuid import uuid4

from tools.run_cost_workload import prepare_cost_runtime, run_cost_mode


BACKENDS = ("sqlite", "langgraph")


def compare_backend_results(results):
    """Check semantic equality independently of timings and storage capability."""
    if set(results) != set(BACKENDS):
        raise ValueError("paired comparison needs sqlite and langgraph results")
    reference = results["sqlite"]
    other = results["langgraph"]
    keys = ("task_id", "status", "expression", "tool_result", "answer", "quality",
            "reused_memory_id", "accepted_reuse", "reuse_source_task", "plan_attempts")
    task_equal = ([{key: row.get(key) for key in keys} for row in reference["tasks"]]
                  == [{key: row.get(key) for key in keys} for row in other["tasks"]])
    invariants = ("completed", "passed_count", "model_calls", "prompt_tokens", "generated_tokens",
                  "message_count", "message_tokens", "total_transfer_bytes", "state_transfers",
                  "accepted_reuse", "corpus_searches", "source_parses", "plan_attempt_count", "new_memory_records")
    return {"task_results_equal": task_equal,
            "invariant_metrics_equal": all(reference[key] == other[key] for key in invariants),
            "both_quality_passed": all(value["passed_count"] == value["task_count"] for value in results.values()),
            "different_durability": reference["memory_persistent"] and not other["memory_persistent"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-plan-repairs", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--report", default="results/memory_backend_comparison.json")
    args = parser.parse_args()
    if args.repeat < 1 or args.max_new_tokens < 1 or not 1 <= args.max_plan_repairs <= 5 or not -1 <= args.threshold <= 1:
        parser.error("invalid trial count, generation budget, correction budget or threshold")
    target = Path(args.report)
    directory = target.parent / (target.stem + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    started = time.perf_counter()
    llm, encoder, docs, tasks, metadata = prepare_cost_runtime(
        device=args.device, embedding_device=args.embedding_device,
        max_new_tokens=args.max_new_tokens, repair_plan=True, max_plan_repairs=args.max_plan_repairs,
    )
    report = {"experiment": "paired_memory_backend_comparison", "config": vars(args), **metadata,
              "fixed_mode": "protocol_vector_memory", "durable_state_and_checkpoints": True,
              "artifact_directory": str(directory), "trials": [],
              "limitations": ["LangGraph InMemoryStore is not durable; SQLite task memory is durable.",
                              "Both adapters use the same exact scan/cosine ranker, not native ANN or vector database indexing.",
                              "Only task memory changes; corpus, state/checkpoints, protocol and model prompts are fixed.",
                              "Small synthetic workload; adapter timings do not establish general backend performance.",
                              "Snapshots are audit exports, not automatic restart support for InMemoryStore."]}
    for trial in range(args.repeat):
        order = list(BACKENDS) if trial % 2 == 0 else list(reversed(BACKENDS))
        results = {}
        for backend in order:
            results[backend] = run_cost_mode(
                llm, encoder.encode, docs, tasks, directory / f"trial-{trial}-{backend}", "protocol_vector_memory",
                namespace=metadata["namespace"], threshold=args.threshold,
                max_new_tokens=args.max_new_tokens, durable=True, repair_plan=True,
                max_plan_repairs=args.max_plan_repairs, memory_backend=backend,
            )
        checks = compare_backend_results(results)
        report["trials"].append({"order": order, "backends": results, "checks": checks})
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = {}
    for backend in BACKENDS:
        values = [trial["backends"][backend] for trial in report["trials"]]
        summaries[backend] = {key: mean(value[key] for value in values) for key in (
            "completed", "passed_count", "model_calls", "accepted_reuse", "corpus_searches",
            "prompt_tokens", "generated_tokens", "message_tokens", "new_memory_records", "elapsed_sec")}
        operation_names = sorted({name for value in values for name in value["memory_backend_operations"]})
        summaries[backend]["adapter_operations"] = {
            name: {key: mean(value["memory_backend_operations"].get(name, {}).get(key, 0) for value in values)
                   for key in ("calls", "errors", "elapsed_sec")} for name in operation_names}
    report["summary"] = summaries
    report["all_checks_passed"] = all(all(trial["checks"].values()) for trial in report["trials"])
    report["total_sec_including_preparation"] = time.perf_counter() - started
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "summary": summaries,
                      "report": str(target)}, ensure_ascii=False, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit("backend comparison failed; see task results in report")


if __name__ == "__main__":
    main()
