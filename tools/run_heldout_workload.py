"""Frozen-policy parameter holdout and corpus-scan memory ablation."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from statistics import mean
import time
from uuid import uuid4

from coop.embedding import LocalQwenEmbedding
from coop.heldout_workload import expanded_documents, heldout_tasks
from coop.qwen_agent import QwenCausalAdapter
from coop.memory import MemoryAdapter
from tools.run_cost_workload import EMBEDDING, QWEN, cost_namespace, prepare_cost_runtime, run_cost_mode


MODES = ("protocol_vector", "protocol_vector_memory")


def save_progress(target, report):
    """Atomically replace the audit report; never truncate the prior complete one."""
    target = Path(target)
    temporary = target.with_name(target.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_cell_trials(cell, *, repeat, run_mode, save):
    """Resume completed modes; an unfinished mode is rerun with fresh memory."""
    for trial in range(len(cell["trials"]), repeat):
        order = list(MODES) if trial % 2 == 0 else list(reversed(MODES))
        pending = cell.setdefault("pending_trial", {"index": trial, "order": order, "modes": {}, "artifacts": {}})
        if pending["index"] != trial or pending["order"] != order:
            raise ValueError("pending trial does not match recorded experiment order")
        for mode in order:
            if mode in pending["modes"]:
                continue
            result, artifact = run_mode(trial, mode)
            pending["modes"][mode] = result
            pending["artifacts"][mode] = artifact
            save()
        cell["trials"].append({"order": order, "modes": pending["modes"], "artifacts": pending["artifacts"]})
        del cell["pending_trial"]
        save()
    cell["summary"] = summarize_cell(cell["trials"])
    save()


def load_cached_documents(report, cell):
    """Reuse indexed vectors from a completed mode, with original order restored."""
    directory = Path(report["artifact_directory"])
    source = None
    for trial_index, trial in enumerate(cell["trials"]):
        for mode in trial["modes"]:
            path = Path(trial.get("artifacts", {}).get(mode, directory / f"distractors-{cell['distractor_count']}-trial-{trial_index}-{mode}"))
            if (path / "corpus.sqlite").is_file():
                source = path / "corpus.sqlite"
                break
        if source:
            break
    if source is None:
        raise ValueError("resume needs an indexed corpus from a completed mode")
    store = MemoryAdapter(str(source))
    try:
        by_id = {item.memory_id: item for item in store.items()}
    finally:
        store.close()
    recorded = cell["metadata"]["documents"]
    docs = [by_id[record["memory_id"]] for record in recorded]
    actual = [{key: value for key, value in asdict(item).items() if key != "embedding"} for item in docs]
    if actual != recorded or len(docs) != cell["corpus_size"] or any(len(item.embedding or []) != 768 for item in docs):
        raise ValueError("cached source/vector schema differs from recorded corpus")
    return docs


def resume_experiment(target):
    target = Path(target)
    report = json.loads(target.read_text(encoding="utf-8"))
    if report["experiment"] != "frozen_policy_parameter_holdout_and_corpus_scaling":
        raise ValueError("report is not a heldout experiment")
    config = report["config"]
    tasks = heldout_tasks(config["seed"], config["rounds"])
    if any(json.loads(json.dumps([asdict(task) for task in tasks])) != cell["metadata"]["tasks"] for cell in report["cells"]):
        raise ValueError("recorded tasks differ from the current task generator")
    pending_cells = [cell for cell in report["cells"] if len(cell["trials"]) < config["repeat"]]
    if not pending_cells:
        for cell in report["cells"]:
            cell["summary"] = summarize_cell(cell["trials"])
            print(json.dumps({"corpus_size": cell["corpus_size"], **cell["summary"]}, ensure_ascii=False, indent=2))
        report["scan_accounting"] = "logical vector JSON bytes, including full scans and point reads; excludes task memory storage and physical I/O"
        save_progress(target, report)
        print("All recorded trials are complete; no models were loaded.")
        return
    expected_sizes = list(dict.fromkeys(config["distractors"] if config["distractors"] is not None else [0, 128]))
    if {cell["distractor_count"] for cell in report["cells"]} != set(expected_sizes):
        raise ValueError("a corpus cell has not been initialized; start a new experiment for that cell")
    llm = QwenCausalAdapter(QWEN, device=config["device"], max_new_tokens=config["max_new_tokens"])
    embedding = LocalQwenEmbedding(EMBEDDING, device=config["embedding_device"], output_dim=768)
    cached = {}
    for cell in pending_cells:
        docs = load_cached_documents(report, cell)
        fingerprint = cost_namespace(docs, instruction=embedding.instruction, repair_plan=True,
                                     max_plan_repairs=config["max_plan_repairs"])
        if fingerprint != cell["metadata"]["namespace"]:
            raise ValueError("resume policy/source/vector fingerprint mismatch")
        cached[cell["distractor_count"]] = docs
    started = time.perf_counter()
    llm.load()
    embedding.encode(["Warmup heldout resume"], query=True)
    llm.generate([{"role": "user", "content": "Say ready."}], max_new_tokens=4)
    session = {"session_id": uuid4().hex[:8], "warmup_sec": time.perf_counter() - started,
               "note": "Complete modes are retained; unfinished modes restart with empty memory, no policy/threshold tuning."}
    report.setdefault("resume_sessions", []).append(session)
    save_progress(target, report)
    for cell in pending_cells:
        docs = cached[cell["distractor_count"]]
        def run(trial, mode):
            artifact = Path(report["artifact_directory"]) / f"distractors-{cell['distractor_count']}-trial-{trial}-{mode}-resume-{session['session_id']}"
            data = run_cost_mode(llm, embedding.encode, docs, tasks, artifact, mode,
                                 namespace=cell["metadata"]["namespace"], threshold=config["threshold"],
                                 max_new_tokens=config["max_new_tokens"], durable=True, repair_plan=True,
                                 max_plan_repairs=config["max_plan_repairs"])
            return data, str(artifact)
        run_cell_trials(cell, repeat=config["repeat"], run_mode=run, save=lambda: save_progress(target, report))
        print(json.dumps({"corpus_size": cell["corpus_size"], **cell["summary"]}, ensure_ascii=False, indent=2))
    session["elapsed_sec"] = time.perf_counter() - started
    report["resume_complete"] = True
    # Historical total wall time was not saved before interruption; do not invent it.
    save_progress(target, report)
    print(f"report={target}")


def summarize_cell(trials):
    summary = {}
    for mode in MODES:
        rows = [trial["modes"][mode] for trial in trials]
        summary[mode] = {key: mean(row[key] for row in rows) for key in (
            "task_count", "completed", "passed_count", "model_calls", "accepted_reuse", "corpus_searches",
            "source_parses", "plan_repairs", "prompt_tokens", "generated_tokens", "elapsed_sec",
            "corpus_search_sec", "memory_lookup_sec", "embedding_sec")}
        summary[mode]["corpus_scan_stats"] = {
            key: mean(row["corpus_scan_stats"][key] for row in rows) for key in rows[0]["corpus_scan_stats"]}
        stats = summary[mode]["corpus_scan_stats"]
        summary[mode]["total_corpus_vector_json_bytes_loaded"] = (
            stats["vector_json_bytes_loaded"] + stats.get("point_vector_json_bytes_loaded", 0))
    baseline, memory = summary[MODES[0]], summary[MODES[1]]
    return {"modes": summary,
            "saved_full_scan_rows": baseline["corpus_scan_stats"]["full_scan_rows"] - memory["corpus_scan_stats"]["full_scan_rows"],
            "saved_full_scan_vector_json_bytes": baseline["corpus_scan_stats"]["vector_json_bytes_loaded"] - memory["corpus_scan_stats"]["vector_json_bytes_loaded"],
            "saved_materialized_vector_json_bytes": baseline["total_corpus_vector_json_bytes_loaded"] - memory["total_corpus_vector_json_bytes_loaded"],
            "all_quality_passed": all(row["passed_count"] == row["task_count"] for trial in trials for row in trial["modes"].values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--rounds", type=int, default=1, help="Ten new parameter combinations per round")
    parser.add_argument("--distractors", action="append", type=int, help="Additional documents per corpus; default 0 and 128")
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-plan-repairs", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--report", default="results/heldout_workload.json")
    parser.add_argument("--resume", action="store_true", help="Continue the saved report using recorded config and indexed vectors")
    args = parser.parse_args()
    if args.resume:
        resume_experiment(args.report)
        return
    sizes = list(dict.fromkeys(args.distractors if args.distractors is not None else [0, 128]))
    if (args.rounds < 1 or args.repeat < 1 or args.embedding_batch_size < 1 or args.max_new_tokens < 1
            or not 1 <= args.max_plan_repairs <= 5 or not -1 <= args.threshold <= 1 or any(size < 0 for size in sizes)):
        parser.error("invalid rounds, repeat, batch size, corpus size or policy budget")
    tasks = heldout_tasks(seed=args.seed, rounds=args.rounds)
    target = Path(args.report)
    directory = target.parent / (target.stem + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    started = time.perf_counter()
    # Share weights across all corpus sizes; initializing a second 8B copy would
    # distort comparison and can exceed local GPU memory.
    llm = QwenCausalAdapter(QWEN, device=args.device, max_new_tokens=args.max_new_tokens)
    embedding = LocalQwenEmbedding(EMBEDDING, device=args.embedding_device, output_dim=768)
    report = {"experiment": "frozen_policy_parameter_holdout_and_corpus_scaling", "config": vars(args),
              "scan_accounting": "logical vector JSON bytes, including full scans and point reads; excludes task memory storage and physical I/O",
              "artifact_directory": str(directory), "cells": [],
              "limitations": ["New parameters on known structures/entities, not new-domain generalization.",
                              "Document values and original task templates remain unchanged; independent Decimal labels are evaluator-only.",
                              "Distractors belong to other entities; public entity filters apply. This tests scan cost, not hard within-entity ranking.",
                              "Corpus search currently materializes all rows before filtering; counters are logical rows/serialized bytes, not physical disk I/O.",
                              "LLM prompts, repair policy and memory threshold are fixed before running; all failures remain in the report.",
                              "Model weights are shared, corpus embedding/index preparation is separate from task time."]}
    for size in sizes:
        docs = expanded_documents(distractors=size, seed=args.seed)
        _, _, _, _, metadata = prepare_cost_runtime(
            device=args.device, embedding_device=args.embedding_device, max_new_tokens=args.max_new_tokens,
            repair_plan=True, max_plan_repairs=args.max_plan_repairs, docs=docs, tasks=tasks,
            embedding_batch_size=args.embedding_batch_size, llm=llm, embedding=embedding,
        )
        cell = {"distractor_count": size, "corpus_size": len(docs), "metadata": metadata, "trials": []}
        report["cells"].append(cell)
        save_progress(target, report)
        def run(trial, mode):
            artifact = directory / f"distractors-{size}-trial-{trial}-{mode}"
            data = run_cost_mode(llm, embedding.encode, docs, tasks, artifact, mode,
                                 namespace=metadata["namespace"], threshold=args.threshold,
                                 max_new_tokens=args.max_new_tokens, durable=True, repair_plan=True,
                                 max_plan_repairs=args.max_plan_repairs)
            return data, str(artifact)
        run_cell_trials(cell, repeat=args.repeat, run_mode=run, save=lambda: save_progress(target, report))
        print(json.dumps({"corpus_size": len(docs), **cell["summary"]}, ensure_ascii=False, indent=2))
    report["total_sec_including_all_preparation"] = time.perf_counter() - started
    save_progress(target, report)
    print(f"report={target}")


if __name__ == "__main__":
    main()
