"""Fresh-process workload for the unified demonstration entrypoint."""

import argparse
import hashlib
import json
from pathlib import Path

from coop.demo_support import EMBEDDING, QWEN, FixturePlanner, fixture_encode
from coop.workload import cost_documents, cost_tasks
from tools.run_cost_workload import MODES, prepare_cost_runtime, run_cost_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("fixture", "qwen"), required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-device", default="cuda:1")
    parser.add_argument("--model-path", default=QWEN)
    parser.add_argument("--embedding-path", default=EMBEDDING)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    directory = Path(args.directory)
    docs, tasks = cost_documents(), cost_tasks()
    if args.backend == "fixture":
        llm, encode = FixturePlanner(), fixture_encode
        for item in docs:
            item.embedding = encode([item.summary])[0]
        prepared = {"namespace": hashlib.sha256(b"explicit-scripted-demo-v1").hexdigest(),
                    "runtime": {"backend": "fixture", "planner": "public-intent-recipes", "vector_dimensions": 2,
                                "counter": "whitespace units, not model tokenizer tokens"}}
    else:
        from coop.qwen_agent import QwenCausalAdapter
        from coop.embedding import LocalQwenEmbedding
        llm = QwenCausalAdapter(args.model_path, device=args.device, max_new_tokens=args.max_new_tokens)
        embedding = LocalQwenEmbedding(args.embedding_path, device=args.embedding_device, output_dim=768)
        llm, embedding, docs, tasks, prepared = prepare_cost_runtime(
            device=args.device, embedding_device=args.embedding_device, max_new_tokens=args.max_new_tokens,
            repair_plan=True, llm=llm, embedding=embedding)
        encode = embedding.encode
    results = {}
    for mode in MODES:
        results[mode] = run_cost_mode(llm, encode, docs, tasks, directory / mode, mode,
            namespace=prepared["namespace"], max_new_tokens=args.max_new_tokens, repair_plan=True,
            durable=True, storage_profile="optimized", ingest_strategy="batch")
    report = {"experiment": "four_role_sourced_cost_workload", "config": vars(args), **prepared,
              "demo_backend": args.backend, "trials": [{"order": list(MODES), "modes": results}],
              "limitations": ["Single demonstration trial, not a counter-ordered performance experiment.",
                              "Fixture mode is scripted wiring with illustrative vectors/counters, not LLM evidence."]}
    path = directory / "workload.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(row["passed_count"] != row["task_count"] for row in results.values()):
        raise SystemExit(f"demo quality failure; see {path}")
    print(f"workload_report={path}")


if __name__ == "__main__":
    main()
