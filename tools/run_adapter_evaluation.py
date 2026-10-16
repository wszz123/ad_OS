"""Run the paired ten-task collaboration adapter ablation."""

import argparse
import json

from coop.evaluator import MODES, run_evaluation, save_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--report", default="results/adapter_ablation.json")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--without-local-model", action="store_true")
    args = parser.parse_args()

    tokenizer = None
    embedder = None
    if not args.without_local_model:
        from transformers import AutoTokenizer
        from coop.embedding import LocalQwenEmbedding

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        encoder = LocalQwenEmbedding(args.model_path, device="cuda", output_dim=768)
        embedder = encoder.encode

    report = run_evaluation(tokenizer=tokenizer, embedder=embedder,
                            modes=args.mode or MODES, repeat=args.repeat)
    save_report(report, args.report)
    summary = {
        mode: {key: value for key, value in metrics.items() if key != "tasks"}
        for mode, metrics in report["modes"].items()
    }
    summary["embedding_prepare_sec"] = report["embedding_prepare_sec"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
