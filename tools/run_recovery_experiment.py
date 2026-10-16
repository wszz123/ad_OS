"""SIGKILL a checkpointed worker and resume its task in a fresh process."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4


def run_recovery(report_path, *, fake=False, device="cuda:0", embedding_device="cuda:1",
                 max_new_tokens=96, timeout=300,
                 model_path="/home/Users/SharedFile/Models/Qwen/Qwen3-8B",
                 embedding_path="/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B"):
    target = Path(report_path).resolve()
    directory = target.parent / (target.stem + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    common = [sys.executable, "-m", "tools.recovery_worker", "--directory", str(directory),
              "--device", device, "--embedding-device", embedding_device,
              "--max-new-tokens", str(max_new_tokens), "--model-path", model_path, "--embedding-path", embedding_path]
    if fake:
        common.append("--fake")
    started = time.perf_counter()
    with (directory / "prepare.log").open("w", encoding="utf-8") as log:
        worker = subprocess.Popen(common + ["--phase", "prepare", "--wait-for-kill"],
                                  cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            marker = directory / "prepare.json"
            while not marker.exists():
                if worker.poll() is not None:
                    raise RuntimeError(f"prepare worker failed; see {directory / 'prepare.log'}")
                if time.perf_counter() - started > timeout:
                    raise TimeoutError("prepare worker did not reach checkpoint")
                time.sleep(0.05)
            # Marker could briefly be visible before json.dump completes.
            while True:
                try:
                    prepared = json.loads(marker.read_text(encoding="utf-8"))
                    break
                except json.JSONDecodeError:
                    if time.perf_counter() - started > timeout:
                        raise TimeoutError("incomplete prepare audit marker")
                    time.sleep(0.05)
            worker.kill()
            return_code = worker.wait(timeout=30)
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=30)
    with (directory / "resume.log").open("w", encoding="utf-8") as log:
        subprocess.run(common + ["--phase", "resume"], cwd=root, stdout=log,
                       stderr=subprocess.STDOUT, timeout=timeout, check=True)
    resumed = json.loads((directory / "resume.json").read_text(encoding="utf-8"))
    checks = {
        "different_worker_pids": prepared["pid"] != resumed["pid"],
        "prepare_killed_by_sigkill": return_code == -9,
        "paused_before_retrieval": prepared["next_nodes"] == ["retriever"],
        "state_payload_preserved": prepared["state_checksum"] == resumed["state_checksum"],
        "state_not_duplicated": prepared["state_count"] == resumed["state_count"] == 1,
        "no_repeated_planner": resumed["calls"] == ["summarizer"],
        "no_repeated_embedding": resumed["query_embedding_calls"] == [],
        "completed": resumed["next_nodes"] == [] and bool(resumed["answer"]),
        "memory_written_once": prepared["memory_records"] == 0 and resumed["memory_records"] == 1,
        "completed_reinvoke_is_noop": resumed["completed_reinvoke_calls"] == 0,
        "metrics_restored": resumed["task_metrics"]["embedding_sec"] == prepared["task_metrics"]["embedding_sec"],
    }
    report = {"experiment": "sqlite_checkpoint_state_sigkill_recovery", "fake": fake,
              "artifact_directory": str(directory), "checks": checks, "all_passed": all(checks.values()),
              "prepare": prepared, "resume": resumed,
              "total_sec": time.perf_counter() - started,
              "scope": "Post-planner committed-boundary recovery; not arbitrary mid-node exactly-once execution"}
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["all_passed"]:
        raise AssertionError(f"recovery checks failed: {checks}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--report", default="results/qwen_recovery.json")
    parser.add_argument("--model-path", default="/home/Users/SharedFile/Models/Qwen/Qwen3-8B")
    parser.add_argument("--embedding-path", default="/home/Users/SharedFile/Models/Qwen/Qwen3-Embedding-4B")
    args = parser.parse_args()
    result = run_recovery(args.report, fake=args.fake, device=args.device,
                          embedding_device=args.embedding_device, max_new_tokens=args.max_new_tokens,
                          model_path=args.model_path, embedding_path=args.embedding_path)
    print(json.dumps({"all_passed": result["all_passed"], "checks": result["checks"],
                      "answer": result["resume"]["answer"], "report": args.report,
                      "artifact_directory": result["artifact_directory"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
