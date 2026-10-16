"""Kill an unfinished Planner; verify the same task's durable repair quota."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4


def run_case(directory, *, case, fake, device, max_tokens, timeout,
             model_path="/home/Users/SharedFile/Models/Qwen/Qwen3-8B"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    common = [sys.executable, "-m", "tools.repair_recovery_worker", "--directory", str(directory),
              "--case", case, "--device", device, "--max-new-tokens", str(max_tokens), "--model-path", model_path]
    if fake:
        common.append("--fake")
    marker_path = directory / "ready.json"
    started = time.perf_counter()
    with (directory / "prepare.log").open("w", encoding="utf-8") as log:
        worker = subprocess.Popen(common + ["--phase", "prepare"], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            while not marker_path.exists():
                if worker.poll() is not None:
                    raise RuntimeError(f"prepare worker failed; see {directory / 'prepare.log'}")
                if time.perf_counter() - started > timeout:
                    raise TimeoutError("worker did not reach durable reservation boundary")
                time.sleep(0.05)
            # Worker publishes the marker using an atomic rename after fsync.
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
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
    attempts = resumed["ledger_attempts"]
    original_count = len(marker["ledger_attempts"])
    checks = {"prepare_killed_by_sigkill": return_code == -9,
              "different_worker_pid": marker["pid"] != resumed["pid"],
              "planner_replayed_from_checkpoint": resumed["checkpoint_before_next"] == ["planner"],
              "original_reservations_preserved": marker["ledger_attempts"] == attempts[:original_count],
              "attempt_numbers_contiguous": [record["attempt"] for record in attempts] == list(range(len(attempts))),
              "remaining_budget_correct": all(record["remaining"] == 2 - record["attempt"] for record in attempts),
              "reservation_bound": len(attempts) <= 3,
              "physical_call_bound": marker["calls"] + resumed["calls"] <= 3,
              "direct_ledger_reinvoke_zero_calls": resumed["direct_ledger_reinvoke_calls"] == 0}
    if case == "inflight":
        checks.update(inflight_reservation_spent=attempts[0]["status"] == "started",
                      no_repeated_draft=resumed["calls"] <= 2,
                      complete_or_explicit_exhaustion=resumed["status"] in ("complete", "PlanRepairError"))
    elif case == "accepted":
        checks.update(accepted_plan_replayed_zero_calls=resumed["calls"] == 0 and resumed["status"] == "complete",
                      plan_unchanged=resumed["value"]["plan"] == marker["ledger_attempts"][-1]["candidate"])
    else:
        checks.update(exhaustion_replayed_zero_calls=resumed["calls"] == 0 and resumed["status"] == "PlanRepairError",
                      exhausted_remaining_zero=attempts[-1]["remaining"] == 0)
    return {"case": case, "checks": checks, "all_passed": all(checks.values()),
            "prepare_marker": marker, "resume": resumed, "elapsed_sec": time.perf_counter() - started,
            "artifact_directory": str(directory)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--model-path", default="/home/Users/SharedFile/Models/Qwen/Qwen3-8B")
    parser.add_argument("--report", default="results/repair_recovery_v2.json")
    args = parser.parse_args()
    target = Path(args.report)
    directory = target.parent / (target.stem + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    cases = ["inflight", "accepted", "exhausted"] if args.fake else ["inflight", "accepted"]
    report = {"experiment": "planner_node_replay_with_atomic_repair_reservations", "config": vars(args),
              "artifact_directory": str(directory), "cases": [],
              "scope": "At-most-budget reservations survive SIGKILL; unknown results are not regenerated. No external-model exactly-once or token-cost inference."}
    for case in cases:
        result = run_case(directory / case, case=case, fake=args.fake, device=args.device,
                          max_tokens=args.max_new_tokens, timeout=90 if args.fake else 600, model_path=args.model_path)
        report["cases"].append(result)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["all_passed"] = all(case["all_passed"] for case in report["cases"])
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_passed": report["all_passed"], "cases": [
        {"case": result["case"], "checks": result["checks"], "resume_calls": result["resume"]["calls"],
         "status": result["resume"]["status"]} for result in report["cases"]], "report": str(target)}, ensure_ascii=False, indent=2))
    if not report["all_passed"]:
        raise SystemExit("repair recovery checks failed; see reports/logs")


if __name__ == "__main__":
    main()
