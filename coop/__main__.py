"""Unified competition prototype CLI: doctor, demo, summary and source package."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4
import zipfile

from .demo_support import EMBEDDING, QWEN, environment_report
from .reporting import collect_reports, write_summary


ROOT = Path(__file__).resolve().parents[1]


def run_step(command, log_path, *, timeout, cpu_threads):
    """Own a process group so a timeout cannot leave nested GPU workers alive."""
    env = {**os.environ, "OMP_NUM_THREADS": str(cpu_threads), "MKL_NUM_THREADS": str(cpu_threads)}
    started = time.perf_counter()
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name == "posix")
        try:
            return_code = process.wait(timeout=timeout)
            status = "ok" if return_code == 0 else "error"
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            return_code = process.wait(timeout=30)
            status = "timeout"
    return {"command": command, "log": str(log_path), "return_code": return_code,
            "status": status, "elapsed_sec": time.perf_counter() - started}


def demo(args):
    directory = Path(args.output_root).resolve() / (args.backend + "-" + uuid4().hex[:8])
    directory.mkdir(parents=True)
    env = environment_report(backend=args.backend, model_path=args.model_path, embedding_path=args.embedding_path)
    (directory / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"experiment": "unified_prototype_demo", "backend": args.backend, "directory": str(directory),
                "environment_ready": env["ready"], "steps": [], "status": "running"}
    path = directory / "demo.json"
    def save():
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    if not env["ready"]:
        manifest["status"] = "environment_error"
        save()
        raise SystemExit(f"Environment incomplete; inspect {directory / 'environment.json'}")
    workload_report = directory / "workload.json"
    workload_name = "fixture_workload" if args.backend == "fixture" else "cost_workload"
    commands = [
        (workload_name,
         [sys.executable, "-m", "tools.demo_workload_worker", "--backend", args.backend,
                      "--directory", str(directory), "--device", args.device, "--embedding-device", args.embedding_device,
                      "--model-path", args.model_path, "--embedding-path", args.embedding_path,
                      "--max-new-tokens", str(args.max_new_tokens)], workload_report),
        ("state_recovery", [sys.executable, "-m", "tools.run_recovery_experiment",
                            "--report", str(directory / "state_recovery.json"), "--device", args.device,
                            "--embedding-device", args.embedding_device, "--model-path", args.model_path,
                            "--embedding-path", args.embedding_path], directory / "state_recovery.json"),
        ("repair_recovery", [sys.executable, "-m", "tools.run_repair_recovery", "--device", args.device,
                             "--model-path", args.model_path, "--max-new-tokens", str(args.max_new_tokens),
                             "--report", str(directory / "repair_recovery.json")], directory / "repair_recovery.json"),
    ]
    for name, command, report in commands:
        if args.skip_recovery and name != workload_name:
            continue
        if args.backend == "fixture" and name in ("state_recovery", "repair_recovery"):
            command.append("--fake")
        print(f"Running {name} ({args.backend}); log={directory / (name + '.log')}", flush=True)
        result = run_step(command, directory / (name + ".log"), timeout=args.timeout, cpu_threads=args.cpu_threads)
        result.update(name=name, report=str(report))
        manifest["steps"].append(result)
        save()
        if result["status"] != "ok":
            manifest["status"] = "failed"
            save()
            raise SystemExit(f"Demo step {name} failed; inspect {result['log']}")
    summary = collect_reports(step["report"] for step in manifest["steps"])
    write_summary(summary, directory)
    manifest["status"] = "complete"
    manifest["summary"] = str(directory / "summary.json")
    save()
    print(json.dumps({"status": manifest["status"], "directory": str(directory),
                      "summary": manifest["summary"], "backend": args.backend}, ensure_ascii=False, indent=2))


def package_source(output, *, include_reports=False):
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    files = set()
    for name in ("README.md", "LICENSE", "AGENTS.md", "proj_requir.md", "requirements.txt",
                 "__init__.py", "run.py", "data.py", "models.py", "prompts.py", "utils.py"):
        files.add(ROOT / name)
    for folder in ("coop", "tools", "tests", "methods"):
        files.update(path for path in (ROOT / folder).rglob("*") if path.suffix in (".py", ".md", ".txt"))
    for folder in ("data",):
        files.update(path for path in (ROOT / folder).rglob("*") if path.suffix in (".json", ".jsonl"))
    if include_reports:
        files.update((ROOT / "results").glob("*.json"))
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "files": [],
                "scope": "Source/docs/checked-in data, optional report JSON; model weights, logs, SQLite/caches and generated ZIPs excluded."}
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                continue
            name = str(path.relative_to(ROOT))
            content = path.read_bytes()
            archive.writestr(name, content)
            manifest["files"].append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"archive": str(target), "file_count": len(manifest["files"]), "bytes": target.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description="LatentMAS competition prototype entrypoint")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect Python/dependencies/model files/proxy/GPU without loading models")
    doctor.add_argument("--backend", choices=("fixture", "qwen"), default="fixture")
    doctor.add_argument("--model-path", default=QWEN)
    doctor.add_argument("--embedding-path", default=EMBEDDING)
    demo_parser = commands.add_parser("demo", help="Run the competition multi-agent prototype")
    demo_parser.add_argument("--backend", choices=("fixture", "qwen"), default="fixture",
                             help="fixture: scripted mechanism check; qwen: local Qwen generation and embeddings")
    demo_parser.add_argument("--output-root", default="results/demo")
    demo_parser.add_argument("--device", default="cuda:0")
    demo_parser.add_argument("--embedding-device", default="cuda:1")
    demo_parser.add_argument("--model-path", default=QWEN)
    demo_parser.add_argument("--embedding-path", default=EMBEDDING)
    demo_parser.add_argument("--max-new-tokens", type=int, default=192)
    demo_parser.add_argument("--cpu-threads", type=int, default=8)
    demo_parser.add_argument("--timeout", type=int, default=1800)
    demo_parser.add_argument("--skip-recovery", action="store_true")
    summarize = commands.add_parser("summarize", help="Summarize existing evidence without rerunning experiments")
    summarize.add_argument("--results-dir", default="results")
    summarize.add_argument("--report", action="append")
    summarize.add_argument("--output", default="results/evidence_summary")
    pack = commands.add_parser("package", help="Build a source/docs ZIP with per-file SHA256 manifest")
    pack.add_argument("--output", default="results/latentmas-prototype-" + uuid4().hex[:8] + ".zip")
    pack.add_argument("--include-reports", action="store_true")
    args = parser.parse_args()
    if args.command == "doctor":
        result = environment_report(backend=args.backend, model_path=args.model_path, embedding_path=args.embedding_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["ready"]:
            raise SystemExit(2)
    elif args.command == "demo":
        if min(args.timeout, args.cpu_threads, args.max_new_tokens) < 1:
            parser.error("timeout, cpu threads and token budget must be positive")
        demo(args)
    elif args.command == "summarize":
        paths = args.report or list(Path(args.results_dir).glob("*.json"))
        result = collect_reports(paths)
        write_summary(result, args.output)
        print(json.dumps({"reports": len(result["reports"]), "skipped": len(result["skipped"]), "output": args.output}))
    else:
        print(json.dumps(package_source(args.output, include_reports=args.include_reports), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
