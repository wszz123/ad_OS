"""Asynchronous sequence jobs; independent worlds, serialized shared model access."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
from uuid import uuid4


class DevOpsJobs:
    def __init__(self, root, execution_lock):
        self.root = Path(root)
        self.execution_lock = execution_lock
        self.lock = threading.RLock()
        self.records = {}

    def get(self, job_id):
        with self.lock:
            record = self.records.get(job_id)
            return json.loads(json.dumps(record)) if record is not None else None

    def start(self, *, mode, max_steps, backend, decision, encoder, workload="service"):
        from tools.run_devops_workload import MODES, run_sequence
        if mode not in MODES:
            raise ValueError("unknown devops mode")
        if type(max_steps) is not int or not 1 <= max_steps <= 100:
            raise ValueError("max_steps must be an integer between 1 and 100")
        if workload not in ("service", "project"):
            raise ValueError("workload must be service or project")
        job_id = uuid4().hex[:12]
        record = {"job_id": job_id, "status": "queued", "backend": backend, "mode": mode, "workload": workload,
                  "created_at": datetime.now(timezone.utc).isoformat(), "report": None, "error": None}
        with self.lock:
            self.records[job_id] = record

        def update(report):
            with self.lock:
                record.update(status="running", report=report)

        def execute():
            try:
                with self.execution_lock:
                    with self.lock:
                        record["status"] = "running"
                    report = run_sequence(self.root / job_id, mode=mode, max_steps=max_steps,
                                          backend=backend, decision=decision, encoder=encoder,
                                          on_update=update, workload=workload)
                    with self.lock:
                        record.update(status=report["status"], report=report,
                                      finished_at=datetime.now(timezone.utc).isoformat())
            except Exception as exc:
                with self.lock:
                    record.update(status="error", error=f"{type(exc).__name__}: {exc}")

        threading.Thread(target=execute, name="devops-" + job_id, daemon=True).start()
        return self.get(job_id)

    def task_detail(self, job_id, task_id):
        record = self.get(job_id)
        if record is None or re.fullmatch(r"[AB][1-5]", task_id) is None:
            return None
        root = self.root / job_id
        trace = root / (task_id + "-trace.json")
        extension = ".json" if record.get("workload") == "project" else ".py"
        if not all((root / (task_id + suffix)).is_file()
                   for suffix in ("-trace.json", "-candidate" + extension, "-deployed" + extension)):
            return None
        result = {"task_id": task_id, "trace": json.loads(trace.read_text()),
                  "candidate": (root / (task_id + "-candidate" + extension)).read_text(),
                  "deployed": (root / (task_id + "-deployed" + extension)).read_text()}
        if extension == ".json":
            result.update(candidate_files=json.loads(result["candidate"]), deployed_files=json.loads(result["deployed"]))
        return result
