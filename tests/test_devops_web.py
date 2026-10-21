import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from coop import web_server
from coop.devops_web import DevOpsJobs


class DevOpsWebTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.jobs = DevOpsJobs(self.directory.name, web_server.EXECUTION_LOCK)
        self.patcher = patch.object(web_server, "DEVOPS_JOBS", self.jobs)
        self.patcher.start()
        web_server.start_engine("fixture")
        self.server = web_server.ThreadingHTTPServer(("127.0.0.1", 0), web_server.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.patcher.stop()
        self.directory.cleanup()

    def request(self, path, payload=None):
        request = Request(self.base + path, data=json.dumps(payload).encode() if payload is not None else None,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())

    def wait(self, job_id):
        for _ in range(200):
            _, job = self.request("/api/devops/runs/" + job_id)
            if job["status"] not in ("queued", "running"):
                return job
            time.sleep(.05)
        self.fail("sequence failed to finish")

    def test_async_sequence_and_independent_task_artifacts(self):
        _, data = self.request("/api/devops/tasks")
        self.assertEqual([t["task_id"] for t in data["tasks"]],
                         [name + str(i) for i in range(1, 6) for name in ("B", "A")])
        status, job = self.request("/api/devops/runs", {"mode": "protocol_vector_memory", "max_steps": 20})
        self.assertEqual(status, 202)
        self.assertTrue(self.request("/api/health")[1]["ok"])
        job = self.wait(job["job_id"])
        self.assertEqual(job["status"], "complete", job)
        self.assertEqual(job["report"]["totals"]["passed_tasks"], 10)
        self.assertEqual(job["report"]["totals"]["effective_cross_group_reuse"], 9)
        _, detail = self.request(f"/api/devops/runs/{job['job_id']}/tasks/A1")
        self.assertIn(".strip()", detail["candidate"])
        self.assertNotIn(".strip()", detail["deployed"])
        self.assertGreater(len(detail["trace"]), 0)
        self.assertTrue((Path(self.directory.name) / job["job_id"] / "report.json").is_file())

    def test_budget_failure_is_visible_and_invalid_requests_are_rejected(self):
        for payload in ({"mode": "unknown"}, {"max_steps": 0}, {"max_steps": True}, {"workload": "unknown"}, ["invalid"]):
            with self.assertRaises(HTTPError) as error:
                self.request("/api/devops/runs", payload)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
        _, job = self.request("/api/devops/runs", {"max_steps": 1})
        job = self.wait(job["job_id"])
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["report"]["tasks"][0]["status"], "budget_exhausted")
        self.assertTrue(all(row["status"] == "blocked" for row in job["report"]["tasks"][1:]))
        for path in ("/api/devops/runs/missing", f"/api/devops/runs/{job['job_id']}/tasks/unknown"):
            with self.assertRaises(HTTPError) as error:
                self.request(path)
            self.assertEqual(error.exception.code, 404)
            error.exception.close()

    def test_project_workload_and_multi_file_artifacts(self):
        _, data = self.request("/api/devops/tasks?workload=project")
        self.assertEqual(data["scope"], "constructed_catalog_project_v2")
        self.assertEqual(data["tasks"][0]["incident"], "tenant_cache")
        _, job = self.request("/api/devops/runs", {"workload": "project", "max_steps": 20})
        job = self.wait(job["job_id"])
        self.assertEqual(job["status"], "complete", job)
        self.assertEqual(job["report"]["totals"]["patch_files"], 11)
        self.assertEqual(job["report"]["totals"]["probe_operations"], 104)
        _, detail = self.request(f"/api/devops/runs/{job['job_id']}/tasks/A1")
        self.assertEqual(len([name for name in detail["candidate_files"] if name.endswith(".py")]), 8)
        self.assertIn('r["tenant"] == tenant', detail["candidate_files"]["repository.py"])
        self.assertNotIn('r["tenant"] == tenant', detail["deployed_files"]["repository.py"])

    def test_devops_static_bundle(self):
        for path, content_type in (("/devops", "text/html"), ("/assets/devops.js", "text/javascript"),
                                   ("/assets/devops.css", "text/css")):
            with urlopen(self.base + path, timeout=10) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(content_type, response.headers["Content-Type"])
                self.assertGreater(len(response.read()), 100)


if __name__ == "__main__":
    unittest.main()
