import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import shutil

from coop.__main__ import ROOT, package_source, run_step
from coop.demo_support import environment_report
from coop.reporting import collect_reports, summarize_report, write_summary


class DemoEntrypointTests(unittest.TestCase):
    def test_help_and_diagnostics_work_without_site_packages(self):
        help_result = subprocess.run([sys.executable, "-S", "-m", "coop", "--help"], cwd=ROOT,
                                     capture_output=True, text=True, timeout=30)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("doctor", help_result.stdout)
        diagnostic = subprocess.run([sys.executable, "-S", "-m", "coop", "doctor"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(diagnostic.returncode, 2)
        data = json.loads(diagnostic.stdout)
        self.assertFalse(data["ready"])
        self.assertIn("torch", data["missing_packages"])

    def test_fixture_demo_executes_real_graph_and_labels_non_model_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-m", "coop", "demo", "--backend", "fixture",
                "--skip-recovery", "--output-root", directory, "--timeout", "120"], cwd=ROOT,
                capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run_directory = next(Path(directory).iterdir())
            manifest = json.loads((run_directory / "demo.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            summary = json.loads((run_directory / "summary.json").read_text())
            workload = summary["reports"][0]
            self.assertEqual(workload["evidence"], "scripted_fixture")
            self.assertTrue(workload["all_quality_passed"])
            self.assertEqual(workload["runtime"]["vector_dimensions"], 2)
            self.assertEqual(workload["metrics"]["protocol_vector_memory"]["accepted_reuse"], 4)
            self.assertTrue((run_directory / "protocol_vector_memory" / "repair_ledger.sqlite").is_file())

    def test_frontend_assets_are_served_from_local_bundle(self):
        from coop.web_server import Handler, ThreadingHTTPServer
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        import threading
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            from urllib.request import urlopen
            for path, content_type in (("/", "text/html"), ("/assets/styles.css", "text/css"),
                                       ("/assets/app.js", "text/javascript")):
                with urlopen(f"http://127.0.0.1:{server.server_port}{path}", timeout=10) as response:
                    self.assertIn(content_type, response.headers.get("Content-Type"))
                    self.assertGreater(len(response.read()), 100)
        finally:
            server.shutdown()
            server.server_close()

    def test_summary_retains_failure_and_skips_unsupported_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "failed.json").write_text(json.dumps({"experiment": "four_role_sourced_cost_workload",
                "demo_backend": "fixture", "trials": [{"modes": {"protocol": {"task_count": 10,
                    "completed": 8, "passed_count": 7}}}]}))
            (path / "unknown.json").write_text(json.dumps({"experiment": "untrusted_old_proof"}))
            summary = collect_reports(path.glob("*.json"))
            self.assertFalse(summary["reports"][0]["all_quality_passed"])
            self.assertEqual(len(summary["skipped"]), 1)
            write_summary(summary, path / "summary")
            self.assertIn("False", (path / "summary" / "summary.md").read_text())

    def test_process_timeout_is_reported_and_owned_process_is_reaped(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_step([sys.executable, "-c", "import time; time.sleep(30)"],
                              Path(directory) / "timeout.log", timeout=0.1, cpu_threads=1)
            self.assertEqual(result["status"], "timeout")
            self.assertNotEqual(result["return_code"], 0)

    def test_source_package_manifest_matches_members_and_excludes_runtime_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            (root / "coop" / "__pycache__").mkdir(parents=True)
            shutil.copytree(ROOT / "web", root / "web", dirs_exist_ok=True)
            (root / "results").mkdir()
            (root / "coop" / "entry.py").write_text("print('fixture')\n")
            (root / "coop" / "state.sqlite").write_text("excluded")
            (root / "coop" / "model.safetensors").write_text("excluded")
            (root / "coop" / "__pycache__" / "entry.pyc").write_text("excluded")
            (root / "results" / "report.json").write_text("{}")
            with patch("coop.__main__.ROOT", root):
                result = package_source(Path(directory) / "package.zip", include_reports=True)
            with zipfile.ZipFile(result["archive"]) as archive:
                manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
                names = archive.namelist()
                self.assertIn("coop/entry.py", names)
                self.assertIn("web/index.html", names)
                self.assertIn("web/assets/app.js", names)
                self.assertIn("results/report.json", names)
                self.assertFalse(any(name.endswith((".sqlite", ".pyc", ".safetensors")) for name in names))
                for item in manifest["files"]:
                    self.assertEqual(hashlib.sha256(archive.read(item["path"])).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()
