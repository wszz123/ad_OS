from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from coop.catalog_project import FILES, PATCHES, project_tasks
from coop.catalog_world import CatalogProjectWorld, ScriptedProject
from tools.evaluate_devops_report import evaluate_saved
from tools.run_devops_workload import run_sequence


class CatalogProjectTests(unittest.TestCase):
    def apply_through(self, world, index):
        for task in project_tasks():
            if task["index"] > index:
                break
            world.prepare_task(task)
            if task["group"] == "dev":
                world.execute(task, {"tool": "patch", "patches": deepcopy(PATCHES[task["incident"]])})

    def test_ten_round_project_and_independent_regrade(self):
        with tempfile.TemporaryDirectory() as root:
            report = run_sequence(root, workload="project", max_steps=20)
            self.assertEqual(report["status"], "complete", report)
            self.assertEqual(report["workload"], "constructed_catalog_project_v2")
            self.assertEqual(report["totals"]["passed_tasks"], 10)
            self.assertEqual(report["totals"]["patch_files"], 11)
            self.assertEqual(report["totals"]["probe_operations"], 104)
            self.assertEqual(report["totals"]["effective_cross_group_reuse"], 9)
            self.assertEqual(report["totals"]["context_filtered_reuse"], 2)
            self.assertEqual(report["totals"]["llm_calls"], 0)
            candidate = json.loads((Path(root) / "A5-candidate.json").read_text())
            self.assertEqual(set(candidate), set(FILES))
            self.assertIn("receipts", candidate["gateway.py"])
            world_dir = Path(root) / "world"
            self.assertTrue((world_dir / "observations.jsonl").is_file())
            with sqlite3.connect(world_dir / "catalog.sqlite") as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM records").fetchone()[0], 361)
            regraded = evaluate_saved(Path(root) / "report.json")
            self.assertEqual(regraded["passed_tasks"], 10, regraded)

    def test_single_module_fix_does_not_solve_cross_tenant_cache(self):
        with tempfile.TemporaryDirectory() as root:
            world = CatalogProjectWorld(root)
            task = project_tasks()[1]
            world.execute(task, {"tool": "patch", "patches": [PATCHES["tenant_cache"][0]]})
            suite = world._suite("source", "tenant_cache")
            self.assertFalse(suite["passed"])
            self.assertTrue(all(case["checks"][0]["passed"] for case in suite["cases"]))
            self.assertTrue(all(not case["checks"][1]["passed"] for case in suite["cases"]))

    def test_revision_bump_without_cache_key_revision_remains_stale(self):
        with tempfile.TemporaryDirectory() as root:
            world = CatalogProjectWorld(root)
            self.apply_through(world, 1)
            task = next(t for t in project_tasks() if t["task_id"] == "A3")
            world.prepare_task(task)
            world.execute(task, {"tool": "patch", "patches": [PATCHES["invalidation"][0]]})
            case = world.scenario("source", "invalidation")
            self.assertFalse(case["passed"])
            self.assertFalse(next(c for c in case["checks"] if c["name"] == "revision_advanced_once")["passed"])
            self.assertFalse(next(c for c in case["checks"] if c["name"] == "read_after_write")["passed"])

    def test_receipt_fix_alone_cannot_hide_partial_transaction_writes(self):
        with tempfile.TemporaryDirectory() as root:
            world = CatalogProjectWorld(root)
            self.apply_through(world, 3)
            task = next(t for t in project_tasks() if t["task_id"] == "A5")
            world.execute(task, {"tool": "patch", "patches": [PATCHES["transactions"][1]]})
            case = world.scenario("source", "transactions")
            self.assertEqual(case["worker_restarts"], 1)
            self.assertFalse(case["passed"])
            self.assertFalse(next(c for c in case["checks"] if c["name"] == "atomic_and_replay_first_stock")["passed"])
            self.assertTrue(next(c for c in case["checks"] if c["name"] == "conflicting_replay_rejected")["passed"])

    def test_multi_file_patch_is_atomic_and_paths_imports_are_restricted(self):
        with tempfile.TemporaryDirectory() as root:
            world = CatalogProjectWorld(root)
            version = world.version()
            patches = deepcopy(PATCHES["tenant_cache"])
            patches[1]["old"] = "missing exact source"
            with self.assertRaises(ValueError):
                world.execute(project_tasks()[1], {"tool": "patch", "patches": patches})
            self.assertEqual(world.version(), version)
            for path in ("../outside.py", "/tmp/outside.py", "docs/policy.md"):
                with self.assertRaises(ValueError):
                    world.execute(project_tasks()[1], {"tool": "patch", "patches": [{"path": path, "old": "a", "new": "b"}]})
            bad = dict(FILES)
            bad["repository.py"] = 'from os import system\n' + bad["repository.py"]
            with self.assertRaises(ValueError):
                world.write_bundle("source", bad)
            self.assertEqual(world.version(), version)
            with self.assertRaises(ValueError):
                world.execute(project_tasks()[0], {"tool": "patch", "patches": PATCHES["tenant_cache"]})

    def test_undefined_globals_in_functions_and_nested_comprehensions_reject_whole_patch(self):
        with tempfile.TemporaryDirectory() as root:
            world = CatalogProjectWorld(root)
            version = world.version()
            edits = deepcopy(PATCHES["tenant_cache"])
            edits[1]["new"] = 'str((request["tenant"], state.get("revision")))'
            with self.assertRaisesRegex(ValueError, r"cache.py: undefined globals.*state"):
                world.execute(project_tasks()[1], {"tool": "patch", "patches": edits})
            self.assertEqual(world.version(), version)
            broken = dict(FILES)
            broken["repository.py"] = 'def select(records, tenant):\n    return [state.get("x") for r in records]\n'
            with self.assertRaisesRegex(ValueError, "undefined globals.*state"):
                world.write_bundle("source", broken)
            self.assertEqual(world.version(), version)

    def test_diagnostic_prefix_never_counts_as_full_completion_and_needs_no_unused_encoder(self):
        def unused_encoder(*args, **kwargs):
            raise AssertionError("non-vector run must not compute embeddings")

        class CaptureGuide(ScriptedProject):
            def __init__(self):
                super().__init__()
                self.guides = []

            def decide(self, role, context):
                if role == "planner":
                    self.guides.append(context["guide"])
                return super().decide(role, context)

        with tempfile.TemporaryDirectory() as root:
            decision = CaptureGuide()
            report = run_sequence(root, workload="project", mode="protocol", task_limit=2,
                                  encoder=unused_encoder, max_steps=20, decision=decision)
            self.assertEqual(report["status"], "partial")
            self.assertFalse(report["full_sequence"])
            self.assertEqual([t["task_id"] for t in report["tasks"]], ["B1", "A1"])
            self.assertTrue(decision.guides)
            self.assertTrue(all(guide == CatalogProjectWorld.guide for guide in decision.guides))
            from coop.reporting import summarize_report
            self.assertFalse(summarize_report(report, "prefix.json")["all_quality_passed"])
            result = evaluate_saved(Path(root) / "report.json")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["passed_tasks"], 2)

    def test_broken_module_is_not_accepted_as_incident_reproduction(self):
        with tempfile.TemporaryDirectory() as root:
            world = CatalogProjectWorld(root)
            bad = world.bundle()
            bad["repository.py"] = 'def select(records, tenant):\n    return records[999999]\n'
            world.write_bundle("source", bad)
            task = project_tasks()[0]
            world.execute(task, {"tool": "deploy"})
            observation = world.execute(task, {"tool": "probe"})
            self.assertFalse(observation["actual"])
            self.assertFalse(world.check(task, [observation])["passed"])
            self.assertGreater(observation["suite"]["runtime_errors"], 0)


if __name__ == "__main__":
    unittest.main()
