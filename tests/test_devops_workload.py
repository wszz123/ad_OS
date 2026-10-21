import json
from pathlib import Path
import tempfile
import unittest

from coop.devops_graph import ScriptedDevOps, build_devops_graph
from coop.devops_world import DevOpsWorld, RECIPES, SOURCE, tasks
from coop.memory import MemoryAdapter, MemoryItem
from coop.protocol import ProtocolAdapter
from tools.run_devops_workload import fixture_embedding, run_sequence
from tools.evaluate_devops_report import evaluate_saved


class DevOpsWorkloadTests(unittest.TestCase):
    def test_ten_tasks_and_cross_group_ablation(self):
        with tempfile.TemporaryDirectory() as root:
            reports = {mode: run_sequence(Path(root) / mode, mode=mode) for mode in (
                "protocol_vector", "protocol_vector_group_memory", "protocol_vector_memory")}
            for mode, report in reports.items():
                self.assertEqual(report["status"], "complete", report)
                self.assertEqual(report["totals"]["passed_tasks"], 10)
                self.assertEqual([r["group"] for r in report["tasks"]], ["ops", "dev"] * 5)
                self.assertEqual(report["totals"]["llm_calls"], 0)
                self.assertEqual(report["totals"]["state_transfers"], 10)
                reproduction = report["tasks"][0]["grade"]["checks"][0]
                self.assertTrue(reproduction["passed"])
                self.assertFalse(reproduction["behavior_correct"])
                self.assertEqual(reproduction["criterion"], "reproduce_incident")
                self.assertTrue((Path(root) / mode / "A5-candidate.py").exists())
            self.assertEqual(reports["protocol_vector"]["totals"]["source_context_reads"], 10)
            group = reports["protocol_vector_group_memory"]["totals"]
            shared = reports["protocol_vector_memory"]["totals"]
            self.assertEqual(group["cross_group_reuse"], 0)
            self.assertEqual(shared["effective_cross_group_reuse"], 9)
            self.assertEqual(shared["source_context_reads"], 1)
            self.assertGreater(group["source_context_reads"], shared["source_context_reads"])
            self.assertTrue(any(r["rejected_memory_ids"] for r in reports["protocol_vector_memory"]["tasks"]))
            path = Path(root) / "protocol_vector_memory" / "report.json"
            self.assertEqual(evaluate_saved(path)["passed_tasks"], 10)
            snapshot = path.parent / "A1-candidate.py"
            snapshot.write_text(snapshot.read_text() + "\n# changed artifact")
            regraded = evaluate_saved(path)
            self.assertFalse(regraded["tasks"][1]["passed"])
            self.assertIn("snapshot does not match", regraded["tasks"][1]["error"])

    def test_failure_blocks_dependent_tasks_without_reference_patch(self):
        with tempfile.TemporaryDirectory() as root:
            report = run_sequence(root, max_steps=1)
            self.assertEqual(report["tasks"][0]["status"], "budget_exhausted")
            self.assertEqual(report["totals"]["passed_tasks"], 0)
            self.assertTrue(all(r["status"] == "blocked" for r in report["tasks"][1:]))
            self.assertEqual((Path(root) / "world" / "candidate.py").read_text(), SOURCE)
            memory = MemoryAdapter(str(Path(root) / "memory.sqlite"))
            try:
                self.assertEqual(memory.count(), 0)
            finally:
                memory.close()

    def test_stale_or_tampered_memory_is_not_consumed(self):
        with tempfile.TemporaryDirectory() as root:
            world, memory = DevOpsWorld(Path(root) / "world"), MemoryAdapter()
            try:
                task = tasks()[0]
                graph, _, _ = build_devops_graph(world, memory, ProtocolAdapter(), ScriptedDevOps(),
                                                 fixture_embedding, run_id="test")
                initial = graph.invoke({"task": task})
                self.assertTrue(world.evaluate(task, initial["observations"])["passed"])
                memory.put(MemoryItem(**initial["memory_candidate"]))
                item = memory.items()[0]
                record = json.loads(item.evidence)
                record["source"] += "\n# corrupted snapshot"
                item.evidence = json.dumps(record)
                memory.put(item)
                graph, _, metrics = build_devops_graph(world, memory, ProtocolAdapter(), ScriptedDevOps(),
                                                       fixture_embedding, run_id="test")
                result = graph.invoke({"task": tasks()[1]})
                self.assertEqual(result["status"], "verified")
                self.assertEqual(metrics["A1"]["accepted_reuse"], 0)
                self.assertEqual(metrics["A1"]["rejected_memory_ids"], [item.memory_id])
            finally:
                memory.close()

    def test_role_permissions_and_code_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            world = DevOpsWorld(root)
            with self.assertRaises(ValueError):
                world.execute(tasks()[0], {"tool": "patch", "old": "lower", "new": "casefold"})
            with self.assertRaises(ValueError):
                world.execute(tasks()[1], {"tool": "deploy"})
            with self.assertRaises(ValueError):
                world.validate("def search(records, request):\n    return open('/etc/passwd').read()\n")
            with self.assertRaises(ValueError):
                world.validate("import os\n" + SOURCE)

    def test_negative_limit_fix_without_upper_bound_fails_public_and_holdout_checks(self):
        with tempfile.TemporaryDirectory() as root:
            world = DevOpsWorld(root)
            source = SOURCE.replace(*RECIPES["query"]).replace(*RECIPES["offset"])
            source = source.replace('int(request.get("limit", 20))', 'max(0, int(request.get("limit", 20)))')
            world.source.write_text(source)
            task = next(t for t in tasks() if t["task_id"] == "A3")
            public = world.check(task, [])
            self.assertTrue(all(check["passed"] for check in public["checks"]))
            self.assertFalse(public["passed"])
            self.assertFalse(next(check for check in public["boundary_checks"]
                                  if check["criterion"] == "limit_upper_bound")["passed"])
            self.assertFalse(world.evaluate(task, [])["passed"])

    def test_budget_is_checked_before_tool_side_effects(self):
        with tempfile.TemporaryDirectory() as root:
            world, memory = DevOpsWorld(root), MemoryAdapter()
            try:
                graph, _, _ = build_devops_graph(world, memory, ProtocolAdapter(), ScriptedDevOps(),
                                                 fixture_embedding, run_id="budget", max_steps=2)
                result = graph.invoke({"task": tasks()[1]})
                self.assertEqual(result["steps"], 2)
                self.assertEqual(result["status"], "budget_exhausted")
                self.assertEqual(world.tool_calls, 0)
                self.assertEqual(world.source.read_text(), SOURCE)
            finally:
                memory.close()

    def test_tool_route_mapping_preserves_original_wire_response(self):
        class ToolRoute(ScriptedDevOps):
            def decide(self, role, context):
                action = super().decide(role, context)
                if role == "planner" and action.get("route") == "execute":
                    return {"route": action["tool"]}
                return action

        with tempfile.TemporaryDirectory() as root:
            world, memory, transport = DevOpsWorld(root), MemoryAdapter(), ProtocolAdapter()
            try:
                graph, _, _ = build_devops_graph(world, memory, transport, ToolRoute(),
                                                 fixture_embedding, run_id="mapping")
                result = graph.invoke({"task": tasks()[0]})
                self.assertEqual(result["status"], "verified")
                replies = [json.loads(frame) for frame in transport.frames if '"in_reply_to"' in frame]
                self.assertTrue(any(json.loads(reply["result"]["result"]) == {"route": "probe"}
                                    for reply in replies))
            finally:
                memory.close()

    def test_past_experience_cannot_replace_current_task_probe(self):
        class SkipProbe(ScriptedDevOps):
            verifier_calls = 0

            def decide(self, role, context):
                if role == "verifier":
                    self.verifier_calls += 1
                if role == "planner" and context["deployment_completed"]:
                    return {"route": "verify"}
                return super().decide(role, context)

        with tempfile.TemporaryDirectory() as root:
            world, memory, decision = DevOpsWorld(root), MemoryAdapter(), SkipProbe()
            try:
                graph, _, _ = build_devops_graph(world, memory, ProtocolAdapter(), decision,
                                                 fixture_embedding, run_id="missing-evidence", max_steps=8)
                result = graph.invoke({"task": tasks()[0]})
                self.assertEqual(result["status"], "budget_exhausted")
                self.assertEqual(decision.verifier_calls, 0)
                self.assertEqual(world.tool_calls, 1)
                self.assertTrue(any("planner_error" in o for o in result["observations"]))
            finally:
                memory.close()

    def test_runtime_error_probe_reopens_repair_instead_of_trapping_in_probe_loop(self):
        class RepairAfterError(ScriptedDevOps):
            patches = 0

            def decide(self, role, context):
                if role == "planner":
                    observations = context["observations"]
                    if not observations or observations[-1].get("tool") == "probe" and "error" in observations[-1]:
                        return {"route": "execute", "tool": "patch"}
                    if "source_version" in observations[-1]:
                        return {"route": "execute", "tool": "probe"}
                    return {"route": "verify"}
                if role == "executor" and context["action"]["tool"] == "patch":
                    self.patches += 1
                    if self.patches == 1:
                        revised = SOURCE.replace(*RECIPES["query"]).replace('request.get("offset") or 1', '"bad"')
                        return {"tool": "patch", "old": SOURCE, "new": revised}
                    return {"tool": "patch", "old": 'offset = "bad"', "new": 'offset = request.get("offset") or 1'}
                return super().decide(role, context)

        with tempfile.TemporaryDirectory() as root:
            world, memory = DevOpsWorld(root), MemoryAdapter()
            try:
                graph, _, _ = build_devops_graph(world, memory, ProtocolAdapter(), RepairAfterError(),
                                                 fixture_embedding, run_id="repair-error", max_steps=16)
                result = graph.invoke({"task": tasks()[1]})
                self.assertEqual(result["status"], "verified", result)
                self.assertTrue(any(o.get("tool") == "probe" and "error" in o for o in result["observations"]))
                self.assertTrue(world.evaluate(tasks()[1], result["observations"])["passed"])
            finally:
                memory.close()

    def test_revisiting_a_source_hash_still_requires_a_new_probe(self):
        class RepeatNoop(ScriptedDevOps):
            def __init__(self):
                super().__init__()
                self.tools = []

            def decide(self, role, context):
                if role == "planner":
                    return {"route": "execute", "tool": context["allowed_tools"][0]}
                if role == "executor":
                    self.tools.append(context["action"]["tool"])
                    if context["action"]["tool"] == "patch":
                        return {"tool": "patch", "old": SOURCE, "new": SOURCE}
                return super().decide(role, context)

        with tempfile.TemporaryDirectory() as root:
            world, memory, decision = DevOpsWorld(root), MemoryAdapter(), RepeatNoop()
            try:
                graph, _, _ = build_devops_graph(world, memory, ProtocolAdapter(), decision,
                                                 fixture_embedding, run_id="revisited-version", max_steps=12)
                result = graph.invoke({"task": tasks()[1]})
                self.assertEqual(result["status"], "budget_exhausted")
                self.assertEqual(decision.tools[:4], ["patch", "probe", "patch", "probe"])
            finally:
                memory.close()

    def test_model_commentary_does_not_become_verified_memory_fact(self):
        class Overclaim(ScriptedDevOps):
            def decide(self, role, context):
                if role == "verifier":
                    return {"accepted": True, "summary": "Every bug is fixed and deployed everywhere."}
                return super().decide(role, context)

        with tempfile.TemporaryDirectory() as root:
            world, memory = DevOpsWorld(root), MemoryAdapter()
            try:
                graph, _, _ = build_devops_graph(world, memory, ProtocolAdapter(), Overclaim(),
                                                 fixture_embedding, run_id="factual-memory")
                result = graph.invoke({"task": tasks()[0]})
                self.assertEqual(result["status"], "verified")
                evidence = json.loads(result["memory_candidate"]["evidence"])
                self.assertIn("remains unresolved", evidence["summary"])
                self.assertNotIn("Every bug is fixed", evidence["summary"])
                self.assertEqual(evidence["summary_origin"], "executable_task_objective")
                self.assertIn("Every bug is fixed", result["summary"])
            finally:
                memory.close()

    def test_incorrect_repair_cannot_pass_verifier_or_write_memory(self):
        class WrongPatch(ScriptedDevOps):
            def decide(self, role, context):
                if role == "verifier":
                    return {"accepted": True, "summary": "Unjustified success claim"}
                if role == "executor" and context["action"]["tool"] == "patch":
                    return {"tool": "patch", "old": '.lower()', "new": '.lower()'}
                return super().decide(role, context)

        with tempfile.TemporaryDirectory() as root:
            world, memory = DevOpsWorld(root), MemoryAdapter()
            try:
                graph, _, _ = build_devops_graph(world, memory, ProtocolAdapter(), WrongPatch(),
                                                 fixture_embedding, run_id="wrong", max_steps=8)
                result = graph.invoke({"task": tasks()[1]})
                self.assertEqual(result["status"], "budget_exhausted")
                self.assertFalse(result["verification"].get("passed", False))
                self.assertEqual(memory.count(), 0)
            finally:
                memory.close()


if __name__ == "__main__":
    unittest.main()
