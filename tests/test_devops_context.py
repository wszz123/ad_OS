from copy import deepcopy
import json
import tempfile
import unittest

from coop.catalog_project import FILES, project_tasks
from coop.catalog_world import CatalogProjectWorld, ScriptedProject
from coop.devops_context import project_context
from coop.devops_graph import QwenDevOps, build_devops_graph
from coop.memory import MemoryAdapter
from coop.protocol import ProtocolAdapter
from tools.run_devops_workload import fixture_embedding


class DevOpsContextTests(unittest.TestCase):
    def test_projection_preserves_every_failure_and_source_without_mutating_raw_evidence(self):
        checks = [{"name": "success", "passed": True, "actual": list(range(1000)), "expected": list(range(1000))}]
        checks += [{"name": "failure-" + str(i), "passed": False, "actual": list(range(30)), "expected": []}
                   for i in range(9)]
        context = {"task": project_tasks()[1], "source": json.dumps(FILES),
                   "observations": [{"suite": {"passed": False, "cases": [{"passed": False, "checks": checks}]}}]}
        original = deepcopy(context)
        projected = project_context(context)
        failures = projected["observations"][0]["suite"]["cases"][0]["failures"]
        self.assertEqual([f["name"] for f in failures], ["failure-" + str(i) for i in range(9)])
        self.assertEqual(failures[0]["actual"]["count"], 30)
        self.assertEqual(projected["source"], context["source"])
        self.assertLess(len(json.dumps(projected)), len(json.dumps(context)))
        self.assertEqual(context, original)

    def test_selected_nonpatch_capabilities_need_no_redundant_model_call(self):
        class NoGeneration:
            call_stats = []

            def generate_json(self, *args, **kwargs):
                raise AssertionError("typed argument-free binding must not generate")

        decision = QwenDevOps(NoGeneration())
        for tool in ("inspect", "deploy", "probe"):
            context = {"task": project_tasks()[0], "action": {"tool": tool}}
            self.assertEqual(decision.decide("executor", context), {"tool": tool})
        self.assertEqual(decision.call_stats, [])
        self.assertEqual(len(decision.prompt_records), 3)

    def test_tool_alias_mapping_keeps_multifile_plan_for_executor(self):
        class FilePlan(ScriptedProject):
            observed_plan = None

            def decide(self, role, context):
                action = super().decide(role, context)
                if role == "planner" and action.get("tool") == "patch":
                    return {"route": "patch", "files": ["repository.py", "cache.py"], "goal": "address independent failures"}
                if role == "executor" and context["action"]["tool"] == "patch":
                    self.observed_plan = context["action"]
                return action

        with tempfile.TemporaryDirectory() as root:
            world, memory, decision = CatalogProjectWorld(root), MemoryAdapter(), FilePlan()
            try:
                graph, _, _ = build_devops_graph(world, memory, ProtocolAdapter(), decision,
                                                 fixture_embedding, run_id="file-plan", max_steps=20)
                result = graph.invoke({"task": project_tasks()[1]})
                self.assertEqual(result["status"], "verified")
                self.assertEqual(decision.observed_plan["files"], ["repository.py", "cache.py"])
                self.assertEqual(decision.observed_plan["goal"], "address independent failures")
            finally:
                memory.close()


if __name__ == "__main__":
    unittest.main()
