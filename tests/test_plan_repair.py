import json
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

from coop.calculator import calculate
from coop.cost_graph import PLAN_PROMPT, REVIEW_PROMPT, build_cost_graph, public_request
from coop.memory import MemoryAdapter
from coop.plan_repair import PlanRepairError, repair_plan
from coop.plan_validation import validate_quantity_contract
from coop.protocol import ProtocolAdapter, Capability, Message
from coop.text_transport import TextAdapter
from coop.workload import cost_documents, cost_tasks, variable_catalog
from tools.run_cost_workload import run_cost_mode


ENERGY = "node_count * power_watts / 1000 * hours_per_day * days * energy_rate"
BAD_ENERGY = "power_watts / 1000 * hours_per_day * days * energy_rate"


class SequenceLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.call_stats = []
        self.prompts = []
        self.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())

    def generate_json(self, messages, **kwargs):
        self.prompts.append(messages)
        self.call_stats.append({"prompt_tokens": 10, "generated_tokens": 3})
        if "Tool result:" in messages[-1]["content"]:
            tool = json.loads(messages[-1]["content"].split("Tool result: ", 1)[1])
            return {"value": {key: tool[key] for key in ("value", "unit", "citations")}}
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return {"value": {"expression": value, "search_query": "cost facts"}}


class PlanRepairTests(unittest.TestCase):
    def repair(self, llm, attempts, max_repairs=2):
        task = cost_tasks()[1]
        return repair_plan(llm, prompt="Entity Orion; calculate monthly electricity for all nodes.",
                           catalog=variable_catalog(task.entity), inputs=task.inputs, question=task.question,
                           plan_prompt=PLAN_PROMPT, review_prompt=REVIEW_PROMPT,
                           max_new_tokens=96, max_repairs=max_repairs, attempts=attempts)

    def test_recovers_after_repeated_invalid_review_and_stops_at_acceptance(self):
        llm, attempts = SequenceLLM([BAD_ENERGY, BAD_ENERGY, ENERGY]), []
        value = self.repair(llm, attempts)
        self.assertEqual(value["expression"], ENERGY)
        self.assertEqual(len(llm.call_stats), 3)
        self.assertTrue(attempts[-1]["accepted"])
        self.assertTrue(attempts[1]["issues"])
        self.assertIn("Earlier attempts", llm.prompts[2][-1]["content"])

    def test_malformed_json_consumes_budget_instead_of_resetting_it(self):
        llm, attempts = SequenceLLM([ValueError("malformed JSON"), BAD_ENERGY, ENERGY]), []
        self.assertEqual(self.repair(llm, attempts)["expression"], ENERGY)
        self.assertIn("generation_error", attempts[0])
        self.assertEqual(len(attempts), 3)

    def test_budget_exhaustion_is_explicit_and_bounded(self):
        llm, attempts = SequenceLLM([BAD_ENERGY] * 20), []
        with self.assertRaisesRegex(PlanRepairError, "budget exhausted"):
            self.repair(llm, attempts)
        self.assertEqual(len(llm.call_stats), 3)
        self.assertEqual(len(attempts), 3)
        for limit in (0, 6, True):
            with self.assertRaises(ValueError):
                self.repair(SequenceLLM([]), [], max_repairs=limit)

    def test_public_quantity_contract_detects_extra_replica_copy(self):
        valid = "daily_gb * retention_days * replicas * storage_rate + daily_gb * days * egress_fraction * egress_rate"
        validate_quantity_contract(valid, "Calculate monthly total.")
        with self.assertRaisesRegex(ValueError, "replicas"):
            validate_quantity_contract(valid.replace("replicas", "(1 + replicas)"), "Calculate monthly total.")
        with self.assertRaisesRegex(ValueError, "excludes recurring"):
            validate_quantity_contract("node_count * node_price + monthly_maintenance", "Exclude recurring costs.")
        with self.assertRaisesRegex(ValueError, "original data"):
            validate_quantity_contract("daily_gb * days * egress_fraction * egress_rate * replicas", "Calculate traffic.")

    def test_graph_failure_does_not_compute_or_write_memory_and_counts_request(self):
        corpus, memory = MemoryAdapter(), MemoryAdapter()
        try:
            llm, transport = SequenceLLM([BAD_ENERGY] * 3), ProtocolAdapter()
            graph, _, events = build_cost_graph(llm, corpus, memory, transport, namespace="test",
                                               repair_plan=True, max_plan_repairs=2, reuse=True)
            with self.assertRaises(PlanRepairError):
                graph.invoke(public_request(cost_tasks()[1]))
            event = events["orion-02"]
            self.assertEqual(event["plan_attempt_count"], 3)
            self.assertEqual(event["plan_repair_exhausted"], 1)
            self.assertEqual(event["tool_calls"], 0)
            self.assertEqual(len(transport.frames), 1)
            self.assertEqual(memory.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 0)
        finally:
            corpus.close()
            memory.close()

    def test_text_and_protocol_keep_failed_requests_for_actual_communication_cost(self):
        for transport in (TextAdapter(), ProtocolAdapter()):
            def fail(params):
                raise ValueError("failed")
            transport.register(Capability("agent", "planner", ("plan",)), {"plan": fail})
            with self.assertRaises(ValueError):
                transport.dispatch(Message("agent", "agent", "plan", {"query": "test"}))
            self.assertEqual(len(transport.frames), 1)

    def test_runner_counts_correction_calls_and_never_passes_gold_to_model(self):
        task = cost_tasks()[1]
        llm = SequenceLLM([BAD_ENERGY, BAD_ENERGY, ENERGY])
        docs = cost_documents()
        with tempfile.TemporaryDirectory() as directory:
            report = run_cost_mode(llm, None, docs, [task], Path(directory) / "run", "protocol",
                                   namespace="test", repair_plan=True, max_plan_repairs=2)
            self.assertEqual(report["model_calls"], 4)
            self.assertEqual(report["plan_repairs"], 2)
            self.assertEqual(report["plan_attempt_count"], 3)
            self.assertEqual(report["passed_count"], 1)
            plan_prompts = json.dumps(llm.prompts[:3])
            self.assertNotIn(task.expected_value, plan_prompts)
            self.assertNotIn("expected_value", plan_prompts)
            self.assertNotIn("required_citations", plan_prompts)
            self.assertEqual(report["tasks"][0]["plan_attempts"][2]["candidate"]["expression"], ENERGY)


if __name__ == "__main__":
    unittest.main()
