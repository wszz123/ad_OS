from dataclasses import replace
import json
import unittest

from coop.answer_validation import validate_cost_answer
from coop.calculator import calculate
from coop.cost_graph import build_cost_graph, document_snapshot, public_request, read_facts
from coop.memory import MemoryAdapter
from coop.protocol import ProtocolAdapter
from coop.plan_validation import validate_plan_expression
from coop.text_transport import TextAdapter
from coop.workload import cost_documents, cost_tasks


# Independent formulas only in tests, never provided to runtime agents.
FORMULAS = [
    "node_count * node_price + network_setup",
    "node_count * power_watts / 1000 * hours_per_day * days * energy_rate",
    "node_count * node_price + network_setup + node_count * power_watts / 1000 * hours_per_day * days * energy_rate + monthly_maintenance",
    "node_count * node_price + network_setup + months * (node_count * power_watts / 1000 * hours_per_day * days * energy_rate + monthly_maintenance)",
    "node_count * node_price + network_setup + months * (node_count * power_watts / 1000 * hours_per_day * days * energy_rate + monthly_maintenance)",
    "daily_gb * retention_days * replicas * storage_rate",
    "daily_gb * days * egress_fraction * egress_rate",
    "daily_gb * retention_days * replicas * storage_rate + daily_gb * days * egress_fraction * egress_rate",
    "months * (daily_gb * retention_days * replicas * storage_rate + daily_gb * days * egress_fraction * egress_rate)",
    "daily_gb * retention_days * replicas * storage_rate",
]


class CostLLM:
    def __init__(self, expressions):
        self.expressions = iter(expressions)
        self.prompts = []

    def generate_json(self, messages, *, max_new_tokens):
        self.prompts.append(messages)
        if "search_query" in messages[0]["content"]:
            return {"value": {"search_query": "cost facts", "expression": next(self.expressions)}}
        result = json.loads(messages[-1]["content"].split("Tool result: ", 1)[1])
        return {"value": {"value": result["value"], "unit": result["unit"],
                          "citations": result["citations"], "explanation": "Sourced calculation."}}


class CostWorkloadTests(unittest.TestCase):
    def setUp(self):
        self.docs = cost_documents()
        for doc in self.docs:
            doc.embedding = [1.0, 0.0] if doc.task_topic == "Orion" else [0.0, 1.0]
        self.corpus = MemoryAdapter()
        self.memory = MemoryAdapter()
        for doc in self.docs:
            self.corpus.put(doc)

    def tearDown(self):
        self.corpus.close()
        self.memory.close()

    def test_gold_values_and_source_chains_are_independently_reproducible(self):
        for task, formula in zip(cost_tasks(), FORMULAS):
            facts = read_facts([document_snapshot(doc) for doc in self.docs if doc.task_topic == task.entity], task.entity)
            result = calculate(formula, task.inputs, facts)
            self.assertEqual(result["value"], task.expected_value)
            self.assertEqual(set(result["citations"]), set(task.required_citations))
            self.assertTrue(validate_cost_answer(task, result, result)["passed"])
            self.assertNotIn("expected_value", public_request(task))
            self.assertNotIn("required_citations", public_request(task))

    def test_calculator_rejects_code_unknown_variables_and_source_override(self):
        for expression in ("__import__('os').system('id')", "2 ** 1000", "[1, 2]", "True", "missing + 1", "1 / 0"):
            with self.assertRaises(ValueError):
                calculate(expression, {}, {})
        with self.assertRaises(ValueError):
            calculate("node_price", {"node_price": "1"}, {"node_price": {"value": "2400", "source_id": "doc"}})

    def test_grader_rejects_wrong_amount_units_missing_and_invented_citations(self):
        task = cost_tasks()[0]
        result = {"value": task.expected_value, "unit": "CNY", "citations": list(task.required_citations)}
        for answer in ({**result, "value": "21201"}, {**result, "value": True},
                       {**result, "value": "NaN"}, {**result, "unit": "USD"},
                       {**result, "citations": []}, {**result, "citations": ["invented"]}):
            self.assertFalse(validate_cost_answer(task, answer, result)["passed"])
        self.assertFalse(validate_cost_answer(task, {"decimal string": task.expected_value,
                                                   "unit": "CNY", "citations": list(task.required_citations)}, result)["passed"])
        self.assertFalse(validate_cost_answer(task, result, {**result, "value": "0"})["passed"])

    def test_ten_four_role_tasks_and_identical_text_protocol_model_prompts(self):
        outputs, prompts = [], []
        for transport in (TextAdapter(), ProtocolAdapter()):
            llm = CostLLM(FORMULAS)
            graph, _, _ = build_cost_graph(llm, self.corpus, self.memory, transport, namespace="test")
            rows = []
            for task in cost_tasks():
                if isinstance(transport, TextAdapter):
                    transport.begin_task()
                result = graph.invoke(public_request(task))
                answer, tool = json.loads(result["answer_json"]), json.loads(result["tool_json"])
                self.assertTrue(validate_cost_answer(task, answer, tool)["passed"])
                rows.append(answer)
            self.assertEqual(len(transport.frames), 80)
            self.assertEqual({item.role for item in transport.capabilities.values()},
                             {"runtime", "planning", "retrieval", "tool_execution", "summary"})
            outputs.append(rows)
            prompts.append(llm.prompts)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(prompts[0], prompts[1])

    def test_reuse_recalculates_changed_parameters_and_invalidates_modified_source(self):
        tasks = cost_tasks()
        llm = CostLLM([FORMULAS[0], FORMULAS[0], FORMULAS[0]])
        graph, packets, events = build_cost_graph(
            llm, self.corpus, self.memory, ProtocolAdapter(), namespace="test", reuse=True,
            encoder=lambda texts, query: [[1.0, 0.0]],
        )
        graph.invoke(public_request(tasks[0]))
        changed = replace(tasks[0], task_id="changed-input", inputs={"node_count": "12"}, expected_value="30800.00")
        result = graph.invoke(public_request(changed))
        self.assertTrue(validate_cost_answer(changed, json.loads(result["answer_json"]), json.loads(result["tool_json"]))["passed"])
        self.assertEqual(events["changed-input"]["accepted_reuse"], 1)
        self.assertEqual(events["changed-input"]["corpus_searches"], 0)
        self.assertEqual(events["changed-input"]["tool_calls"], 1)
        self.assertGreater(events["changed-input"]["cache_validation_parses"], 0)
        self.assertEqual(packets["changed-input:facts-query"].target_agent, "retriever")
        original = self.corpus.get("orion-price")
        self.corpus.put(replace(original, summary="Updated node price", evidence=json.dumps({"entity": "Orion", "facts": {"node_price": "2500"}})))
        new = replace(tasks[0], task_id="changed-source")
        result = graph.invoke(public_request(new))
        self.assertEqual(events["changed-source"]["accepted_reuse"], 0)
        self.assertEqual(json.loads(result["tool_json"])["value"], "22000.00")

    def test_memory_missing_required_variables_does_not_short_circuit_retrieval(self):
        llm = CostLLM([FORMULAS[0], FORMULAS[1]])
        graph, _, events = build_cost_graph(
            llm, self.corpus, self.memory, ProtocolAdapter(), namespace="test", reuse=True,
            encoder=lambda texts, query: [[1.0, 0.0]],
        )
        graph.invoke(public_request(cost_tasks()[0]))
        graph.invoke(public_request(cost_tasks()[1]))
        self.assertEqual(events["orion-02"]["accepted_reuse"], 0)
        self.assertEqual(events["orion-02"]["corpus_searches"], 1)

    def test_planner_cannot_mix_another_entitys_variables(self):
        graph, _, _ = build_cost_graph(CostLLM(["daily_gb * storage_rate"]), self.corpus,
                                       self.memory, ProtocolAdapter(), namespace="test")
        with self.assertRaisesRegex(ValueError, "outside entity schema"):
            graph.invoke(public_request(cost_tasks()[0]))

    def test_unit_checks_reject_per_node_total_and_unused_inputs_without_gold(self):
        task = cost_tasks()[1]
        validate_plan_expression(FORMULAS[1], task.inputs)
        with self.assertRaises(ValueError):
            validate_plan_expression("power_watts / 1000 * hours_per_day * days * energy_rate", task.inputs)
        with self.assertRaises(ValueError):
            validate_plan_expression("node_price * node_count + power_watts", {"node_count": "8"})

    def test_optional_plan_review_records_draft_and_uses_corrected_expression(self):
        llm = CostLLM(["power_watts / 1000 * hours_per_day * days * energy_rate", FORMULAS[1]])
        graph, _, events = build_cost_graph(llm, self.corpus, self.memory, ProtocolAdapter(),
                                           namespace="test", review_plan=True)
        task = cost_tasks()[1]
        result = graph.invoke(public_request(task))
        self.assertTrue(validate_cost_answer(task, json.loads(result["answer_json"]), json.loads(result["tool_json"]))["passed"])
        self.assertEqual(len(llm.prompts), 3)
        self.assertEqual(events[task.task_id]["plan_reviews"], 1)
        self.assertNotEqual(events[task.task_id]["draft_expression"], result["expression"])


if __name__ == "__main__":
    unittest.main()
