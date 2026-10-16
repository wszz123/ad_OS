from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from coop.calculator import calculate
from coop.cost_graph import document_snapshot, public_request, read_facts
from coop.heldout_workload import expanded_documents, heldout_tasks, reference_amount
from coop.memory import MemoryAdapter
from coop.workload import cost_documents, cost_tasks
from tools.run_cost_workload import populate_embeddings, run_cost_mode
from tools.run_heldout_workload import run_cell_trials, save_progress, summarize_cell
from test_cost_workload import FORMULAS
from test_memory_backends import RecordedCostLLM


class HeldoutTests(unittest.TestCase):
    def test_mode_level_resume_keeps_completed_work_and_retries_only_missing_mode(self):
        cell = {"trials": []}
        calls = []
        saves = []
        data = {key: 0 for key in ("task_count", "completed", "passed_count", "model_calls",
                "accepted_reuse", "corpus_searches", "source_parses", "plan_repairs", "prompt_tokens",
                "generated_tokens", "elapsed_sec", "corpus_search_sec", "memory_lookup_sec", "embedding_sec")}
        data.update(task_count=1, completed=1, passed_count=1,
                    corpus_scan_stats={"full_scan_rows": 1, "vector_json_bytes_loaded": 1})
        def interrupted(trial, mode):
            calls.append(mode)
            if mode == "protocol_vector_memory":
                raise RuntimeError("simulated timeout")
            return dict(data), "completed-artifact"
        with self.assertRaises(RuntimeError):
            run_cell_trials(cell, repeat=1, run_mode=interrupted, save=lambda: saves.append(True))
        self.assertIn("protocol_vector", cell["pending_trial"]["modes"])
        self.assertTrue(saves)
        def resumed(trial, mode):
            calls.append(mode)
            return dict(data), "retry-artifact"
        run_cell_trials(cell, repeat=1, run_mode=resumed, save=lambda: None)
        self.assertEqual(calls, ["protocol_vector", "protocol_vector_memory", "protocol_vector_memory"])
        self.assertEqual(cell["trials"][0]["artifacts"]["protocol_vector"], "completed-artifact")
        self.assertNotIn("pending_trial", cell)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            save_progress(target, {"status": "first"})
            save_progress(target, {"status": "complete"})
            import json
            self.assertEqual(json.loads(target.read_text())["status"], "complete")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_summary_counts_point_read_overhead_instead_of_only_reporting_scan_savings(self):
        values = {key: 0 for key in ("task_count", "completed", "passed_count", "model_calls", "accepted_reuse",
                  "corpus_searches", "source_parses", "plan_repairs", "prompt_tokens", "generated_tokens",
                  "elapsed_sec", "corpus_search_sec", "memory_lookup_sec", "embedding_sec")}
        values.update(task_count=1, passed_count=1)
        baseline = {**values, "corpus_scan_stats": {"full_scan_rows": 10, "vector_json_bytes_loaded": 100,
                                                   "point_vector_json_bytes_loaded": 0}}
        memory = {**values, "corpus_scan_stats": {"full_scan_rows": 6, "vector_json_bytes_loaded": 60,
                                                 "point_vector_json_bytes_loaded": 60}}
        summary = summarize_cell([{"modes": {"protocol_vector": baseline, "protocol_vector_memory": memory}}])
        self.assertEqual(summary["saved_full_scan_vector_json_bytes"], 40)
        self.assertEqual(summary["saved_materialized_vector_json_bytes"], -20)

    def test_seed_reproducibility_and_no_original_input_combinations(self):
        first, second = heldout_tasks(rounds=2), heldout_tasks(rounds=2)
        self.assertEqual(first, second)
        self.assertNotEqual(first, heldout_tasks(seed=42, rounds=2))
        self.assertEqual(len(first), 20)
        self.assertEqual(len({task.task_id for task in first}), 20)
        originals = cost_tasks()
        for index, task in enumerate(first):
            self.assertNotEqual(task.inputs, originals[index % 10].inputs)
            self.assertNotIn("expected_value", public_request(task))
            if "retention_days" in task.inputs and "days" in task.inputs:
                self.assertNotEqual(task.inputs["retention_days"], task.inputs["days"])

    def test_independent_oracle_matches_original_manual_labels_and_ast_calculation(self):
        for index, task in enumerate(cost_tasks()):
            self.assertEqual(reference_amount(index, task.inputs), task.expected_value)
        docs = cost_documents()
        for index, task in enumerate(heldout_tasks(rounds=3)):
            facts = read_facts([document_snapshot(doc) for doc in docs if doc.task_topic == task.entity], task.entity)
            value = calculate(FORMULAS[index % 10], task.inputs, facts)
            self.assertEqual(value["value"], task.expected_value)

    def test_expansion_preserves_original_sources_and_batches_all_documents(self):
        docs = expanded_documents(17)
        self.assertEqual(len(docs), 26)
        self.assertEqual([asdict(doc) for doc in docs[:9]], [asdict(doc) for doc in cost_documents()])
        self.assertEqual(len({doc.memory_id for doc in docs}), 26)
        self.assertTrue(all("entity:Orion" not in doc.tags and "entity:Lyra" not in doc.tags for doc in docs[9:]))
        batches = []
        def encode(texts):
            batches.append(list(texts))
            return [[1.0, 0.0] for _ in texts]
        populate_embeddings(docs, encode, batch_size=4)
        self.assertEqual(sum(len(batch) for batch in batches), len(docs))
        self.assertTrue(all(len(batch) <= 4 for batch in batches))
        self.assertTrue(all(doc.embedding == [1.0, 0.0] for doc in docs))
        with self.assertRaises(ValueError):
            populate_embeddings(docs, lambda texts: [], batch_size=4)

    def test_scan_counters_separate_full_materialization_and_point_reads(self):
        store = MemoryAdapter()
        try:
            for doc in expanded_documents(5):
                doc.embedding = [1.0, 0.0]
                store.put(doc)
            store.search_vector([1.0, 0.0], tags=["entity:Orion"])
            stats = store.scan_stats()
            self.assertEqual(stats["full_scan_rows"], 14)
            self.assertEqual(stats["full_scan_calls"], 1)
            self.assertGreater(stats["vector_json_bytes_loaded"], 0)
            store.get("orion-price")
            store.get("missing")
            stats = store.scan_stats()
            self.assertEqual(stats["point_get_calls"], 2)
            self.assertEqual(stats["point_get_rows"], 1)
            self.assertEqual(stats["full_scan_rows"], 14)
        finally:
            store.close()

    def test_memory_reuse_reduces_scanned_rows_on_unseen_inputs_without_changing_answers(self):
        tasks, docs = heldout_tasks(), expanded_documents(17)
        for doc in docs:
            doc.embedding = [1.0, 0.0] if doc.task_topic == "Orion" else [0.0, 1.0]
        def encoder(texts, query):
            return [[1.0, 0.0] if text.startswith("Orion") else [0.0, 1.0] for text in texts]
        with tempfile.TemporaryDirectory() as directory:
            results = {}
            for mode in ("protocol_vector", "protocol_vector_memory"):
                results[mode] = run_cost_mode(RecordedCostLLM(FORMULAS), encoder, docs, tasks,
                                             Path(directory) / mode, mode, namespace="heldout-test")
                self.assertEqual(results[mode]["passed_count"], 10)
            baseline, memory = results.values()
            self.assertEqual(baseline["corpus_scan_stats"]["full_scan_rows"], 260)
            self.assertEqual(memory["corpus_scan_stats"]["full_scan_rows"], 156)
            self.assertEqual([row["answer"] for row in baseline["tasks"]], [row["answer"] for row in memory["tasks"]])
            self.assertEqual(sum(row["corpus_scan_stats"]["full_scan_rows"] for row in memory["tasks"]), 156)
            self.assertEqual(memory["accepted_reuse"], 4)


if __name__ == "__main__":
    unittest.main()
