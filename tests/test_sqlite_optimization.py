from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from coop.cost_graph import build_cost_graph, public_request
from coop.heldout_workload import expanded_documents, heldout_tasks
from coop.memory import MemoryAdapter, MemoryItem
from coop.protocol import ProtocolAdapter
from coop.sqlite_indexed_memory import IndexedSQLiteMemoryAdapter
from coop.workload import cost_tasks
from tools.run_cost_workload import STORAGE_PROFILES, run_cost_mode
from test_cost_workload import CostLLM, FORMULAS
from test_memory_backends import RecordedCostLLM


class SQLiteOptimizationTests(unittest.TestCase):
    def test_upgrade_and_tag_index_changes_follow_reference_writes_and_deletes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "memory.sqlite")
            reference = MemoryAdapter(path)
            special = "entity:quoted' OR 1=1 --"
            first = MemoryItem("b", "retriever", "2026-01-01", "cache", "cache", tags=["x", "y", special], embedding=[1.0, 0.0])
            reference.put(first)
            reference.put(replace(first, memory_id="a", tags=["y"]))
            indexed = IndexedSQLiteMemoryAdapter(path)
            try:
                for tags in (["x"], ["y"], ["x", "y", "x"], [special], ["missing"], []):
                    self.assertEqual(reference.search_vector([1.0, 0.0], tags=tags), indexed.search_vector([1.0, 0.0], tags=tags))
                    self.assertEqual(reference.search("cache", tags=tags), indexed.search("cache", tags=tags))
                reference.put(replace(first, tags=["new"]))
                self.assertEqual(indexed.search_vector([1.0, 0.0], tags=["x"]), [])
                self.assertEqual(indexed.search_vector([1.0, 0.0], tags=["new"])[0].item.memory_id, "b")
                with reference.connection:
                    reference.connection.execute("UPDATE memories SET tags_json=? WHERE memory_id='b'", ('["updated"]',))
                self.assertEqual(indexed.search_vector([1.0, 0.0], tags=["new"]), [])
                self.assertEqual(indexed.search_vector([1.0, 0.0], tags=["updated"])[0].item.memory_id, "b")
                with reference.connection:
                    reference.connection.execute("DELETE FROM memories WHERE memory_id='b'")
                self.assertEqual(indexed.search_vector([1.0, 0.0], tags=["updated"]), [])
                self.assertEqual(indexed.connection.execute("SELECT COUNT(*) FROM coop_memory_tags WHERE memory_id='b'").fetchone()[0], 0)
                self.assertEqual(indexed.count(), reference.count())
            finally:
                reference.close()
                indexed.close()

    def test_metadata_projection_avoids_vector_decoding_and_filtered_scan_fetches_only_matches(self):
        store = IndexedSQLiteMemoryAdapter()
        try:
            for doc in expanded_documents(30):
                doc.embedding = [1.0, 0.0]
                store.put(doc)
            store.search_vector([1.0, 0.0], tags=["entity:Orion"])
            stats = store.scan_stats()
            self.assertEqual(stats["full_scan_rows"], 0)
            self.assertEqual(stats["filtered_scan_rows"], 4)
            with store.connection:
                store.connection.execute("UPDATE memories SET embedding_json='not JSON' WHERE memory_id='orion-price'")
            # Malformed vector proves metadata lookup does not deserialize it.
            self.assertIsNone(store.get_metadata("orion-price").embedding)
            self.assertIsNone(store.get_metadata("missing"))
            with self.assertRaises(ValueError):
                store.get("orion-price")
            self.assertEqual(store.scan_stats()["metadata_get_rows"], 1)
            self.assertEqual(store.scan_stats()["metadata_get_calls"], 2)
        finally:
            store.close()

    def test_profiles_keep_answers_and_reuse_identical_and_reduce_actual_vector_materialization(self):
        docs, tasks = expanded_documents(30), heldout_tasks()
        for doc in docs:
            doc.embedding = [1.0, 0.0] if doc.task_topic == "Orion" else [0.0, 1.0]
        def encoder(texts, query):
            return [[1.0, 0.0] if text.startswith("Orion") else [0.0, 1.0] for text in texts]
        def vector_bytes(result):
            stats = result["corpus_scan_stats"]
            return sum(stats[key] for key in ("vector_json_bytes_loaded", "filtered_vector_json_bytes_loaded", "point_vector_json_bytes_loaded"))
        outputs = {}
        with tempfile.TemporaryDirectory() as directory:
            for profile in STORAGE_PROFILES:
                outputs[profile] = run_cost_mode(RecordedCostLLM(FORMULAS), encoder, docs, tasks,
                    Path(directory) / profile, "protocol_vector_memory", namespace="test",
                    storage_profile=profile, durable=True)
                self.assertEqual(outputs[profile]["passed_count"], 10)
                self.assertEqual(outputs[profile]["accepted_reuse"], 4)
            reference = outputs["reference"]
            for result in outputs.values():
                self.assertEqual([row["answer"] for row in reference["tasks"]], [row["answer"] for row in result["tasks"]])
                self.assertEqual(reference["message_tokens"], result["message_tokens"])
                self.assertEqual(reference["model_calls"], result["model_calls"])
            pushed = outputs["tag_pushdown"]
            projected = outputs["metadata_projection"]
            optimized = outputs["optimized"]
            self.assertEqual(reference["corpus_scan_stats"]["full_scan_rows"], 39 * 6)
            self.assertEqual(pushed["corpus_scan_stats"]["filtered_scan_rows"], 21)
            self.assertLess(vector_bytes(pushed), vector_bytes(reference))
            self.assertLess(vector_bytes(projected), vector_bytes(pushed))
            self.assertLess(optimized["corpus_scan_stats"]["metadata_get_calls"], projected["corpus_scan_stats"]["metadata_get_calls"])

    def test_projected_deduplicated_validation_still_invalidates_modified_sources(self):
        corpus, memory = IndexedSQLiteMemoryAdapter(), MemoryAdapter()
        try:
            for doc in expanded_documents(2):
                doc.embedding = [1.0, 0.0]
                corpus.put(doc)
            graph, _, events = build_cost_graph(CostLLM([FORMULAS[0]] * 2), corpus, memory, ProtocolAdapter(),
                namespace="test", encoder=lambda texts, query: [[1.0, 0.0]], reuse=True,
                metadata_projection=True, deduplicate_validation=True)
            graph.invoke(public_request(cost_tasks()[0]))
            original = corpus.get("orion-price")
            corpus.put(replace(original, summary="updated", evidence='{"entity":"Orion","facts":{"node_price":"2500"}}'))
            task = replace(cost_tasks()[0], task_id="new-source")
            result = graph.invoke(public_request(task))
            self.assertEqual(events["new-source"]["accepted_reuse"], 0)
            self.assertIn("22000.00", result["tool_json"])
        finally:
            corpus.close()
            memory.close()


if __name__ == "__main__":
    unittest.main()
