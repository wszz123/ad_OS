from dataclasses import asdict, replace
from pathlib import Path
import sqlite3
import tempfile
import unittest

from coop.heldout_workload import expanded_documents, heldout_tasks
from coop.memory import MemoryAdapter, MemoryItem
from coop.sqlite_indexed_memory import IndexedSQLiteMemoryAdapter
from tools.run_cost_workload import run_cost_mode
from test_cost_workload import FORMULAS
from test_memory_backends import RecordedCostLLM


class BatchIngestionTests(unittest.TestCase):
    def item(self, memory_id, **kwargs):
        return MemoryItem(memory_id, "source", "2026-01-01", "topic", "summary", tags=["old"], embedding=[1.0, 0.0], **kwargs)

    def test_batch_matches_per_item_with_repeated_ids_and_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [self.item("a"), self.item("b"), replace(self.item("a"), summary="new", tags=["new"])]
            paths = [str(Path(directory) / name) for name in ("individual.sqlite", "batch.sqlite")]
            individual, batch = [IndexedSQLiteMemoryAdapter(path) for path in paths]
            try:
                expected_ids = [individual.put(replace(row)) for row in rows]
                actual_ids = batch.put_many(replace(row) for row in rows)
                self.assertEqual(expected_ids, actual_ids)
                self.assertEqual(batch.count(), 2)
                self.assertEqual([asdict(row) for row in individual.items()], [asdict(row) for row in batch.items()])
                self.assertEqual(individual.write_stats()["commits"], 3)
                self.assertEqual(batch.write_stats()["commits"], 1)
            finally:
                individual.close()
                batch.close()
            reopened = IndexedSQLiteMemoryAdapter(paths[1])
            try:
                self.assertEqual(reopened.get("a").summary, "new")
                self.assertEqual([hit.item.memory_id for hit in reopened.search_vector([1.0, 0.0], tags=["new"])], ["a"])
                self.assertEqual([hit.item.memory_id for hit in reopened.search_vector([1.0, 0.0], tags=["old"])], ["b"])
            finally:
                reopened.close()

    def test_database_failure_rolls_back_earlier_upserts_and_trigger_changes(self):
        store = IndexedSQLiteMemoryAdapter()
        try:
            original = self.item("existing")
            store.put(original)
            bad = replace(self.item("bad"), source_agent=None)
            with self.assertRaises(sqlite3.IntegrityError):
                store.put_many([replace(original, summary="changed", tags=["changed"]), self.item("partial"), bad])
            self.assertEqual(store.count(), 1)
            self.assertEqual(store.get("existing"), original)
            self.assertEqual(store.search_vector([1.0, 0.0], tags=["changed"]), [])
            self.assertIsNone(store.get("partial"))
            self.assertFalse(store.connection.in_transaction)
            self.assertEqual(store.write_stats()["rolled_back_scopes"], 1)
            self.assertEqual(store.write_stats()["commits"], 1)
        finally:
            store.close()

    def test_nested_savepoint_does_not_commit_or_rollback_caller_work(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "nested.sqlite")
            store = IndexedSQLiteMemoryAdapter(path)
            observer = MemoryAdapter(path)
            try:
                store.connection.execute("BEGIN")
                store.put(self.item("caller"))
                self.assertTrue(store.connection.in_transaction)
                self.assertIsNone(observer.get("caller"))
                with self.assertRaises(sqlite3.IntegrityError):
                    store.put_many([self.item("partial"), replace(self.item("bad"), source_agent=None)])
                self.assertIsNotNone(store.get("caller"))
                self.assertIsNone(store.get("partial"))
                self.assertEqual(store.write_stats()["commits"], 0)
                store.connection.rollback()
                self.assertEqual(store.count(), 0)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM coop_memory_tags").fetchone()[0], 0)
            finally:
                observer.close()
                store.close()

    def test_empty_and_preparation_failure_do_not_open_a_write_transaction(self):
        store = MemoryAdapter()
        try:
            self.assertEqual(store.put_many([]), [])
            with self.assertRaises(TypeError):
                store.put_many([self.item("first"), replace(self.item("bad"), embedding=[object()])])
            self.assertEqual(store.count(), 0)
            self.assertEqual(store.write_stats()["transaction_scopes"], 0)
            self.assertFalse(store.connection.in_transaction)
        finally:
            store.close()

    def test_workflow_ingestion_strategy_keeps_results_and_reuse_identical(self):
        docs, tasks = expanded_documents(30), heldout_tasks()
        for doc in docs:
            doc.embedding = [1.0, 0.0] if doc.task_topic == "Orion" else [0.0, 1.0]
        def encoder(texts, query):
            return [[1.0, 0.0] if text.startswith("Orion") else [0.0, 1.0] for text in texts]
        results = {}
        with tempfile.TemporaryDirectory() as directory:
            for strategy in ("per_item", "batch"):
                results[strategy] = run_cost_mode(RecordedCostLLM(FORMULAS), encoder, docs, tasks,
                    Path(directory) / strategy, "protocol_vector_memory", namespace="batch-test",
                    storage_profile="optimized", ingest_strategy=strategy, durable=True)
            a, b = results.values()
            self.assertEqual(a["passed_count"], 10)
            self.assertEqual(b["passed_count"], 10)
            self.assertEqual(a["corpus_write_stats"]["commits"], len(docs))
            self.assertEqual(b["corpus_write_stats"]["commits"], 1)
            self.assertEqual([row["answer"] for row in a["tasks"]], [row["answer"] for row in b["tasks"]])
            self.assertEqual(a["accepted_reuse"], b["accepted_reuse"])
            self.assertEqual(a["message_tokens"], b["message_tokens"])


if __name__ == "__main__":
    unittest.main()
