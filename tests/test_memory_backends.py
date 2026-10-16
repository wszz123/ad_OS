from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

from langgraph.store.memory import InMemoryStore

from coop.memory import MemoryAdapter, MemoryItem
from coop.memory_backends import LangGraphMemoryAdapter, MeasuredMemoryBackend, create_memory_backend
from coop.workload import cost_documents, cost_tasks
from tools.run_cost_workload import run_cost_mode
from tools.run_memory_backend_comparison import compare_backend_results
from test_cost_workload import CostLLM, FORMULAS


class RecordedCostLLM(CostLLM):
    def __init__(self, expressions):
        super().__init__(expressions)
        self.call_stats = []
        self.tokenizer = type("Tokenizer", (), {"encode": lambda self, text, **kwargs: text.split()})()

    def generate_json(self, messages, **kwargs):
        self.call_stats.append({"prompt_tokens": 10, "generated_tokens": 5})
        return super().generate_json(messages, **kwargs)


class MemoryBackendTests(unittest.TestCase):
    def test_schema_search_scores_tags_ties_and_upsert_match_sqlite(self):
        sqlite, external = MemoryAdapter(), LangGraphMemoryAdapter(page_size=2)
        try:
            # Reverse insertion order exposes native insertion-order ranking bugs.
            items = [MemoryItem("z", "planner", "2026-01-01", "cache", "cache facts", tags=["gpu"], embedding=[1.0, 0.0]),
                     MemoryItem("a", "retriever", "2026-01-01", "cache", "cache facts", tags=["gpu"], embedding=[1.0, 0.0]),
                     MemoryItem("b", "summarizer", "2026-01-01", "memory", "saved evidence", tags=["memory"], embedding=[0.0, 1.0]),
                     MemoryItem("c", "executor", "2026-01-01", "other", "other", tags=[])]
            for item in items:
                sqlite.put(replace(item))
                external.put(replace(item))
            self.assertEqual([asdict(item) for item in sqlite.items()], [asdict(item) for item in external.items()])
            for query in ("cache", "missing"):
                self.assertEqual(sqlite.search(query, limit=10), external.search(query, limit=10))
            self.assertEqual(sqlite.search_vector([1.0, 0.0], tags=["gpu"]), external.search_vector([1.0, 0.0], tags=["gpu"]))
            self.assertEqual([hit.item.memory_id for hit in external.search_vector([1.0, 0.0], tags=["gpu"])], ["a", "z"])
            for backend in (sqlite, external):
                backend.put(replace(items[0], summary="updated"))
                self.assertEqual(backend.count(), 4)
                self.assertEqual(backend.get("z").summary, "updated")
                self.assertIsNone(backend.get("missing"))
                with self.assertRaises(ValueError):
                    backend.search_vector([1.0], limit=1)
                with self.assertRaises(ValueError):
                    backend.search("cache", limit=0)
            detached = external.get("a")
            detached.tags.append("tampered")
            self.assertNotIn("tampered", external.get("a").tags)
        finally:
            sqlite.close()
            external.close()

    def test_paginated_store_uses_exact_namespace_and_preserves_shared_store(self):
        store = InMemoryStore()
        first = LangGraphMemoryAdapter(store, namespace=("team",), page_size=1)
        second = LangGraphMemoryAdapter(store, namespace=("other",))
        child = LangGraphMemoryAdapter(store, namespace=("team", "child"))
        for backend, memory_id in ((first, "root"), (second, "other"), (child, "child")):
            backend.put(MemoryItem(memory_id, "planner", "", "topic", "summary"))
        self.assertEqual([item.memory_id for item in first.items()], ["root"])
        self.assertIsNone(first.get("child"))
        first.close()
        reopened = LangGraphMemoryAdapter(store, namespace=("team",))
        self.assertEqual(reopened.count(), 1)
        self.assertEqual(second.count(), 1)
        self.assertEqual(child.count(), 1)
        # New InMemoryStore has no history; rewrapping the same store is not restart persistence.
        self.assertEqual(LangGraphMemoryAdapter(namespace=("team",)).count(), 0)

    def test_query_embedding_callbacks_match_and_direct_vector_needs_no_encoding(self):
        calls = []
        def documents(texts):
            calls.append("document")
            return [[1.0, 0.0] for _ in texts]
        def queries(texts):
            calls.append("query")
            return [[1.0, 0.0] for _ in texts]
        for backend in (MemoryAdapter(embedder=documents, query_embedder=queries),
                        LangGraphMemoryAdapter(embedder=documents, query_embedder=queries)):
            backend.put(MemoryItem("m", "retriever", "", "cache", "cache"))
            self.assertEqual(backend.search("paraphrase")[0].match_type, "semantic")
            before = len(calls)
            backend.search_vector([1.0, 0.0])
            self.assertEqual(len(calls), before)
            backend.close()
        self.assertEqual(calls, ["document", "query", "document", "query"])

    def test_profiler_records_errors_and_factory_rejects_unknown_backend(self):
        backend = MeasuredMemoryBackend(MemoryAdapter())
        try:
            backend.put(MemoryItem("m", "executor", "", "cache", "summary", embedding=[1.0, 0.0]))
            backend.get("m")
            with self.assertRaises(ValueError):
                backend.search_vector([1.0])
            stats = backend.stats()
            self.assertEqual(stats["put"]["calls"], 1)
            self.assertEqual(stats["search_vector"]["errors"], 1)
            self.assertGreaterEqual(stats["get"]["elapsed_sec"], 0)
            with self.assertRaises(ValueError):
                create_memory_backend("unknown")
        finally:
            backend.close()

    def test_ten_task_backend_substitution_keeps_results_and_reuse_identical(self):
        docs = cost_documents()
        for doc in docs:
            doc.embedding = [1.0, 0.0] if doc.task_topic == "Orion" else [0.0, 1.0]
        def encoder(texts, query):
            return [[1.0, 0.0] if text.startswith("Orion") else [0.0, 1.0] for text in texts]
        results = {}
        with tempfile.TemporaryDirectory() as directory:
            for backend in ("sqlite", "langgraph"):
                results[backend] = run_cost_mode(
                    RecordedCostLLM(FORMULAS), encoder, docs, cost_tasks(),
                    Path(directory) / backend, "protocol_vector_memory", namespace="test",
                    memory_backend=backend, durable=True,
                )
                self.assertEqual(results[backend]["passed_count"], 10)
                self.assertEqual(results[backend]["memory_backend_operations"]["put"]["calls"], 10)
                self.assertEqual(results[backend]["memory_backend_operations"]["search_vector"]["calls"], 10)
                self.assertTrue((Path(directory) / backend / "memory_snapshot.json").is_file())
            self.assertTrue(all(compare_backend_results(results).values()))
            self.assertFalse((Path(directory) / "langgraph" / "memory.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
