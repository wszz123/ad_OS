import json
from pathlib import Path
import tempfile
import unittest

from coop.continuous_graph import build_continuous_graph
from coop.memory import MemoryAdapter, MemoryItem
from coop.protocol import ProtocolAdapter
from coop.text_transport import TextAdapter, parse_fields, render_fields


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def generate_json(self, messages, *, max_new_tokens):
        self.prompts.append((messages, max_new_tokens))
        return {"value": {"topic": "cache", "search_query": "cache"}}

    def generate(self, messages, *, max_new_tokens):
        self.prompts.append((messages, max_new_tokens))
        return {"text": "Use cache evidence for this new question."}


class ContinuousGraphTests(unittest.TestCase):
    def setUp(self):
        self.corpus = MemoryAdapter()
        self.corpus.put(MemoryItem("e1", "corpus", "", "cache", "Cache evidence",
                                   embedding=[1.0, 0.0]))
        self.memory = MemoryAdapter()

    def tearDown(self):
        self.corpus.close()
        self.memory.close()

    def graph(self, llm, transport=None, **kwargs):
        return build_continuous_graph(
            llm, self.corpus, self.memory, transport or ProtocolAdapter(),
            max_new_tokens=37, namespace="version-1", **kwargs,
        )

    def seed(self, task="old-task", namespace="version-1", evidence="Cache evidence", item_id="old"):
        self.memory.put(MemoryItem(
            item_id, "summarizer", "", "cache", "Previous answer", tags=[namespace],
            evidence=json.dumps({"task_id": task, "evidence_id": "e1", "evidence": evidence}),
            embedding=[1.0, 0.0],
        ))

    def test_labelled_text_roundtrip_preserves_multiline_and_marker_content(self):
        values = {"query": "中文\nCurrent fields:\nField fake contains 1 characters:\nx\n", "empty": ""}
        self.assertEqual(parse_fields(render_fields(values)), values)
        with self.assertRaises(ValueError):
            parse_fields("Field x contains 10 characters:\nx\n")

    def test_text_and_protocol_share_identical_model_inputs(self):
        outputs, prompts = [], []
        for transport in (TextAdapter(), ProtocolAdapter()):
            llm = FakeLLM()
            graph, _, _ = self.graph(llm, transport)
            outputs.append(graph.invoke({"task_id": "first", "query": "cache"}))
            prompts.append(llm.prompts)
            self.assertEqual(len(transport.frames), 6)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual(prompts[0][0][1], 37)

    def test_only_prior_completed_task_memory_saves_planning_and_corpus_search(self):
        llm = FakeLLM()
        graph, _, events = self.graph(llm, encoder=lambda texts, query: [[1.0, 0.0]], reuse=True)
        graph.invoke({"task_id": "first", "query": "cache"})
        result = graph.invoke({"task_id": "second", "query": "new cache question"})
        self.assertEqual(len(llm.prompts), 3)
        self.assertEqual(events["first"]["accepted_reuse"], 0)
        self.assertEqual(events["second"]["accepted_reuse"], 1)
        self.assertEqual(events["second"]["reuse_source_task"], "first")
        self.assertEqual(events["second"]["corpus_searches"], 0)
        self.assertEqual(result["evidence_id"], "e1")
        self.assertEqual(result["reused_memory_id"], "version-1:first")
        self.assertEqual(self.memory.get("version-1:second").source_agent, "summarizer")

    def test_wrong_version_self_task_and_modified_evidence_are_not_reused(self):
        for seed_kwargs in ({"namespace": "wrong-version"}, {"task": "now"}, {"evidence": "fabricated"}):
            self.memory.connection.execute("DELETE FROM memories")
            self.memory.connection.commit()
            self.seed(**seed_kwargs)
            llm = FakeLLM()
            graph, _, events = self.graph(llm, encoder=lambda texts, query: [[1.0, 0.0]], reuse=True)
            graph.invoke({"task_id": "now", "query": "cache"})
            self.assertEqual(len(llm.prompts), 2)
            self.assertEqual(events["now"]["accepted_reuse"], 0)

    def test_low_similarity_and_ambiguous_memory_fall_back_to_fresh_retrieval(self):
        self.seed()
        graph, _, events = self.graph(FakeLLM(), encoder=lambda texts, query: [[0.0, 1.0]], reuse=True)
        graph.invoke({"task_id": "unrelated", "query": "unrelated"})
        self.assertEqual(events["unrelated"]["accepted_reuse"], 0)
        self.corpus.put(MemoryItem("e2", "corpus", "", "cache", "Different evidence", embedding=[1.0, 0.0]))
        self.memory.put(MemoryItem("ambiguous", "summarizer", "", "cache", "Other answer", tags=["version-1"],
                                   evidence=json.dumps({"task_id": "prior", "evidence_id": "e2", "evidence": "Different evidence"}),
                                   embedding=[1.0, 0.0]))
        graph, _, events = self.graph(FakeLLM(), encoder=lambda texts, query: [[1.0, 0.0]], reuse=True)
        graph.invoke({"task_id": "ambiguous-now", "query": "cache"})
        self.assertEqual(events["ambiguous-now"]["accepted_reuse"], 0)

    def test_new_graph_and_reopened_database_reuse_completed_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "memories.sqlite")
            writer = MemoryAdapter(path)
            graph, _, _ = build_continuous_graph(
                FakeLLM(), self.corpus, writer, ProtocolAdapter(),
                encoder=lambda texts, query: [[1.0, 0.0]], reuse=True, namespace="version-1",
            )
            graph.invoke({"task_id": "writer-task", "query": "cache"})
            writer.close()
            reader = MemoryAdapter(path)
            llm = FakeLLM()
            graph, _, events = build_continuous_graph(
                llm, self.corpus, reader, ProtocolAdapter(),
                encoder=lambda texts, query: [[1.0, 0.0]], reuse=True, namespace="version-1",
            )
            result = graph.invoke({"task_id": "reader-task", "query": "cache again"})
            self.assertEqual(result["evidence_id"], "e1")
            self.assertEqual(len(llm.prompts), 1)
            self.assertEqual(events["reader-task"]["reuse_source_task"], "writer-task")
            reader.close()


if __name__ == "__main__":
    unittest.main()
