import unittest

from langgraph.checkpoint.memory import InMemorySaver

from coop.graph import build_memory_graph
from coop.memory import MemoryAdapter, MemoryItem
from coop.protocol import Capability, Message, ProtocolAdapter
from coop.qwen_graph import build_qwen_graph


class GraphTests(unittest.TestCase):
    def test_discovery_and_invalid_actions(self):
        protocol = ProtocolAdapter()
        protocol.register(Capability("planner", "planning", ("plan",)), {"plan": lambda p: p})
        self.assertEqual(protocol.discover("plan")[0].agent_id, "planner")
        with self.assertRaises(ValueError):
            protocol.dispatch(Message("planner", "planner", "unknown", {}))
        with self.assertRaises(ValueError):
            protocol.dispatch(Message("planner", "planner", "plan", {}, version="2.0"))

    def test_ten_graph_invocations_consume_binary_state(self):
        memory = MemoryAdapter()
        memory.put(MemoryItem("m1", "previous-agent", "", "cache", "Avoid repeated computation", embedding=[1.0, 0.0]))
        protocol = ProtocolAdapter()
        packets = {}
        graph = build_memory_graph(memory, lambda texts: [[1.0, 0.0]], protocol,
                                   packets, checkpointer=InMemorySaver())
        for index in range(10):
            config = {"configurable": {"thread_id": f"task-{index}"}}
            result = graph.invoke({"task_id": f"task-{index}", "query": "cache"}, config)
            self.assertEqual(result["memory_id"], "m1")
            self.assertEqual(result["answer"], "Avoid repeated computation")
            self.assertEqual(graph.get_state(config).values["memory_id"], "m1")
        self.assertEqual(len(protocol.frames), 60)
        self.assertEqual(len(packets), 10)
        memory.close()

    def test_qwen_graph_routes_float16_state_to_retriever(self):
        class FakeLLM:
            def generate_json(self, messages, *, max_new_tokens):
                return {"value": {"topic": "cache", "search_query": "cache reuse"},
                        "prompt_tokens": 2, "generated_tokens": 5}

            def generate(self, messages, *, max_new_tokens):
                return {"text": "Grounded answer", "prompt_tokens": 3, "generated_tokens": 2}

        memory = MemoryAdapter()
        memory.put(MemoryItem("m-cache", "retriever", "", "cache", "Cache reuse evidence",
                              embedding=[1.0, 0.0]))
        protocol = ProtocolAdapter()
        packets = {}
        graph = build_qwen_graph(
            FakeLLM(), memory, protocol,
            query_encoder=lambda texts, query: [[1.0, 0.0]],
            state_packets=packets,
        )
        result = graph.invoke({"task_id": "float-state", "query": "cache"})
        self.assertEqual(result["answer"], "Grounded answer")
        self.assertEqual(result["evidence_id"], "m-cache")
        self.assertEqual(packets["float-state:query"].encoding, "float16")
        self.assertEqual(packets["float-state:query"].payload_bytes, 4)
        memory.close()


if __name__ == "__main__":
    unittest.main()
