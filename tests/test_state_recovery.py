from dataclasses import replace
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from coop.state_exchange import StateExchange
from coop.state_store import SQLiteStateStore
from coop.continuous_tasks import ContinuousTask
from coop.memory import MemoryItem
from tools.run_qwen_continuous import run_mode
from tools.run_recovery_experiment import run_recovery


class StateRecoveryTests(unittest.TestCase):
    def packet(self, vector=(0.25, -0.5)):
        return StateExchange().encode(vector, state_id="s1", source_agent="planner", target_agent="retriever")

    def test_reopen_namespace_isolation_and_immutable_idempotent_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "states.sqlite")
            packet = self.packet()
            with SQLiteStateStore(path, namespace="v1") as writer:
                writer["s1"] = packet
                writer["s1"] = packet
                with self.assertRaises(ValueError):
                    writer["s1"] = self.packet((0.5, 0.1))
                self.assertEqual(len(writer), 1)
            with SQLiteStateStore(path, namespace="v1") as reader:
                self.assertEqual(reader["s1"], packet)
            with SQLiteStateStore(path, namespace="v2") as isolated:
                with self.assertRaises(KeyError):
                    isolated["s1"]

    def test_corrupt_payload_and_metadata_are_rejected(self):
        exchange = StateExchange()
        packet = self.packet()
        for altered in (replace(packet, payload=b"xxxx"), replace(packet, shape=(3,)),
                        replace(packet, dtype="float32")):
            with self.assertRaises(ValueError):
                exchange.decode(altered)
        for vector in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                self.packet(vector)
        with tempfile.TemporaryDirectory() as directory:
            with SQLiteStateStore(str(Path(directory) / "states.sqlite"), namespace="v1") as store:
                store["s1"] = packet
                with store.connection:
                    store.connection.execute("UPDATE state_packets SET payload=? WHERE state_id='s1'", (b"xxxx",))
                with self.assertRaises(ValueError):
                    store["s1"]

    def test_sigkill_and_fresh_process_resume_without_replanning(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_recovery(str(Path(directory) / "recovery.json"), fake=True, timeout=90)
            self.assertTrue(report["all_passed"])
            self.assertEqual(report["prepare"]["calls"], ["planner"])
            self.assertEqual(report["resume"]["calls"], ["summarizer"])
            self.assertEqual(report["resume"]["task_metrics"]["corpus_searches"], 1)

    def test_continuous_evaluator_supports_durable_backends_and_counts_packets(self):
        class LLM:
            def __init__(self):
                self.call_stats = []
                self.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())

            def generate_json(self, messages, **kwargs):
                self.call_stats.append({"prompt_tokens": 1, "generated_tokens": 1})
                return {"value": {"topic": "cache", "search_query": "cache"}}

            def generate(self, messages, **kwargs):
                self.call_stats.append({"prompt_tokens": 1, "generated_tokens": 1})
                return {"text": "Use cache evidence."}

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "durable-run"
            report = run_mode(
                LLM(), lambda texts, query: [[1.0, 0.0]],
                [MemoryItem("e1", "corpus", "", "cache", "Cache evidence", embedding=[1.0, 0.0])],
                [ContinuousTask("t1", "cache", "cache", "e1"), ContinuousTask("t2", "cache", "cache again", "e1")],
                target, "protocol_vector_memory", namespace="test", threshold=0.6, margin=0.05,
                max_new_tokens=16, durable=True,
            )
            self.assertEqual(report["completed"], 2)
            self.assertEqual(report["model_calls"], 3)
            self.assertEqual(report["state_transfers"], 2)
            self.assertEqual(report["state_payload_bytes"], 8)
            self.assertEqual(report["state_backend"], "sqlite")
            with SQLiteStateStore(str(target / "states.sqlite"), namespace="test") as states:
                self.assertEqual(len(states), 2)


if __name__ == "__main__":
    unittest.main()
