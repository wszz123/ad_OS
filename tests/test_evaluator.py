import unittest

from coop.evaluator import MODES, run_evaluation


class EvaluatorTests(unittest.TestCase):
    def test_four_paired_modes_run_same_ten_tasks(self):
        report = run_evaluation(modes=MODES)
        self.assertEqual(set(report["modes"]), set(MODES))
        for metrics in report["modes"].values():
            self.assertEqual(metrics["task_count"], 10)
            self.assertEqual(len(metrics["tasks"]), 10)
        self.assertEqual(report["modes"]["text"]["message_count"], 60)
        self.assertEqual(report["modes"]["protocol"]["message_count"], 60)
        self.assertEqual(report["modes"]["protocol_vector"]["state_transfers"], 0)
        self.assertEqual(report["modes"]["protocol_vector_memory"]["memory_queries"], 10)

    def test_repeat_produces_twenty_continuous_tasks(self):
        report = run_evaluation(modes=("protocol_vector_memory",), repeat=2)
        metrics = report["modes"]["protocol_vector_memory"]
        self.assertEqual(metrics["task_count"], 20)
        self.assertEqual(metrics["memory_queries"], 20)


if __name__ == "__main__":
    unittest.main()
