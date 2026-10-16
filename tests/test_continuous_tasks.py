import unittest

from coop.continuous_tasks import continuous_tasks, evidence_items


class ContinuousTaskTests(unittest.TestCase):
    def test_two_groups_have_five_tasks_each_and_stable_ids(self):
        tasks = continuous_tasks()
        self.assertEqual(len(tasks), 10)
        self.assertEqual([task.task_id for task in tasks], [f"retrieval-{i:02d}" for i in range(1, 6)] + [f"memory-{i:02d}" for i in range(1, 6)])
        self.assertEqual({task.group for task in tasks}, {"retrieval", "memory"})
        self.assertEqual(len(evidence_items()), 2)


if __name__ == "__main__":
    unittest.main()
