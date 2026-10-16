from pathlib import Path
import tempfile
import unittest

from coop.repair_ledger import RepairLedger
from coop.plan_repair import repair_plan, PlanRepairError
from test_plan_repair import SequenceLLM, BAD_ENERGY, ENERGY
from coop.cost_graph import PLAN_PROMPT, REVIEW_PROMPT
from coop.workload import cost_tasks, variable_catalog
from tools.run_repair_recovery import run_case


class RepairLedgerTests(unittest.TestCase):
    def kwargs(self):
        task = cost_tasks()[1]
        return dict(prompt="Entity Orion", catalog=variable_catalog("Orion"), inputs=task.inputs,
                    question=task.question, plan_prompt=PLAN_PROMPT, review_prompt=REVIEW_PROMPT,
                    max_new_tokens=16, max_repairs=2, task_id="t", policy_context="source-v1")

    def test_reopened_ledger_consumes_unfinished_calls_and_replays_accepted_plan_without_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "repair.sqlite")
            with RepairLedger(path) as ledger:
                first = SequenceLLM([BAD_ENERGY, RuntimeError("simulated process death")])
                with self.assertRaises(RuntimeError):
                    repair_plan(first, attempts=[], ledger=ledger, **self.kwargs())
                records = ledger.attempts("t")
                self.assertEqual([item["status"] for item in records], ["rejected", "started"])
                self.assertEqual([item["remaining"] for item in records], [2, 1])
            with RepairLedger(path) as ledger:
                second = SequenceLLM([ENERGY])
                value = repair_plan(second, attempts=[], ledger=ledger, **self.kwargs())
                self.assertEqual(value["expression"], ENERGY)
                self.assertEqual(len(second.call_stats), 1)
                self.assertEqual([item["remaining"] for item in ledger.attempts("t")], [2, 1, 0])
            with RepairLedger(path) as ledger:
                replay = SequenceLLM([])
                self.assertEqual(repair_plan(replay, attempts=[], ledger=ledger, **self.kwargs())["expression"], ENERGY)
                self.assertEqual(replay.call_stats, [])
                self.assertEqual(len(ledger.attempts("t")), 3)

    def test_repeated_deaths_cannot_reset_or_overrun_total_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "repair.sqlite")
            for number in range(3):
                with RepairLedger(path) as ledger:
                    model = SequenceLLM([RuntimeError("interrupted")])
                    with self.assertRaises(RuntimeError):
                        repair_plan(model, attempts=[], ledger=ledger, **self.kwargs())
                    self.assertEqual(len(ledger.attempts("t")), number + 1)
                    self.assertEqual(ledger.attempts("t")[-1]["remaining"], 2 - number)
            with RepairLedger(path) as ledger:
                model = SequenceLLM([ENERGY])
                with self.assertRaises(PlanRepairError):
                    repair_plan(model, attempts=[], ledger=ledger, **self.kwargs())
                self.assertEqual(model.call_stats, [])
                self.assertEqual(len(ledger.attempts("t")), 3)

    def test_changed_inputs_policy_generation_budget_and_checker_context_are_rejected(self):
        with RepairLedger() as ledger:
            repair_plan(SequenceLLM([BAD_ENERGY, ENERGY]), attempts=[], ledger=ledger, **self.kwargs())
            for change in ({"inputs": {"node_count": "9", "days": "30"}},
                           {"question": "changed"}, {"max_new_tokens": 32},
                           {"max_repairs": 3}, {"policy_context": "source-v2"}):
                model = SequenceLLM([])
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    repair_plan(model, attempts=[], ledger=ledger, **{**self.kwargs(), **change})
                self.assertEqual(model.call_stats, [])

    def test_reservation_numbers_are_atomic_across_two_connections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "repair.sqlite")
            with RepairLedger(path) as first, RepairLedger(path) as second:
                first.begin("t", max_repairs=2, prompt_fingerprint="fp")
                records = [first.reserve("t", max_repairs=2, prompt_fingerprint="fp", payload={}),
                           second.reserve("t", max_repairs=2, prompt_fingerprint="fp", payload={}),
                           first.reserve("t", max_repairs=2, prompt_fingerprint="fp", payload={})]
                self.assertEqual([record["attempt"] for record in records], [0, 1, 2])
                with self.assertRaises(ValueError):
                    second.reserve("t", max_repairs=2, prompt_fingerprint="fp", payload={})
                forged = {**records[0], "reservation_token": "not-owner"}
                with self.assertRaises(ValueError):
                    first.finish("t", forged, payload={}, accepted=False)

    def test_sigkill_planner_replay_preserves_inflight_accepted_and_exhausted_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            for case in ("inflight", "accepted", "exhausted"):
                report = run_case(Path(directory) / case, case=case, fake=True,
                                  device="cpu", max_tokens=32, timeout=90)
                self.assertTrue(report["all_passed"], report["checks"])
                self.assertEqual(report["resume"]["checkpoint_before_next"], ["planner"])
                if case != "inflight":
                    self.assertEqual(report["resume"]["calls"], 0)


if __name__ == "__main__":
    unittest.main()
