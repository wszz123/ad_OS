import unittest

from coop.experiment_audit import audit_cost_report, audit_report


class AuditTests(unittest.TestCase):
    def test_cost_audit_keeps_quality_failures_and_supports_pre_review_reports(self):
        row = {"completed": 1, "passed_count": 0, "quality_pass_rate": 0,
               "model_calls": 2, "message_tokens": 4, "total_transfer_bytes": 100,
               "prompt_tokens": 10, "generated_tokens": 5, "accepted_reuse": 0,
               "corpus_searches": 1, "source_parses": 2, "cache_validation_parses": 0,
               "elapsed_sec": 1.0,
               "tasks": [{"task_id": "t1", "status": "ok", "expected_value": "10.00",
                          "answer": {"value": "9.00"}, "quality": {"passed": False}}]}
        report = {"experiment": "cost", "config": {"device": "cuda:1", "embedding_device": "cpu"},
                  "trials": [{"modes": {"text": row, "protocol": {**row, "message_tokens": 2}}}]}
        audit = audit_cost_report(report)
        self.assertEqual(audit["averages"]["protocol"]["communication_token_reduction_vs_text_pct"], 50)
        self.assertEqual(audit["averages"]["protocol"]["plan_reviews"], 0)
        self.assertEqual(len(audit["failures"]), 2)
        self.assertFalse(audit["review_plan"])

    def test_reuse_must_reference_a_completed_earlier_task_with_matching_evidence(self):
        rows = [
            {"task_id": "first", "status": "ok", "evidence_id": "e1", "evidence": "source", "evidence_correct": True},
            {"task_id": "second", "status": "ok", "evidence_id": "e1", "evidence": "source", "evidence_correct": False,
             "accepted_reuse": 1, "reuse_source_task": "first", "reuse_source_agent": "summarizer"},
        ]
        data = {"tasks": rows, "completed": 2, "evidence_accuracy_proxy": 0.5,
                "model_calls": 3, "prompt_tokens": 10, "generated_tokens": 5, "message_tokens": 20,
                "total_transfer_bytes": 100, "elapsed_sec": 1.0, "new_memory_records": 2}
        report = {"experiment": "test", "trials": [{"modes": {"memory": data}}]}
        audit = audit_report(report)
        self.assertTrue(audit["all_reuse_provenance_valid"])
        self.assertEqual(audit["trial_metrics"][0]["memory"]["reuse_evidence_label_accuracy"], 0)
        rows[1]["reuse_source_task"] = "second"
        self.assertFalse(audit_report(report)["all_reuse_provenance_valid"])
        rows[1]["reuse_source_task"] = "first"
        rows[1]["evidence"] = "modified"
        self.assertFalse(audit_report(report)["all_reuse_provenance_valid"])


if __name__ == "__main__":
    unittest.main()
