from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from coop.medical_graph import MODES, FixtureMedicalDecision, build_medical_graph, fixture_encode
from coop.medical_memory import MedicalMemory
from coop.medical_memory import exact_quotes
from coop.memory import MemoryAdapter
from coop.protocol import ProtocolAdapter
from tools.run_medical_workload import run_mode
from tools.verify_medical_report import verify_report


def example_rows(gold="D"):
    rows = []
    for sequence, stem in enumerate(("The hemoglobin is low in this patient. What explains these findings?",
                                     "Another patient has low hemoglobin and a different history. What explains the findings?"), 1):
        rows.append({"sequence": sequence, "public": {"question_id": "q" + str(sequence), "question": stem,
                     "options": dict(zip("ABCD", ("one", "two", "three", "four")))},
                     "gold": gold, "topic": "LEAK_EVALUATOR_TOPIC", "source_type": "authored",
                     "evaluation_only": {"rationale": "LEAK_PRIVATE_RATIONALE"}})
    return rows


class MedicalNativeTests(unittest.TestCase):
    def test_five_modes_preserve_semantics_and_online_memory_is_gold_blind(self):
        reports = {}
        with tempfile.TemporaryDirectory() as root:
            for mode in MODES:
                reports[mode] = run_mode(example_rows(), Path(root) / mode, mode=mode)
                report = reports[mode]
                self.assertEqual(report["status"], "complete", report)
                self.assertEqual([row["prediction"] for row in report["results"]], ["A", "A"])
                self.assertEqual(report["summary"]["correct"], 0)
                self.assertEqual(report["initial_memory_count"], 0)
                self.assertEqual(report["summary"]["message_count"], 16)
                if "vector" in mode:
                    self.assertEqual(report["summary"]["state_transfers"], 2)
                    self.assertGreater(report["summary"]["state_payload_bytes"], 0)
                    self.assertTrue((Path(root) / mode / "01-query.bin").is_file())
                if "memory" in mode:
                    self.assertEqual(report["final_memory_count"], 2)
                    self.assertEqual(report["results"][0]["declared_reuse"], 0)
                    self.assertEqual(report["results"][1]["declared_reuse"], 1)
                else:
                    self.assertEqual(report["final_memory_count"], 0)
            self.assertGreater(reports["text"]["summary"]["message_bytes"], reports["protocol"]["summary"]["message_bytes"])
            other = run_mode(example_rows("A"), Path(root) / "different-gold", mode="protocol_vector_memory")
            self.assertEqual(other["summary"]["correct"], 2)
            self.assertEqual(other["final_memory_count"], reports["protocol_vector_memory"]["final_memory_count"])

    def test_model_contexts_do_not_include_labels_topics_or_presentation_annotations(self):
        class Spy(FixtureMedicalDecision):
            def __init__(self):
                super().__init__()
                self.seen = []

            def decide(self, role, context):
                self.seen.append(deepcopy(context))
                return super().decide(role, context)

        with tempfile.TemporaryDirectory() as root:
            decision = Spy()
            run_mode(example_rows(), root, mode="protocol_vector_memory", decision=decision)
            text = json.dumps(decision.seen)
            self.assertNotIn("LEAK_", text)
            self.assertNotIn('"gold"', text)
            self.assertNotIn("evaluation_only", text)
            for context in decision.seen:
                self.assertEqual(set(context["case"]), {"question", "options"})

    def test_notes_remain_unverified_and_invalid_quotes_are_not_admitted(self):
        backend = MemoryAdapter()
        try:
            memory = MedicalMemory(backend, namespace="test", model_id="fixture", encoder=fixture_encode)
            public = example_rows()[0]["public"]
            plan = {"case_quotes": ["Fabricated laboratory fact"], "topic": "anemia", "search_query": "anemia"}
            final = {"memory_note": {"claim": "A model claim", "applies_when": "Recheck findings", "exceptions": "Different conditions"}}
            self.assertFalse(memory.publish(public, 1, plan, final)["written"])
            self.assertEqual(backend.count(), 0)
            plan["case_quotes"] = [public["question"][:30]]
            self.assertTrue(memory.publish(public, 1, plan, final)["written"])
            item = backend.items()[0]
            self.assertEqual(item.memory_type, "medical_unverified_model_note")
            payload = json.loads(item.evidence)["payload"]
            self.assertEqual(payload["clinical_status"], "unverified_model_note")
            self.assertNotIn("answer", payload)
            self.assertNotIn("gold", payload)
            self.assertEqual(payload["source_quotes"][0]["text"], public["question"][:30].strip())
        finally:
            backend.close()

    def test_self_future_and_corrupted_memories_are_rejected(self):
        backend = MemoryAdapter()
        try:
            memory = MedicalMemory(backend, namespace="test", model_id="fixture", encoder=fixture_encode)
            public = example_rows()[0]["public"]
            plan = {"case_quotes": [public["question"][:30]], "topic": "anemia", "search_query": "anemia"}
            final = {"memory_note": {"claim": "A model claim", "applies_when": "Recheck findings", "exceptions": "Different conditions"}}
            memory.publish(public, 1, plan, final)
            vector = fixture_encode(["anemia"])[0]
            self.assertEqual(memory.select(public, 1, "anemia", vector=vector)["candidates"], [])
            next_public = example_rows()[1]["public"]
            self.assertEqual(len(memory.select(next_public, 2, "anemia", vector=vector)["candidates"]), 1)
            item = backend.items()[0]
            record = json.loads(item.evidence)
            record["payload"]["note"]["claim"] = "Corrupted note"
            item.evidence = json.dumps(record)
            backend.put(item)
            selected = memory.select(next_public, 2, "anemia", vector=vector)
            self.assertEqual(selected["candidates"], [])
            self.assertEqual(selected["rejected_ids"], [item.memory_id])
        finally:
            backend.close()

    def test_unknown_memory_citation_fails_schema_and_does_not_write_note(self):
        class InventedCitation(FixtureMedicalDecision):
            def decide(self, role, context):
                value = super().decide(role, context)
                if role == "verifier":
                    value["used_memory_ids"] = ["invented-source"]
                return value

        with tempfile.TemporaryDirectory() as root:
            report = run_mode(example_rows()[:1], root, mode="protocol_vector_memory", decision=InventedCitation())
            self.assertEqual(report["status"], "complete_with_errors")
            self.assertEqual(report["summary"]["errors"], 1)
            self.assertEqual(report["summary"]["schema_retries"], 1)
            self.assertEqual(report["final_memory_count"], 0)

    def test_extra_evaluator_fields_in_public_input_are_rejected_before_model(self):
        backend = MemoryAdapter()
        corpus = None
        try:
            memory = MedicalMemory(backend, namespace="test", model_id="fixture")
            graph, _, _, corpus = build_medical_graph(FixtureMedicalDecision(), memory, ProtocolAdapter())
            public = deepcopy(example_rows()[0]["public"])
            public["evaluation_only"] = {"answer": "D"}
            with self.assertRaisesRegex(ValueError, "only question_id"):
                graph.invoke({"public": public, "sequence": 1})
        finally:
            backend.close()
            if corpus:
                corpus.close()

    def test_independent_report_verifier_rechecks_gold_and_memory_provenance(self):
        from tools.run_medical_workload import write_json
        with tempfile.TemporaryDirectory() as root:
            rows = example_rows()
            run = Path(root) / "run"
            run.mkdir()
            public = "\n".join(json.dumps(row["public"], ensure_ascii=False) for row in rows).encode()
            (run / "questions.public.jsonl").write_bytes(public + b"\n")
            from tools.moreagent_data import sha256
            write_json(run / "labels.evaluation-only.json", [{"question_id": row["public"]["question_id"],
                       "gold": row["gold"], "topic": row["topic"], "source_type": row["source_type"]} for row in rows])
            report = {"implementation": "child repository only; no parent methods/models imports",
                      "dataset": {"source_sha256": "source"}, "config": {"dataset": "ignored"},
                      "selected_questions": 2, "public_input_sha256": sha256(public), "status": "complete", "modes": {}}
            (run / "presentation.json").write_text(json.dumps({"questions": []}))
            # Supply a real dataset path and digest through the manifest values.
            source = Path(root) / "source.json"
            presentation = {"overview": {"total_questions": 2}, "questions": []}
            for row in rows:
                presentation["questions"].append({"sequence": row["sequence"], **row["public"],
                    "topic": row["topic"], "source_type": row["source_type"],
                    "evaluation_only": {"answer": row["gold"], "answer_text": row["public"]["options"][row["gold"]]}})
            source.write_text(json.dumps(presentation))
            report["config"]["dataset"] = str(source)
            from tools.moreagent_data import load_questions
            _, manifest = load_questions(source)
            report["dataset"] = manifest
            for mode in ("text", "protocol", "protocol_vector", "protocol_vector_group_memory", "protocol_vector_memory"):
                results = []
                for row in rows:
                    public_row = row["public"]
                    prediction = "A"
                    final = {"answer": prediction, "rationale": "Short reason.",
                             "memory_note": {"claim": "An unverified note.", "applies_when": "Conditions match.",
                                             "exceptions": "Patient details may differ."},
                             "used_memory_ids": [], "condition_checks": []}
                    results.append({"sequence": row["sequence"], "question_id": public_row["question_id"],
                        "gold": row["gold"], "topic": row["topic"], "source_type": row["source_type"],
                        "status": "ok", "prediction": prediction, "raw_prediction": json.dumps(final),
                        "correct": prediction == row["gold"], "final": final, "memory_writes": 1,
                        "admission": {"written": True, "memory_id": "case-note:" + public_row["question_id"],
                                      "clinical_status": "unverified_model_note", "exact_quotes": 1},
                        "retrieval": {"memories": []}})
                report["modes"][mode] = {"results": results, "status": "complete"}
            report_path = run / "report.json"
            report_path.write_text(json.dumps(report))
            verified = verify_report(report_path)
            self.assertEqual(verified["methods"]["protocol_vector_memory"]["correct"], 0)
            self.assertEqual(len(verified["paired"]), 10)
            report["modes"]["protocol"]["results"][0]["correct"] = True
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "score mismatch"):
                verify_report(report_path)

    def test_independent_report_verifier_accepts_online_note_and_rejects_future_note(self):
        from tools.moreagent_data import load_questions, sha256
        from tools.run_medical_workload import write_json

        with tempfile.TemporaryDirectory() as root:
            rows = example_rows("A")
            dataset = Path(root) / "dataset.json"
            presentation = {"overview": {"total_questions": 2}, "questions": []}
            for row in rows:
                presentation["questions"].append({"sequence": row["sequence"], **row["public"],
                    "topic": row["topic"], "source_type": row["source_type"],
                    "evaluation_only": {"answer": row["gold"], "answer_text": row["public"]["options"][row["gold"]]}})
            dataset.write_text(json.dumps(presentation))
            loaded, manifest = load_questions(dataset)
            directory = Path(root) / "run"
            directory.mkdir()
            public = "\n".join(json.dumps(row["public"], ensure_ascii=False) for row in loaded).encode()
            (directory / "questions.public.jsonl").write_bytes(public + b"\n")
            labels = [{"question_id": row["public"]["question_id"], "gold": row["gold"],
                       "topic": row["topic"], "source_type": row["source_type"]} for row in loaded]
            write_json(directory / "labels.evaluation-only.json", labels)
            ids = ["case-note:q1"]
            rows_out = []
            for index, row in enumerate(loaded):
                final = {"answer": "A", "used_memory_ids": ids if index else [], "condition_checks": [],
                         "memory_note": {"claim": "note", "applies_when": "rechecked", "exceptions": "conditions differ"}}
                retrieved = []
                if index:
                    source = loaded[0]["public"]
                    retrieved = [{"memory_id": ids[0], "source_sequence": 1,
                                  "source_question_id": "q1",
                                  "source_question": source["question"],
                                  "source_quotes": exact_quotes(source["question"], [source["question"][:30]]), "topic": "anemia"}]
                    final["condition_checks"] = [{"memory_id": ids[0], "current_conditions": "rechecked", "decision": "used"}]
                rows_out.append({"sequence": index + 1, "question_id": row["public"]["question_id"],
                    "gold": row["gold"], "topic": row["topic"], "source_type": row["source_type"],
                    "status": "ok", "prediction": "A", "raw_prediction": json.dumps(final), "correct": True,
                    "final": final, "memory_writes": 1, "admission": {"written": True,
                    "memory_id": ids[0] if not index else "case-note:q2", "clinical_status": "unverified_model_note",
                    "exact_quotes": 1}, "retrieval": {"memories": retrieved}})
            report = {"implementation": "child repository only; no parent methods/models imports",
                      "dataset": manifest, "config": {"dataset": str(dataset)}, "selected_questions": 2,
                      "public_input_sha256": sha256(public), "status": "complete",
                      "modes": {"protocol_vector_memory": {"status": "complete", "results": rows_out}}}
            report_path = directory / "report.json"
            write_json(report_path, report)
            self.assertEqual(verify_report(report_path)["methods"]["protocol_vector_memory"]["declared_reuse"], 1)
            rows_out[1]["retrieval"]["memories"][0]["source_sequence"] = 2
            write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "source quotation provenance|non-prior memory"):
                verify_report(report_path)

    def test_candidate_relation_annotations_are_used_only_by_evaluator_audit(self):
        from tools.moreagent_data import load_questions, sha256
        from tools.run_medical_workload import write_json

        with tempfile.TemporaryDirectory() as root:
            rows = example_rows("A")
            rows[0]["public"]["question_id"] = "prior"
            rows[1]["public"]["question_id"] = "later"
            source = Path(root) / "dataset.json"
            source.write_text(json.dumps({"overview": {"total_questions": 2}, "questions":[
                {"sequence": row["sequence"], **row["public"], "topic": row["topic"], "source_type": row["source_type"],
                 "evaluation_only": {"answer": row["gold"], "answer_text": row["public"]["options"][row["gold"]],
                                     "candidate_continuity": {"related_prior_question_ids": ["prior"] if row["sequence"] == 2 else []}}
                 } for row in rows]}))
            loaded, manifest = load_questions(source)
            directory = Path(root) / "run"; directory.mkdir()
            public_bytes = "\n".join(json.dumps(row["public"], ensure_ascii=False) for row in loaded).encode()
            (directory / "questions.public.jsonl").write_bytes(public_bytes + b"\n")
            write_json(directory / "labels.evaluation-only.json", [{"question_id": row["public"]["question_id"],
                "gold": row["gold"], "topic": row["topic"], "source_type": row["source_type"]} for row in loaded])
            note = {"claim": "possible relation", "applies_when": "conditions reviewed", "exceptions": "not established"}
            first = {"answer": "A", "used_memory_ids": [], "condition_checks": [], "memory_note": note}
            second = {"answer": "A", "used_memory_ids": ["case-note:prior"],
                      "condition_checks": [{"memory_id": "case-note:prior", "current_conditions": "reviewed", "decision": "used"}],
                      "memory_note": note}
            results = []
            for row, final in zip(loaded, [first, second]):
                retrieval = [] if row["sequence"] == 1 else [{"memory_id": "case-note:prior", "source_sequence": 1,
                    "source_question_id": "prior", "source_question": loaded[0]["public"]["question"],
                    "source_quotes": exact_quotes(loaded[0]["public"]["question"], [loaded[0]["public"]["question"][:30]]),
                    "topic": "anemia", "note": note, "clinical_status": "unverified_model_note"}]
                results.append({"sequence": row["sequence"], "question_id": row["public"]["question_id"],
                    "gold": row["gold"], "topic": row["topic"], "source_type": row["source_type"],
                    "status": "ok", "prediction": "A", "raw_prediction": json.dumps(final), "correct": True,
                    "final": final, "memory_writes": 1, "admission": {"written": True, "memory_id": "case-note:"+row["public"]["question_id"],
                    "clinical_status":"unverified_model_note", "exact_quotes":1}, "retrieval":{"memories":retrieval}})
            report = {"implementation":"child repository only; no parent methods/models imports", "dataset":manifest,
                "config":{"dataset":str(source)}, "selected_questions":2, "public_input_sha256":sha256(public_bytes),
                "status":"complete", "modes":{"protocol_vector_memory":{"status":"complete", "results":results}}}
            path = directory / "report.json"; write_json(path, report)
            audit = verify_report(path)["methods"]["protocol_vector_memory"]["offline_candidate_link_audit"]
            self.assertEqual(audit["matched_candidate_annotation_refs"], 1)
            self.assertEqual(audit["cross_topic_refs"], 1)


if __name__ == "__main__":
    unittest.main()
