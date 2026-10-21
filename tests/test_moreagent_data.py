from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from tools.moreagent_data import (CONCISE_CONTRACT, agent_item, load_questions, parse_answer,
                                  response_messages, sha256, summarize)


class MoreAgentDataTests(unittest.TestCase):
    def example(self):
        return {"overview": {"total_questions": 1}, "questions": [{
            "sequence": 1, "question_id": "q1", "question": "Select a treatment.",
            "options": {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"},
            "topic": "private_topic", "task_focus_zh": "LEAK_FOCUS",
            "key_cues_from_existing_annotations": ["LEAK_CUE"],
            "evaluation_only": {"answer": "B", "answer_text": "Second", "rationale": "LEAK_RATIONALE",
                                "candidate_continuity": {"hint": "LEAK_RELATIONSHIP"}}}]}

    def load(self, data):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "dataset.json"
            path.write_text(json.dumps(data))
            return load_questions(path)

    def test_only_public_stem_and_options_enter_agent_input(self):
        rows, _ = self.load(self.example())
        item = agent_item(rows[0])
        serialized = json.dumps(item)
        self.assertNotIn("LEAK_", serialized)
        self.assertNotIn("private_topic", serialized)
        self.assertNotIn("evaluation_only", serialized)
        self.assertEqual(item["gold"], "")
        self.assertEqual(rows[0]["gold"], "B")
        self.assertEqual(set(rows[0]["public"]), {"question_id", "question", "options"})
        self.assertIn("B. Second", item["question"])

    def test_labels_ids_options_and_order_are_validated(self):
        for mutate in (
            lambda d: d["questions"][0].update(sequence=2),
            lambda d: d["questions"][0]["evaluation_only"].update(answer="E"),
            lambda d: d["questions"][0]["evaluation_only"].update(answer_text="Incorrect text"),
            lambda d: d["questions"][0]["options"].update(E="Fifth"),
            lambda d: d["overview"].update(total_questions=50),
        ):
            data = deepcopy(self.example())
            mutate(data)
            with self.assertRaises(ValueError):
                self.load(data)

    def test_answer_parser_does_not_guess_letters_from_reasoning(self):
        self.assertEqual(parse_answer("A and B are wrong; C is plausible."), (None, "missing"))
        self.assertEqual(parse_answer(r"Reasoning. \boxed{B}"), ("B", "boxed"))
        self.assertEqual(parse_answer(r"\boxed{A} then \boxed{\text{D}}"), ("D", "boxed"))
        self.assertEqual(parse_answer(" c. "), ("C", "plain_letter"))
        self.assertEqual(parse_answer(r"\boxed{E}"), (None, "missing"))

    def test_shared_output_contract_preserves_question_and_does_not_mutate_prompts(self):
        messages = [{"role": "system", "content": "Role instructions"}, {"role": "user", "content": "Public question and options"}]
        original = deepcopy(messages)
        changed = response_messages(messages, "concise")
        self.assertEqual(messages, original)
        self.assertEqual(changed[0], original[0])
        self.assertTrue(changed[1]["content"].startswith(original[1]["content"]))
        self.assertIn(CONCISE_CONTRACT, changed[1]["content"])
        self.assertEqual(response_messages(messages, "original"), original)

    def test_real_file_has_fifty_items_with_separate_authored_subset(self):
        path = Path(__file__).resolve().parents[1] / "moreagent_50_questions.json"
        if not path.exists():
            self.skipTest("user-supplied presentation dataset is not packaged")
        rows, manifest = load_questions(path)
        self.assertEqual(len(rows), 50)
        self.assertEqual(manifest["source_counts"], {"original_medqa": 30, "project_authored_medqa_style": 20})
        self.assertEqual(manifest["topic_counts"], {"anemia": 17, "renal": 17, "thyroid": 16})

    def test_errors_and_invalid_answers_stay_in_accuracy_denominator(self):
        rows = [
            {"status": "ok", "correct": True, "parse_status": "boxed", "topic": "anemia", "source_type": "original"},
            {"status": "ok", "correct": False, "parse_status": "missing", "topic": "anemia", "source_type": "original"},
            {"status": "error", "correct": False, "topic": "renal", "source_type": "authored"},
        ]
        result = summarize(rows)
        self.assertEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["invalid_answers"], 1)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["by_source_type"]["authored"]["accuracy"], 0)

    def test_tiled_disabled_realign_matches_full_fp32_normalization(self):
        import torch
        from tools.run_moreagent_eval import prepare_identity_normalization

        embedding = torch.nn.Embedding(17, 8).to(torch.bfloat16)
        underlying = SimpleNamespace(get_input_embeddings=lambda: embedding)
        wrapper = SimpleNamespace(model=underlying, args=SimpleNamespace(latent_space_realign=False),
                                  _latent_realign_matrices={})
        prepare_identity_normalization(wrapper, chunk_rows=3)
        matrix, norm = wrapper._latent_realign_matrices[id(underlying)]
        self.assertTrue(torch.equal(matrix, torch.eye(8)))
        self.assertTrue(torch.allclose(norm, embedding.weight.float().norm(dim=1).mean(), atol=1e-7, rtol=1e-7))
        wrapper.args.latent_space_realign = True
        with self.assertRaises(ValueError):
            prepare_identity_normalization(wrapper)

    def test_saved_scores_are_rechecked_against_raw_prediction_and_dataset(self):
        from tools.summarize_moreagent_eval import verify_report

        with tempfile.TemporaryDirectory() as root:
            dataset = Path(root) / "questions.json"
            dataset.write_text(json.dumps(self.example()))
            questions, manifest = load_questions(dataset)
            public = json.dumps(questions[0]["public"], ensure_ascii=False).encode()
            (Path(root) / "questions.public.jsonl").write_bytes(public + b"\n")
            (Path(root) / "labels.evaluation-only.json").write_text(json.dumps([
                {"question_id": "q1", "gold": "B", "topic": "private_topic", "source_type": "unknown"}]))
            row = {"question_id": "q1", "gold": "B", "topic": "private_topic", "source_type": "unknown",
                   "status": "ok", "raw_prediction": r"\boxed{B}", "prediction": "B",
                   "parse_status": "boxed", "correct": True}
            report = {"config": {"dataset": str(dataset)}, "dataset": manifest, "selected_questions": 1,
                      "public_input_sha256": sha256(public), "status": "complete",
                      "methods": {"baseline": {"results": [row]}}}
            path = Path(root) / "report.json"
            path.write_text(json.dumps(report))
            self.assertEqual(verify_report(path)[1]["methods"]["baseline"]["accuracy"], 1)
            row["correct"] = False
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "correctness disagrees"):
                verify_report(path)
            row["correct"] = True
            report["methods"]["baseline"]["results"] = []
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "missing predictions"):
                verify_report(path)


if __name__ == "__main__":
    unittest.main()
