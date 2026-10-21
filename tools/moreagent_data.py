"""Strict evaluation-side extraction from a presentation JSON, not an agent input."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def load_questions(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    rows = data.get("questions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset must contain a nonempty questions list")
    questions, seen = [], set()
    for position, row in enumerate(rows, 1):
        if row.get("sequence") != position:
            raise ValueError("sequence must be consecutive and preserve the supplied order")
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id or question_id in seen:
            raise ValueError("question IDs must be nonempty and unique")
        seen.add(question_id)
        question, options = row.get("question"), row.get("options")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"missing question: {question_id}")
        if (not isinstance(options, dict) or set(options) != set("ABCD")
                or any(not isinstance(text, str) or not text.strip() for text in options.values())):
            raise ValueError(f"original MedQA prompts require four nonempty A-D options: {question_id}")
        evaluation = row.get("evaluation_only", {})
        gold = evaluation.get("answer")
        if gold not in options:
            raise ValueError(f"invalid evaluator answer: {question_id}")
        if evaluation.get("answer_text") is not None and evaluation["answer_text"].strip() != options[gold].strip():
            raise ValueError(f"answer label/text mismatch: {question_id}")
        # Deliberate whitelist: no focus/cues/rationale/candidate relationships.
        public = {"question_id": question_id, "question": question,
                  "options": {letter: options[letter] for letter in "ABCD"}}
        questions.append({"sequence": position, "public": public, "gold": gold,
                          "topic": row.get("topic", "unknown"), "source_type": row.get("source_type", "unknown"),
                          "evaluator_related_prior_ids": list(evaluation.get("candidate_continuity", {}).get(
                              "related_prior_question_ids", []))})
    declared = data.get("overview", {}).get("total_questions")
    if declared is not None and declared != len(questions):
        raise ValueError("declared question count differs from actual rows")
    manifest = {
        "dataset_name": data.get("dataset_name"), "schema_version": data.get("schema_version"),
        "source_sha256": sha256(raw), "total_questions": len(questions),
        "topic_counts": dict(Counter(row["topic"] for row in questions)),
        "source_counts": dict(Counter(row["source_type"] for row in questions)),
        "source_purpose": data.get("purpose_zh"),
        "limitations": data.get("overview", {}).get("limitations_zh", []),
        "conversion": "explicit question/options whitelist; evaluation_only and all presentation annotations excluded",
    }
    return questions, manifest


def agent_item(row):
    public = row["public"]
    question = public["question"] + "\n\nOptions:\n" + "\n".join(
        f"{letter}. {public['options'][letter]}" for letter in "ABCD")
    return {"task_id": public["question_id"], "question": question, "gold": "", "solution": ""}


BOX = re.compile(r"\\boxed\s*\{\s*(?:([A-D])|\\(?:text|mathrm|mathbf)\s*\{\s*([A-D])\s*\})\s*\}", re.I)

CONCISE_CONTRACT = (
    "Keep the visible response within 60 words while following the assigned role. "
    "If your role requires the final answer, put the selected option in \\boxed{LETTER} "
    "at the beginning, followed by a brief justification. Planning/review roles must not give the final answer."
)


def response_messages(messages, style):
    copied = [dict(message) for message in messages]
    if style == "concise":
        user_positions = [index for index, message in enumerate(copied) if message["role"] == "user"]
        if not user_positions:
            raise ValueError("no user message available for the shared output contract")
        index = user_positions[-1]
        copied[index]["content"] += "\n\nResponse contract:\n" + CONCISE_CONTRACT
    elif style != "original":
        raise ValueError("unknown response style")
    return copied


def parse_answer(text):
    matches = list(BOX.finditer(text))
    if matches:
        match = matches[-1]
        return (match.group(1) or match.group(2)).upper(), "boxed"
    if re.fullmatch(r"\s*[A-D][.。]?\s*", text, re.I):
        return text.strip().rstrip(".。").upper(), "plain_letter"
    return None, "missing"


def summarize(rows):
    successful = [row for row in rows if row.get("status") == "ok"]
    result = {"attempted": len(rows), "completed": len(successful),
              "correct": sum(row.get("correct", False) for row in rows),
              "invalid_answers": sum(row.get("parse_status") == "missing" for row in successful),
              "errors": sum(row.get("status") != "ok" for row in rows)}
    result["accuracy"] = result["correct"] / len(rows) if rows else None
    for key in ("elapsed_sec", "prompt_tokens", "generated_tokens", "intermediate_text_tokens",
                "forward_calls", "text_generation_calls", "latent_rollout_calls", "latent_recurrent_steps",
                "kv_tensor_bytes_logical", "protocol_frame_bytes"):
        result[key] = sum(row.get(key, 0) for row in rows)
    for field in ("topic", "source_type"):
        groups = {}
        for row in rows:
            cell = groups.setdefault(row[field], {"count": 0, "correct": 0})
            cell["count"] += 1
            cell["correct"] += int(row.get("correct", False))
        for cell in groups.values():
            cell["accuracy"] = cell["correct"] / cell["count"]
        result["by_" + field] = groups
    return result
