"""Online, provenance-checked case notes; medical conclusions remain unverified."""

from datetime import datetime, timezone
import hashlib
import json
import re

from .memory import MemoryItem


def content_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def exact_quotes(question, quotes):
    result = []
    for quote in quotes:
        if not isinstance(quote, str) or len(quote.strip()) < 8:
            continue
        quote = quote.strip()
        start = question.find(quote)
        if start >= 0:
            result.append({"start": start, "end": start + len(quote), "text": quote})
    return result[:3]


def validate_public(public):
    if (not isinstance(public, dict) or set(public) != {"question_id", "question", "options"}
            or not isinstance(public["question_id"], str) or not public["question_id"]
            or not isinstance(public["question"], str) or not public["question"].strip()
            or not isinstance(public["options"], dict) or set(public["options"]) != set("ABCD")
            or any(not isinstance(value, str) or not value for value in public["options"].values())):
        raise ValueError("public input must contain only question_id, question and A-D options")


class MedicalMemory:
    MEMORY_TYPE = "medical_unverified_model_note"

    def __init__(self, backend, *, namespace, model_id, encoder=None):
        self.backend, self.namespace, self.model_id, self.encoder = backend, namespace, model_id, encoder
        self.sources = {}

    def select(self, public, sequence, topic, *, vector=None, query="", scope="shared", limit=2):
        validate_public(public)
        if scope not in ("group", "shared"):
            raise ValueError("memory selection scope must be group or shared")
        tags = ["run:" + self.namespace]
        if scope == "group":
            tags.append("topic:" + topic)
        count = max(1, self.backend.count())
        hits = (self.backend.search_vector(vector, tags=tags, limit=count) if vector is not None
                else self.backend.search(query, tags=tags, limit=count))
        selected, rejected = [], []
        for hit in hits:
            item = hit.item
            try:
                record = json.loads(item.evidence)
                payload = record["payload"]
                source = self.sources.get(payload["question_id"])
                valid = (item.memory_type == self.MEMORY_TYPE and item.source_agent == "verifier"
                         and payload["version"] == 1
                         and record["sha256"] == content_hash(payload)
                         and payload["namespace"] == self.namespace and payload["model_id"] == self.model_id
                         and payload["clinical_status"] == "unverified_model_note"
                         and payload["topic"] == item.task_topic
                         and (scope != "group" or payload["topic"] == topic)
                         and source is not None and source["sequence"] == payload["sequence"] < sequence
                         and payload["question_id"] != public["question_id"]
                         and payload["source_sha256"] == content_hash(source["public"])
                         and payload["source_question"] == source["public"]["question"]
                         and bool(payload["source_quotes"])
                         and payload["source_quotes"] == exact_quotes(payload["source_question"],
                                                                     [q["text"] for q in payload["source_quotes"]]))
            except (ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                rejected.append(item.memory_id)
                continue
            if len(selected) < limit:
                selected.append({"memory_id": item.memory_id, "source_question_id": payload["question_id"],
                                 "source_sequence": payload["sequence"], "source_question": payload["source_question"],
                                 "source_quotes": payload["source_quotes"], "note": payload["note"],
                                 "clinical_status": payload["clinical_status"], "topic": payload["topic"],
                                 "score": hit.score})
        return {"candidates": selected, "rejected_ids": rejected, "candidate_count": len(hits)}

    def publish(self, public, sequence, plan, final):
        """No gold/correctness argument exists; admission is not clinical validation."""
        validate_public(public)
        quotes = exact_quotes(public["question"], plan["case_quotes"])
        if not quotes:
            return {"written": False, "reason": "no_exact_case_quote"}
        note = final["memory_note"]
        if re.search(r"\b(?:answer|option|choice)\s*(?:is\s+)?[:=]?\s*[\(\[]?\s*[A-D]\b", " ".join(note.values()), re.I):
            return {"written": False, "reason": "option_letter_in_note"}
        payload = {"version": 1, "namespace": self.namespace, "model_id": self.model_id,
                   "question_id": public["question_id"], "sequence": sequence, "topic": plan["topic"],
                   "source_question": public["question"], "source_sha256": content_hash(public),
                   "source_quotes": quotes, "note": note, "clinical_status": "unverified_model_note",
                   "source_status": "exact_quote_from_public_question_not_medical_truth_validation"}
        vector = None
        if self.encoder is not None:
            text = plan["search_query"] + " " + " ".join(note.values())
            vector = list(map(float, self.encoder([text])[0]))
        memory_id = "case-note:" + public["question_id"]
        item = MemoryItem(memory_id, "verifier", datetime.now(timezone.utc).isoformat(), plan["topic"],
                          "UNVERIFIED model note: " + " ".join(note.values()),
                          memory_type=self.MEMORY_TYPE,
                          tags=["run:" + self.namespace, "topic:" + plan["topic"]],
                          evidence=json.dumps({"payload": payload, "sha256": content_hash(payload)}, ensure_ascii=False),
                          embedding=vector)
        self.backend.put(item)
        self.sources[public["question_id"]] = {"public": public, "sequence": sequence}
        return {"written": True, "memory_id": memory_id, "clinical_status": "unverified_model_note",
                "exact_quotes": len(quotes)}
