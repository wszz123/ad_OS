"""Native medical MCQ graph: protocol, retrieval state, and online case-note memory."""

from dataclasses import asdict
import hashlib
import json
import math
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .medical_memory import exact_quotes, validate_public
from .memory import MemoryAdapter, MemoryItem
from .protocol import Capability, Message
from .state_exchange import StateExchange


MODES = ("text", "protocol", "protocol_vector", "protocol_vector_group_memory", "protocol_vector_memory")
WORKFLOW_GUIDES = {
    "guide:diagnosis": "Diagnosis workflow: compare the current patient's findings with alternatives; distinguish similar presentations.",
    "guide:mechanism": "Mechanism workflow: connect laboratory and clinical findings to the proposed physiological mechanism.",
    "guide:treatment": "Treatment workflow: recheck age, severity, comorbidities and contraindications before selecting management.",
    "guide:test": "Investigation workflow: choose a test that discriminates the competing explanations in this case.",
    "guide:memory": "Historical case notes are unverified and describe different patients; recheck applicability, exceptions and changed conditions.",
}
TOPICS = {"anemia", "renal", "thyroid", "other"}


class MedicalState(TypedDict, total=False):
    public: dict
    sequence: int
    plan: dict
    state_ref: str
    retrieval: dict
    draft: dict
    final: dict


def fixture_encode(texts, *, query=False):
    """Hashed bag-of-words fixture, not learned medical embeddings."""
    vectors = []
    for text in texts:
        vector = [0.0] * 32
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            index = int.from_bytes(hashlib.sha256(word.encode()).digest()[:2], "little") % len(vector)
            vector[index] += 1
        norm = math.sqrt(sum(value * value for value in vector)) or 1
        vectors.append([value / norm for value in vector])
    return vectors


class FixtureMedicalDecision:
    """Always chooses A, without labels; wiring evidence only."""
    model_id = "fixture-medical-always-A-v1"

    def __init__(self):
        self.call_stats, self.prompt_records = [], []

    def decide(self, role, context):
        if role == "planner":
            question = context["case"]["question"]
            lower = question.lower()
            topic = ("thyroid" if "thyroid" in lower else "renal" if "kidney" in lower or "creatinine" in lower
                     else "anemia" if "hemoglobin" in lower or "anemia" in lower else "other")
            return {"topic": topic, "search_query": question[:120], "checks": ["Compare current case conditions"],
                    "case_quotes": [question[:min(80, len(question))]]}
        if role == "executor":
            return {"answer": "A", "rationale": "Scripted wiring response; not a medical judgment."}
        memories = context["retrieval"]["memories"]
        used = [item["memory_id"] for item in memories[:1]]
        return {"answer": "A", "rationale": "Scripted review; not medical reasoning evidence.",
                "memory_note": {"claim": "Compare current findings before reusing a prior case note.",
                                "applies_when": "Current conditions have been rechecked.",
                                "exceptions": "Different patient conditions may invalidate prior suggestions."},
                "used_memory_ids": used,
                "condition_checks": [{"memory_id": item, "current_conditions": "Fixture checked the current public case; no clinical validation.",
                                      "decision": "used"} for item in used]}


class QwenMedicalDecision:
    def __init__(self, llm):
        self.llm = llm
        self.model_id = "qwen-medical-json:" + llm.model_path
        self.prompt_records = []

    @property
    def call_stats(self):
        return self.llm.call_stats

    def decide(self, role, context):
        instructions = {
            "planner": ('Analyze only the supplied case and options. Return JSON with topic (anemia/renal/thyroid/other), '
                        'search_query (at most 25 words), checks (1-4 short reasoning checks), '
                        'case_quotes (1-3 short EXACT substrings copied from the question stem, not options). '
                        'Do not choose the final option. Do not infer facts absent from the case.'),
            "executor": ('Solve the current medical multiple-choice case. Return JSON {"answer":"A|B|C|D","rationale":"brief explanation"}. '
                         'Use the current case as primary evidence. Workflow guides are generic checklists, not medical authorities. '
                         'Historical model notes are unverified, refer to different patients, and may be wrong. '
                         'Recheck age, time course and laboratory/clinical conditions; never copy a prior option letter. '
                         'Keep rationale within 60 words.'),
            "verifier": ('Review the candidate answer against the CURRENT case; correct it if needed. Return JSON with '
                         'answer (A/B/C/D), rationale (at most 60 words), '
                         'memory_note {claim,applies_when,exceptions} (one short sentence each, NO option letters), '
                         'used_memory_ids (only IDs actually supplied by retrieval), '
                         'condition_checks [{memory_id,current_conditions,decision:"used"|"rejected"}]. '
                         'For each used memory explicitly recheck current conditions. You may use no memory. '
                         'Notes are unverified model claims, NOT validated medical facts; source quotation only verifies provenance. '
                         'Do not claim that peer agreement establishes medical truth. Keep all fields concise.'),
        }
        messages = [{"role": "system", "content": instructions[role]},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
        budget = 512 if role == "verifier" else 256
        record = {"role": role, "messages": messages, "max_new_tokens": budget}
        self.prompt_records.append(record)
        before_calls = len(self.llm.call_stats)
        try:
            value = self.llm.generate_json(messages, max_new_tokens=budget)["value"]
            record["response"] = value
            return value
        except Exception as exc:
            record["error"] = str(exc)
            raise
        finally:
            if len(self.llm.call_stats) > before_calls:
                record["usage"] = dict(self.llm.call_stats[-1])


def nonempty(value, name, maximum=1600):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be nonempty text of at most {maximum} characters")


def validate_plan(value):
    if value.get("topic") not in TOPICS:
        raise ValueError("topic must be anemia, renal, thyroid or other")
    nonempty(value.get("search_query"), "search_query", 500)
    for name in ("checks", "case_quotes"):
        items = value.get(name)
        if not isinstance(items, list) or not 1 <= len(items) <= (4 if name == "checks" else 3):
            raise ValueError(name + " must be a bounded nonempty list")
        for item in items:
            nonempty(item, name, 600)


def validate_answer(value):
    if value.get("answer") not in set("ABCD"):
        raise ValueError("answer must be one of A, B, C, D")
    nonempty(value.get("rationale"), "rationale")


def validate_final(value, candidates):
    validate_answer(value)
    note = value.get("memory_note")
    if not isinstance(note, dict) or set(note) != {"claim", "applies_when", "exceptions"}:
        raise ValueError("memory_note needs claim, applies_when, exceptions")
    for key, text in note.items():
        nonempty(text, key, 500)
    used = value.get("used_memory_ids")
    allowed = {item["memory_id"] for item in candidates}
    if not isinstance(used, list) or any(not isinstance(item, str) for item in used) or len(set(used)) != len(used) or not set(used).issubset(allowed):
        raise ValueError("used_memory_ids must uniquely reference supplied candidates")
    checks = value.get("condition_checks")
    if not isinstance(checks, list) or len(checks) > len(allowed):
        raise ValueError("condition_checks must be a bounded list")
    checked = set()
    for check in checks:
        if not isinstance(check, dict) or check.get("memory_id") not in allowed or check["memory_id"] in checked:
            raise ValueError("condition check references unknown or repeated memory")
        checked.add(check["memory_id"])
        nonempty(check.get("current_conditions"), "current_conditions", 800)
        expected = "used" if check["memory_id"] in used else "rejected"
        if check.get("decision") != expected:
            raise ValueError("condition check decision must match used_memory_ids")
    if not set(used).issubset(checked):
        raise ValueError("each used memory needs a current-condition check")


def create_medical_corpus(encoder=None):
    corpus = MemoryAdapter()
    guide_ids = list(WORKFLOW_GUIDES)
    guide_vectors = encoder(list(WORKFLOW_GUIDES.values())) if encoder is not None else [None] * len(guide_ids)
    if len(guide_vectors) != len(guide_ids):
        corpus.close()
        raise ValueError("encoder must return one vector per workflow guide")
    for name, vector in zip(guide_ids, guide_vectors):
        corpus.put(MemoryItem(name, "runtime", "", "medical_workflow", WORKFLOW_GUIDES[name],
                              memory_type="generic_workflow_checklist_not_clinical_evidence",
                              embedding=list(map(float, vector)) if vector is not None else None))
    return corpus


def build_medical_graph(decision, memory, transport, *, encoder=None, use_vector=False,
                        memory_scope="none", top_k=2, schema_attempts=2, corpus=None):
    if memory_scope not in ("none", "group", "shared") or not 1 <= top_k <= 5 or not 1 <= schema_attempts <= 3:
        raise ValueError("invalid medical graph configuration")
    if use_vector and encoder is None:
        raise ValueError("vector mode requires an encoder")
    corpus = create_medical_corpus(encoder if use_vector else None) if corpus is None else corpus
    exchange, packets = StateExchange(), {}
    metrics = {"state_transfers": 0, "state_payload_bytes": 0, "state_metadata_bytes": 0,
               "memory_queries": 0, "memory_candidate_count": 0, "memory_rejected_ids": [],
               "schema_retries": 0, "role_calls": 0}

    def validated(role, context, validator):
        feedback = None
        for attempt in range(schema_attempts):
            payload = dict(context)
            if feedback:
                payload["schema_feedback"] = feedback
            metrics["role_calls"] += 1
            try:
                value = decision.decide(role, payload)
                if not isinstance(value, dict):
                    raise ValueError("role response must be a JSON object")
                validator(value)
                return value
            except (ValueError, KeyError, TypeError) as exc:
                feedback = str(exc)[:1200]
                if attempt + 1 == schema_attempts:
                    raise ValueError(f"{role} schema budget exhausted: {feedback}") from exc
                metrics["schema_retries"] += 1

    def planner(context):
        validate_public(context["public"])
        value = validated("planner", {"case": {key: context["public"][key] for key in ("question", "options")}}, validate_plan)
        state_ref = ""
        if use_vector:
            vector = encoder([value["search_query"]], query=True)[0]
            state_ref = context["public"]["question_id"] + ":medical-query"
            packet = exchange.encode(vector, state_id=state_ref, source_agent="planner", target_agent="retriever")
            packets[state_ref] = packet
            metadata = asdict(packet)
            metadata.pop("payload")
            metrics["state_transfers"] += 1
            metrics["state_payload_bytes"] += packet.payload_bytes
            metrics["state_metadata_bytes"] += len(json.dumps(metadata, separators=(",", ":")).encode())
        return {"plan": value, "state_ref": state_ref}

    def retriever(context):
        plan, public = context["plan"], context["public"]
        vector = exchange.decode(packets[context["state_ref"]]) if context["state_ref"] else None
        guides = (corpus.search_vector(vector, limit=2) if vector is not None
                  else corpus.search(plan["search_query"], limit=2))
        selected = {"candidates": [], "rejected_ids": [], "candidate_count": 0}
        if memory_scope != "none":
            metrics["memory_queries"] += 1
            selected = memory.select(public, context["sequence"], plan["topic"], vector=vector,
                                     query=plan["search_query"], scope=memory_scope, limit=top_k)
            metrics["memory_candidate_count"] += selected["candidate_count"]
            metrics["memory_rejected_ids"].extend(selected["rejected_ids"])
        return {"current_case_quotes": exact_quotes(public["question"], plan["case_quotes"]),
                "workflow_guides": [{"id": hit.item.memory_id, "text": hit.item.summary,
                                     "source_kind": "generic_workflow_not_medical_authority"} for hit in guides],
                "memories": selected["candidates"]}

    def executor(context):
        return validated("executor", {"case": {key: context["public"][key] for key in ("question", "options")}, "checks": context["plan"]["checks"],
                                      "retrieval": context["retrieval"]}, validate_answer)

    def verifier(context):
        return validated("verifier", {"case": {key: context["public"][key] for key in ("question", "options")}, "checks": context["plan"]["checks"],
                                      "retrieval": context["retrieval"], "draft": context["draft"]},
                         lambda value: validate_final(value, context["retrieval"]["memories"]))

    handlers = {"planner": ("plan_medical", planner), "retriever": ("retrieve_medical", retriever),
                "executor": ("solve_medical", executor), "verifier": ("review_medical", verifier)}
    transport.register(Capability("runtime", "runtime", ()), {})
    for agent, (action, handler) in handlers.items():
        def call(params, handler=handler):
            return {"result": json.dumps(handler(json.loads(params["context"])), ensure_ascii=False)}
        transport.register(Capability(agent, agent, (action,)), {action: call})

    def dispatch(state, source, target, keys):
        metrics["last_role"] = target
        response = transport.dispatch(Message(source, target, handlers[target][0],
                                             {"context": json.dumps({key: state[key] for key in keys}, ensure_ascii=False)}))
        return json.loads(response["result"])

    builder = StateGraph(MedicalState)
    builder.add_node("planner", lambda s: dispatch(s, "runtime", "planner", ("public",)))
    builder.add_node("retriever", lambda s: {"retrieval": dispatch(s, "planner", "retriever", ("public", "sequence", "plan", "state_ref"))})
    builder.add_node("executor", lambda s: {"draft": dispatch(s, "retriever", "executor", ("public", "plan", "retrieval"))})
    builder.add_node("verifier", lambda s: {"final": dispatch(s, "executor", "verifier", ("public", "plan", "retrieval", "draft"))})
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "executor")
    builder.add_edge("executor", "verifier")
    builder.add_edge("verifier", END)
    return builder.compile(), packets, metrics, corpus
