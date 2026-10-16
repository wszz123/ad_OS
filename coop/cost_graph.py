"""Four-role retrieval/calculation/synthesis graph with sourced-fact reuse."""

import json
import time
from typing import Dict, TypedDict

import numpy as np

from .calculator import calculate, decimal_value, expression_names
from .memory import MemoryItem
from .protocol import Capability, Message
from .state_exchange import StateExchange
from .workload import variable_catalog
from .plan_validation import validate_plan_expression
from .plan_repair import repair_plan as refine_plan


PLAN_PROMPT = (
    "You plan a sourced cost calculation. Return ONLY JSON with string keys search_query and expression. "
    "expression may use numbers, named variables and + - * / with parentheses. "
    "Use documented variables rather than inventing rates or prices. "
    "Use ONLY variables in this entity's source catalog and supplied user input names. "
    "Respect units and distinguish one-time from recurring costs. "
    "Each per-node expense scales with node_count. Power in watts must be divided by 1000 for kWh. "
    "A monthly cost repeats months times, but one-time capital costs do not. "
    "Storage counts replicas; egress uses original data only. "
    "Do not replace user inputs with constants, because the inputs can change. "
    'Output shape: {"search_query":"relevant source search","expression":"arithmetic using variable names"}. '
    "Return no markdown."
)
SUMMARY_PROMPT = (
    'Return ONLY JSON with this shape: {"value":"decimal number here","unit":"CNY",'
    '"citations":["source-id"],"explanation":"one brief sentence"}. '
    'The exact field name is "value", not "decimal string". Use the supplied verified tool result exactly for '
    "value/unit/citations. Explain it for the current task, not a previous task. No markdown."
)

REVIEW_PROMPT = (
    "Review a draft cost expression against the current task and source variable units. "
    "Fix any scope, quantity or time-period mistake. Respect exclusions: capital-only excludes "
    "recurring expenses; storage-only excludes traffic. Scale every per-node component by "
    "node_count. Annual totals repeat recurring costs, not capital cost. Use all explicit input "
    "parameter names rather than replacing them with numeric constants. "
    "Do not invent source values. Return ONLY JSON with string keys search_query and expression. "
    "Return a corrected expression if needed, otherwise retain the draft."
)


class CostState(TypedDict, total=False):
    task_id: str
    entity: str
    question: str
    inputs_json: str
    unit: str
    search_query: str
    expression: str
    state_id: str
    facts_json: str
    documents_json: str
    reused_memory_id: str
    tool_json: str
    answer_json: str
    task_metrics: Dict


def public_request(task):
    """Public request only; this function deliberately excludes expected answers."""
    return {"task_id": task.task_id, "entity": task.entity, "question": task.question,
            "inputs_json": json.dumps(task.inputs, sort_keys=True), "unit": task.unit}


def read_facts(documents, entity):
    facts = {}
    for document in documents:
        payload = json.loads(document["evidence"])
        if payload.get("entity") != entity or not isinstance(payload.get("facts"), dict):
            raise ValueError("evidence entity or fact schema mismatch")
        for name, value in payload["facts"].items():
            if not name.isidentifier() or name in facts:
                raise ValueError("duplicate or invalid fact name")
            facts[name] = {"value": str(decimal_value(value)), "source_id": document["memory_id"]}
    return facts


def document_snapshot(item):
    return {"memory_id": item.memory_id, "summary": item.summary, "evidence": item.evidence}


def build_cost_graph(llm, corpus, memory, transport, *, namespace, encoder=None,
                     reuse=False, threshold=0.55, max_new_tokens=192,
                     state_store=None, checkpointer=None, review_plan=False,
                     repair_plan=False, max_plan_repairs=2,
                     metadata_projection=False, deduplicate_validation=False,
                     repair_ledger=None):
    """Reuse parsed authoritative facts; fresh plans/calculations for new parameters.

    No task answer, required citation list or gold formula is available to agents.
    Memory writes retain source snapshots; reuse requires exact current-source
    identity and full coverage of the variables in the new generated expression.
    """
    from langgraph.graph import END, START, StateGraph

    if repair_plan and review_plan:
        raise ValueError("select either legacy review or bounded repair, not both")
    if repair_plan and (isinstance(max_plan_repairs, bool) or not isinstance(max_plan_repairs, int) or not 1 <= max_plan_repairs <= 5):
        raise ValueError("max_plan_repairs must be between 1 and 5")

    exchange = StateExchange()
    packets = state_store if state_store is not None else {}
    events = {}

    def plan(params):
        events[params["task_id"]] = {"memory_queries": 0, "accepted_reuse": 0,
                                     "corpus_searches": 0, "source_parses": 0,
                                     "tool_calls": 0, "embedding_sec": 0.0,
                                     "cache_validation_parses": 0, "plan_reviews": 0,
                                     "corpus_search_sec": 0.0, "memory_lookup_sec": 0.0}
        catalog = variable_catalog(params["entity"])
        prompt = (f"Entity: {params['entity']}\nTask: {params['question']}\n"
                  f"Source variable meanings and units (values must be retrieved): {json.dumps(catalog, sort_keys=True)}\n"
                  f"User inputs: {params['inputs_json']}\nUnit: {params['unit']}")
        if repair_plan:
            attempts = events[params["task_id"]]["plan_attempts"] = []
            try:
                value = refine_plan(
                    llm, prompt=prompt, catalog=catalog, inputs=json.loads(params["inputs_json"]),
                    question=params["question"], plan_prompt=PLAN_PROMPT, review_prompt=REVIEW_PROMPT,
                    max_new_tokens=max_new_tokens, max_repairs=max_plan_repairs, attempts=attempts,
                    ledger=repair_ledger, task_id=params["task_id"], policy_context=namespace,
                )
            finally:
                events[params["task_id"]].update(
                    plan_reviews=max(len(attempts) - 1, 0), plan_repairs=max(len(attempts) - 1, 0),
                    plan_attempt_count=len(attempts),
                    plan_repair_exhausted=int(bool(attempts) and not attempts[-1].get("accepted", False)),
                )
        else:
            value = llm.generate_json([
                {"role": "system", "content": PLAN_PROMPT}, {"role": "user", "content": prompt},
            ], max_new_tokens=max_new_tokens)["value"]
        draft = dict(value)
        if review_plan:
            issue = "No unit issue found; still check task scope and repeated costs."
            try:
                validate_plan_expression(value.get("expression"), json.loads(params["inputs_json"]))
            except ValueError as exc:
                issue = str(exc)
            value = llm.generate_json([
                {"role": "system", "content": REVIEW_PROMPT},
                {"role": "user", "content": prompt + f"\nDraft: {json.dumps(draft)}\nUnit/input check: {issue}"},
            ], max_new_tokens=max_new_tokens)["value"]
            events[params["task_id"]].update(plan_reviews=1, draft_expression=draft.get("expression"))
        if any(not isinstance(value.get(key), str) or not value[key].strip()
               for key in ("search_query", "expression")):
            raise ValueError("planner requires nonempty search_query and expression")
        _, names = expression_names(value["expression"])
        unknown = names - set(catalog) - set(json.loads(params["inputs_json"]))
        if unknown:
            raise ValueError(f"planner used variables outside entity schema: {sorted(unknown)}")
        if review_plan:
            validate_plan_expression(value["expression"], json.loads(params["inputs_json"]))
        state_id = ""
        if encoder is not None:
            started = time.perf_counter()
            query = params["entity"] + " " + value["search_query"]
            vector = encoder([query], query=True)[0]
            state_id = params["task_id"] + ":facts-query"
            packets[state_id] = exchange.encode(vector, state_id=state_id,
                                                source_agent="planner", target_agent="retriever")
            events[params["task_id"]]["embedding_sec"] = time.perf_counter() - started
        return {"search_query": value["search_query"], "expression": value["expression"], "state_id": state_id}

    def retrieve(params):
        event = events[params["task_id"]]
        _, variables = expression_names(params["expression"])
        required_facts = variables - set(json.loads(params["inputs_json"]))
        query = params["entity"] + " " + params["search_query"]
        state_id = params["state_id"]
        vector = exchange.decode(packets[state_id]) if state_id else None
        tags = [namespace, "entity:" + params["entity"]]
        if reuse and vector is not None:
            event["memory_queries"] += 1
            started = time.perf_counter()
            candidates = memory.search_vector(vector, tags=tags, limit=1000)
            event["memory_lookup_sec"] += time.perf_counter() - started
            for match in candidates:
                if match.score < threshold:
                    break
                try:
                    cached = json.loads(match.item.evidence)
                    documents = cached["documents"]
                    facts = cached["facts"]
                    if cached["task_id"] == params["task_id"] or not required_facts.issubset(facts):
                        continue
                    source_lookup = corpus.get_metadata if metadata_projection else corpus.get
                    if deduplicate_validation:
                        sources_match = True
                        for doc in documents:
                            source = source_lookup(doc["memory_id"])
                            if source is None or document_snapshot(source) != doc:
                                sources_match = False
                                break
                    else:
                        # Reference path deliberately retains the old two reads,
                        # so projection and deduplication can be ablated separately.
                        sources_match = not any(source_lookup(doc["memory_id"]) is None or
                                                document_snapshot(source_lookup(doc["memory_id"])) != doc for doc in documents)
                    if not sources_match:
                        continue
                    event["source_parses"] += len(documents)
                    event["cache_validation_parses"] += len(documents)
                    # Count validation parsing too: reuse does not eliminate this cost.
                    if facts != read_facts(documents, params["entity"]):
                        continue
                except (KeyError, TypeError, ValueError):
                    continue
                event.update(accepted_reuse=1, reuse_source_task=cached["task_id"],
                             reuse_source_agent=match.item.source_agent, reuse_score=match.score)
                return {"facts_json": json.dumps(facts, sort_keys=True),
                        "documents_json": json.dumps(documents, sort_keys=True),
                        "reused_memory_id": match.item.memory_id, "state_id": state_id}
        event["corpus_searches"] += 1
        entity_tags = ["entity:" + params["entity"]]
        started = time.perf_counter()
        hits = (corpus.search_vector(vector, tags=entity_tags, limit=100) if vector is not None
                else corpus.search(query, tags=entity_tags, limit=100))
        event["corpus_search_sec"] += time.perf_counter() - started
        documents = [document_snapshot(hit.item) for hit in hits
                     if any("fact:" + name in hit.item.tags for name in required_facts)]
        documents.sort(key=lambda doc: doc["memory_id"])
        facts = read_facts(documents, params["entity"])
        event["source_parses"] += len(documents)
        if not required_facts.issubset(facts):
            raise ValueError(f"retrieval missing required facts: {sorted(required_facts - set(facts))}")
        return {"facts_json": json.dumps(facts, sort_keys=True),
                "documents_json": json.dumps(documents, sort_keys=True),
                "reused_memory_id": "", "state_id": state_id}

    def execute(params):
        events[params["task_id"]]["tool_calls"] += 1
        result = calculate(params["expression"], json.loads(params["inputs_json"]),
                           json.loads(params["facts_json"]), params["unit"])
        return {"tool_json": json.dumps(result, sort_keys=True)}

    def summarize(params):
        answer = llm.generate_json([
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Entity: {params['entity']}\nTask: {params['question']}\nTool result: {params['tool_json']}"},
        ], max_new_tokens=max_new_tokens)["value"]
        if reuse:
            documents = json.loads(params["documents_json"])
            if documents:
                vectors = np.asarray([corpus.get(doc["memory_id"]).embedding for doc in documents], dtype=np.float32)
                vector = vectors.mean(axis=0)
                vector = vector / max(float(np.linalg.norm(vector)), 1e-8)
                memory.put(MemoryItem(
                    namespace + ":" + params["task_id"], "executor", "", params["entity"],
                    f"Validated facts for {params['entity']} from {','.join(doc['memory_id'] for doc in documents)}",
                    memory_type="sourced_facts", tags=[namespace, "entity:" + params["entity"]],
                    evidence=json.dumps({"task_id": params["task_id"], "documents": documents,
                                         "facts": json.loads(params["facts_json"]),
                                         "tool_result": json.loads(params["tool_json"]), "answer": answer}, sort_keys=True),
                    embedding=vector.tolist(),
                ))
        return {"answer_json": json.dumps(answer, sort_keys=True)}

    transport.register(Capability("coordinator", "runtime", ()), {})
    handlers = (("planner", "planning", "plan", plan), ("retriever", "retrieval", "retrieve", retrieve),
                ("executor", "tool_execution", "calculate", execute), ("summarizer", "summary", "summarize", summarize))
    for agent, role, action, handler in handlers:
        transport.register(Capability(agent, role, (action,)), {action: handler})

    def dispatch(state, source, target, action, keys):
        if "task_metrics" in state:
            events[state["task_id"]] = dict(state["task_metrics"])
        result = transport.dispatch(Message(source, target, action, {key: state[key] for key in keys},
                                              message_id=state["task_id"] + ":" + action))
        return {**result, "task_metrics": dict(events[state["task_id"]])}

    builder = StateGraph(CostState)
    builder.add_node("planner", lambda s: dispatch(s, "coordinator", "planner", "plan",
                                                   ("task_id", "entity", "question", "inputs_json", "unit")))
    builder.add_node("retriever", lambda s: dispatch(s, "planner", "retriever", "retrieve",
                                                     ("task_id", "entity", "search_query", "expression", "inputs_json", "state_id")))
    builder.add_node("executor", lambda s: dispatch(s, "retriever", "executor", "calculate",
                                                    ("task_id", "expression", "facts_json", "inputs_json", "unit")))
    builder.add_node("summarizer", lambda s: dispatch(s, "executor", "summarizer", "summarize",
                                                      ("task_id", "entity", "question", "tool_json", "facts_json", "documents_json")))
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "executor")
    builder.add_edge("executor", "summarizer")
    builder.add_edge("summarizer", END)
    return builder.compile(checkpointer=checkpointer), packets, events
