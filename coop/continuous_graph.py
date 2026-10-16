"""Shared real-model workflow for communication and memory ablations."""

from dataclasses import asdict
import json
import time
from typing import Dict, TypedDict

from .memory import MemoryItem
from .protocol import Capability, Message
from .state_exchange import StateExchange


PLAN_SYSTEM = "Return only a JSON object with string keys topic and search_query. Keep both concise. No markdown."
SUMMARY_SYSTEM = "Answer the task using only the supplied evidence. Say if evidence is insufficient. Be concise and do not invent sources."


class ContinuousState(TypedDict, total=False):
    task_id: str
    query: str
    topic: str
    search_query: str
    state_id: str
    reused_memory_id: str
    evidence_id: str
    evidence: str
    answer: str
    task_metrics: Dict


def build_continuous_graph(llm, corpus, memory, transport, *, encoder=None,
                           reuse=False, namespace="default", threshold=0.60,
                           margin=0.05, max_new_tokens=96, checkpointer=None,
                           state_store=None, interrupt_after=None):
    """Four modes share handlers and LLM prompts; gold labels never enter nodes.

    Reuse means reusing an evidence retrieval strategy, not copying an old answer.
    A hit avoids one planner generation and one corpus search. Every task still
    receives a newly generated summary grounded in the original evidence.
    """
    from langgraph.graph import END, START, StateGraph

    exchange = StateExchange()
    packets = state_store if state_store is not None else {}
    events = {}

    def reusable(vector, task_id):
        ranked = memory.search_vector(vector, tags=[namespace], limit=1000)
        valid = []
        seen = set()
        for match in ranked:
            try:
                provenance = json.loads(match.item.evidence)
                source = corpus.get(provenance["evidence_id"])
            except (ValueError, KeyError, TypeError):
                continue
            if (source is None or provenance.get("task_id") == task_id
                    or provenance.get("evidence") != source.summary
                    or source.memory_id in seen):
                continue
            seen.add(source.memory_id)
            valid.append(match)
        if not valid or valid[0].score < threshold:
            return None
        if len(valid) > 1 and valid[0].score - valid[1].score < margin:
            return None
        return valid[0]

    def plan(params):
        task_id = params["task_id"]
        event = events[task_id] = {"memory_queries": 0,
                                   "accepted_reuse": 0, "corpus_searches": 0,
                                   "embedding_sec": 0.0, "memory_write_sec": 0.0}
        update = {"state_id": "", "reused_memory_id": ""}
        if encoder is not None:
            started = time.perf_counter()
            vector = encoder([params["query"]], query=True)[0]
            event["embedding_sec"] = time.perf_counter() - started
            state_id = task_id + ":query"
            packets[state_id] = exchange.encode(vector, state_id=state_id,
                                                source_agent="planner", target_agent="retriever")
            update["state_id"] = state_id
            if reuse:
                event["memory_queries"] = 1
                match = reusable(vector, task_id)
                if match is not None:
                    event.update({"accepted_reuse": 1, "reuse_score": match.score,
                                  "reuse_source_agent": match.item.source_agent,
                                  "reuse_source_task": json.loads(match.item.evidence)["task_id"]})
                    update.update(topic=match.item.task_topic, search_query=params["query"],
                                  reused_memory_id=match.item.memory_id)
                    return update
        result = llm.generate_json([
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": params["query"]},
        ], max_new_tokens=max_new_tokens)["value"]
        if any(not isinstance(result.get(key), str) or not result[key].strip()
               for key in ("topic", "search_query")):
            raise ValueError("planner requires nonempty string topic and search_query")
        return {**update, "topic": result["topic"], "search_query": result["search_query"]}

    def retrieve(params):
        event = events[params["task_id"]]
        if params["state_id"]:
            vector = exchange.decode(packets[params["state_id"]])
        else:
            vector = None
        if params["reused_memory_id"]:
            item = memory.get(params["reused_memory_id"])
            if item is None:
                raise ValueError("accepted memory disappeared before retrieval")
            provenance = json.loads(item.evidence)
            return {"evidence_id": provenance["evidence_id"], "evidence": provenance["evidence"]}
        event["corpus_searches"] += 1
        hits = (corpus.search_vector(vector, limit=1) if vector is not None
                else corpus.search(params["search_query"], limit=1))
        if not hits:
            return {"evidence_id": "", "evidence": "No matching evidence."}
        return {"evidence_id": hits[0].item.memory_id, "evidence": hits[0].item.summary}

    def summarize(params):
        result = llm.generate([
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": f"Task: {params['query']}\nEvidence: {params['evidence']}"},
        ], max_new_tokens=max_new_tokens)
        if reuse and params["evidence_id"]:
            started = time.perf_counter()
            source = corpus.get(params["evidence_id"])
            provenance = {"task_id": params["task_id"], "query": params["query"],
                          "evidence_id": params["evidence_id"], "evidence": params["evidence"]}
            memory.put(MemoryItem(
                namespace + ":" + params["task_id"], "summarizer", "", params["topic"],
                result["text"], memory_type="evidence_strategy", tags=[namespace],
                evidence=json.dumps(provenance, ensure_ascii=False), embedding=source.embedding,
            ))
            events[params["task_id"]]["memory_write_sec"] = time.perf_counter() - started
        return {"answer": result["text"]}

    transport.register(Capability("coordinator", "runtime", ()), {})
    for agent, role, action, handler in (("planner", "planning", "plan", plan),
                                       ("retriever", "retrieval", "retrieve", retrieve),
                                       ("summarizer", "summary", "summarize", summarize)):
        transport.register(Capability(agent, role, (action,)), {action: handler})

    def dispatch(state, source, target, action, keys):
        if "task_metrics" in state:
            events[state["task_id"]] = dict(state["task_metrics"])
        result = transport.dispatch(Message(source, target, action, {key: state[key] for key in keys},
                                              message_id=state["task_id"] + ":" + action))
        return {**result, "task_metrics": dict(events[state["task_id"]])}

    builder = StateGraph(ContinuousState)
    builder.add_node("planner", lambda s: dispatch(s, "coordinator", "planner", "plan", ("task_id", "query")))
    builder.add_node("retriever", lambda s: dispatch(s, "planner", "retriever", "retrieve",
                                                    ("task_id", "search_query", "state_id", "reused_memory_id")))
    builder.add_node("summarizer", lambda s: dispatch(s, "retriever", "summarizer", "summarize",
                                                     ("task_id", "query", "topic", "evidence_id", "evidence")))
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "summarizer")
    builder.add_edge("summarizer", END)
    return builder.compile(checkpointer=checkpointer, interrupt_after=interrupt_after), packets, events


def state_sizes(packet):
    """Count payload and the complete metadata envelope independently."""
    metadata = asdict(packet)
    metadata.pop("payload")
    metadata_bytes = len(json.dumps(metadata, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return packet.payload_bytes, metadata_bytes
