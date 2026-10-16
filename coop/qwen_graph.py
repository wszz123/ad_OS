"""Three-role LangGraph using local Qwen for planning and synthesis."""

from typing import Callable, Dict, TypedDict

from .memory import MemoryBackend
from .protocol import Capability, Message, ProtocolAdapter
from .state_exchange import StateExchange, StatePacket


class QwenTaskState(TypedDict, total=False):
    task_id: str
    query: str
    plan: Dict
    evidence: str
    evidence_id: str
    state_id: str
    answer: str
    usage: list


def build_qwen_graph(llm, memory: MemoryBackend, protocol: ProtocolAdapter,
                     *, max_new_tokens: int = 96, query_encoder: Callable = None,
                     state_packets: Dict[str, StatePacket] = None,
                     checkpointer=None):
    """Build a coordinator -> planner -> retriever -> summarizer workflow.

    Qwen owns plan/summary generation. Retrieval remains a deterministic tool-like
    role so the smoke test can verify evidence grounding independently.
    """
    from langgraph.graph import END, START, StateGraph
    exchange = StateExchange()
    state_packets = state_packets if state_packets is not None else {}

    def plan(params):
        result = llm.generate_json([
            {"role": "system", "content": (
                "Return only a JSON object with keys topic and search_query. "
                "Keep each value concise. Do not include markdown."
            )},
            {"role": "user", "content": params["query"]},
        ], max_new_tokens=max_new_tokens)
        value = result["value"]
        if not isinstance(value.get("topic"), str) or not isinstance(value.get("search_query"), str):
            raise ValueError("planner JSON requires string topic and search_query")
        update = {"plan": value, "usage": [result]}
        if query_encoder is not None:
            vector = query_encoder([value["search_query"]], query=True)[0]
            state_id = params.get("task_id", "task") + ":query"
            state_packets[state_id] = exchange.encode(
                vector, state_id=state_id, source_agent="planner", target_agent="retriever",
                encoding="float16",
            )
            update["state_id"] = state_id
        return update

    def retrieve(params):
        if "state_id" in params:
            vector = exchange.decode(state_packets[params["state_id"]])
            results = memory.search_vector(vector, limit=1)
        else:
            results = memory.search(params["search_query"], limit=1)
        if not results:
            return {"evidence_id": "", "evidence": "No matching stored evidence."}
        return {"evidence_id": results[0].item.memory_id,
                "evidence": results[0].item.summary}

    def summarize(params):
        result = llm.generate([
            {"role": "system", "content": (
                "Answer the user's task using the supplied evidence. If evidence is insufficient, "
                "say so. Keep the answer concise and do not invent sources."
            )},
            {"role": "user", "content": (
                f"Task: {params['query']}\nPlan: {params['plan']}\n"
                f"Evidence: {params['evidence']}"
            )},
        ], max_new_tokens=max_new_tokens)
        return {"answer": result["text"], "usage": [result]}

    protocol.register(Capability("coordinator", "runtime", ()), {})
    protocol.register(Capability("planner", "planning", ("plan",)), {"plan": plan})
    protocol.register(Capability("retriever", "retrieval", ("retrieve",)), {"retrieve": retrieve})
    protocol.register(Capability("summarizer", "summary", ("summarize",)), {"summarize": summarize})

    def planner_node(state):
        return protocol.dispatch(Message("coordinator", "planner", "plan", {
            "task_id": state["task_id"],
            "query": state["query"],
        }))

    def retriever_node(state):
        result = protocol.dispatch(Message("planner", "retriever", "retrieve", {
            "search_query": state["plan"]["search_query"],
            **({"state_id": state["state_id"]} if "state_id" in state else {}),
        }))
        return result

    def summarizer_node(state):
        return protocol.dispatch(Message("retriever", "summarizer", "summarize", {
            "query": state["query"], "plan": state["plan"],
            "evidence": state["evidence"],
        }))

    builder = StateGraph(QwenTaskState)
    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "summarizer")
    builder.add_edge("summarizer", END)
    return builder.compile(checkpointer=checkpointer)
