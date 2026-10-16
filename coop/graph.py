"""LangGraph integration check: three roles with independent adapter boundaries."""

from typing import Callable, Dict, List, TypedDict

from .memory import MemoryBackend
from .protocol import Capability, Message, ProtocolAdapter
from .state_exchange import StateExchange, StatePacket


class TaskState(TypedDict, total=False):
    task_id: str
    query: str
    state_id: str
    memory_id: str
    summary: str
    answer: str


def build_memory_graph(memory: MemoryBackend, query_encoder: Callable,
                       protocol: ProtocolAdapter, packets: Dict[str, StatePacket],
                       checkpointer=None):
    """Adapters are dependencies; graph state carries references, not large arrays.

    Packets currently live in a process-local registry. This validates consumption
    of binary states, but is not IPC or durable state recovery.
    """
    from langgraph.graph import END, START, StateGraph

    exchange = StateExchange()

    def plan(params):
        vector = query_encoder([params["query"]])[0]
        state_id = params["task_id"] + ":query"
        packets[state_id] = exchange.encode(
            vector, state_id=state_id, source_agent="planner", target_agent="retriever"
        )
        return {"state_id": state_id}

    def retrieve(params):
        vector = exchange.decode(packets[params["state_id"]])
        matches = memory.search_vector(vector, limit=1)
        return {"memory_id": matches[0].item.memory_id if matches else "",
                "summary": matches[0].item.summary if matches else "No memory found"}

    def summarize(params):
        return {"answer": params["summary"]}

    protocol.register(Capability("coordinator", "runtime", ()), {})
    for agent_id, role, action, handler in (
        ("planner", "planning", "plan", plan),
        ("retriever", "retrieval", "retrieve", retrieve),
        ("summarizer", "summary", "summarize", summarize),
    ):
        protocol.register(Capability(agent_id, role, (action,)), {action: handler})

    def planner_node(state):
        return protocol.dispatch(Message("coordinator", "planner", "plan", {
            "task_id": state["task_id"], "query": state["query"],
        }))

    def retriever_node(state):
        evidence = protocol.dispatch(Message("planner", "retriever", "retrieve", {
            "state_id": state["state_id"],
        }))
        return evidence

    def summarizer_node(state):
        return protocol.dispatch(Message("retriever", "summarizer", "summarize", {
            "memory_id": state["memory_id"], "summary": state["summary"],
        }))

    builder = StateGraph(TaskState)
    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "summarizer")
    builder.add_edge("summarizer", END)
    return builder.compile(checkpointer=checkpointer)
