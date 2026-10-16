"""Fixed related task sequence for the real-model continuity experiment."""

from dataclasses import dataclass
from typing import List

from .memory import MemoryItem


@dataclass(frozen=True)
class ContinuousTask:
    task_id: str
    group: str
    query: str
    expected_memory_id: str


def continuous_tasks() -> List[ContinuousTask]:
    return [
        ContinuousTask("retrieval-01", "retrieval", "Explain why normalized embeddings help semantic retrieval.", "evidence-retrieval"),
        ContinuousTask("retrieval-02", "retrieval", "How can vector similarity find paraphrased passages?", "evidence-retrieval"),
        ContinuousTask("retrieval-03", "retrieval", "Describe nearest-neighbor lookup for shared agent evidence.", "evidence-retrieval"),
        ContinuousTask("retrieval-04", "retrieval", "What information should a retriever return to a summarizer?", "evidence-retrieval"),
        ContinuousTask("retrieval-05", "retrieval", "How does a semantic index reduce repeated document scanning?", "evidence-retrieval"),
        ContinuousTask("memory-01", "memory", "Why should agent evidence persist beyond one task thread?", "evidence-memory"),
        ContinuousTask("memory-02", "memory", "How can a later workflow reuse an earlier agent summary?", "evidence-memory"),
        ContinuousTask("memory-03", "memory", "What metadata makes shared memories traceable and reusable?", "evidence-memory"),
        ContinuousTask("memory-04", "memory", "How does cross-task memory reduce duplicate computation?", "evidence-memory"),
        ContinuousTask("memory-05", "memory", "Explain how a team of agents can build on prior research.", "evidence-memory"),
    ]


def evidence_items() -> List[MemoryItem]:
    return [
        MemoryItem(
            "evidence-retrieval", "retriever", "2026-01-01T00:00:00+00:00", "semantic retrieval",
            "Normalized embeddings allow cosine similarity and nearest-neighbor search to retrieve semantically related evidence, including paraphrases, without scanning long text prompts repeatedly.",
            memory_type="semantic", tags=["retrieval"],
        ),
        MemoryItem(
            "evidence-memory", "summarizer", "2026-01-01T00:00:00+00:00", "shared memory",
            "A shared memory store persists evidence, summaries, source agents, timestamps, topics, and tags so later task threads can reuse prior research and avoid duplicate computation.",
            memory_type="semantic", tags=["memory"],
        ),
    ]
