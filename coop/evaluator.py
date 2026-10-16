"""Paired, deterministic mechanism evaluation for collaboration adapters."""

from dataclasses import asdict, dataclass
import json
import time
from typing import Callable, Dict, Iterable, List, Optional

from .memory import MemoryAdapter, MemoryItem
from .protocol import Capability, Message, ProtocolAdapter
from .state_exchange import StateExchange


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    group: str
    topic: str
    query: str
    expected_memory_id: str


def related_tasks() -> List[EvalTask]:
    """Two five-task sequences with different wording and a shared topic corpus."""
    cases = [
        ("retrieval", "vector search", "How does semantic vector search rank evidence?", "m-vector"),
        ("retrieval", "vector search", "Find related passages by meaning instead of exact words.", "m-vector"),
        ("retrieval", "vector search", "Which similarity score compares normalized embeddings?", "m-vector"),
        ("retrieval", "vector search", "Explain nearest-neighbor lookup over text representations.", "m-vector"),
        ("retrieval", "vector search", "How can semantic recall find paraphrased evidence?", "m-vector"),
        ("collaboration", "shared memory", "How do agents retain evidence across separate tasks?", "m-memory"),
        ("collaboration", "shared memory", "Reuse another worker's saved conclusions in a later run.", "m-memory"),
        ("collaboration", "shared memory", "What lets multiple agents retrieve persistent experience?", "m-memory"),
        ("collaboration", "shared memory", "Carry prior task summaries into a new workflow.", "m-memory"),
        ("collaboration", "shared memory", "How can a team avoid repeating previous research?", "m-memory"),
    ]
    return [EvalTask(f"task-{index + 1:02d}", group, topic, query, memory_id)
            for index, (group, topic, query, memory_id) in enumerate(cases)]


CORPUS = [
    MemoryItem("m-vector", "retriever", "2026-01-01T00:00:00+00:00", "vector search",
               "Normalized embeddings support cosine similarity and nearest-neighbor semantic retrieval.",
               tags=["retrieval", "vectors"]),
    MemoryItem("m-memory", "summarizer", "2026-01-01T00:00:00+00:00", "shared memory",
               "Persistent shared memory stores evidence and summaries for reuse by agents across tasks.",
               tags=["collaboration", "memory"]),
]


MODES = ("text", "protocol", "protocol_vector", "protocol_vector_memory")


def _token_count(tokenizer, text: str) -> int:
    if tokenizer is None:
        return -1
    return len(tokenizer.encode(text, add_special_tokens=False))


def _keyword_retrieve(query: str, corpus: Iterable[MemoryItem]) -> MemoryItem:
    query_terms = set(query.lower().split())
    ranked = sorted(corpus, key=lambda item: len(query_terms & set(
        (item.task_topic + " " + item.summary + " " + " ".join(item.tags)).lower().split()
    )), reverse=True)
    return ranked[0]


def _base_frames(task: EvalTask) -> List[str]:
    history = "\n".join(
        f"Planner: inspect {task.topic}; Retriever: search evidence; Summarizer: retain all relevant context."
        for _ in range(4)
    )
    return [
        f"Orchestrator -> Planner: Create a detailed plan for this request.\nRequest: {task.query}\n{history}",
        f"Planner -> Orchestrator: Plan: identify {task.topic}, search relevant evidence, and prepare a grounded summary.",
        f"Orchestrator -> Retriever: Search all available evidence for this request.\nRequest: {task.query}\nPlan and history:\n{history}",
        f"Retriever -> Orchestrator: I searched the evidence collection and found candidate passages related to {task.topic}. The best candidate should be checked against the request.",
        f"Orchestrator -> Summarizer: Summarize the evidence and prior agent context.\nRequest: {task.query}\n{history}",
        f"Summarizer -> Orchestrator: The evidence concerns {task.topic}. I will provide the supported conclusion and retain its source for later related tasks.",
    ]


def run_evaluation(
    *, tokenizer=None, embedder: Optional[Callable] = None,
    modes: Iterable[str] = MODES, repeat: int = 1,
) -> Dict:
    """Run the same ten ordered tasks under communication/memory ablations."""
    if repeat < 1:
        raise ValueError("repeat must be positive")
    selected_modes = list(modes)
    unknown = set(selected_modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown evaluation modes: {sorted(unknown)}")
    tasks = related_tasks() * repeat
    corpus = [MemoryItem(**asdict(item)) for item in CORPUS]
    corpus_by_id = {item.memory_id: item for item in corpus}

    query_vectors = {}
    corpus_vectors = {}
    embedding_prepare_sec = 0.0
    if embedder is not None and any("vector" in mode for mode in selected_modes):
        embedding_started = time.perf_counter()
        query_values = embedder([task.query for task in tasks], query=True)
        document_values = embedder([item.summary for item in corpus], query=False)
        embedding_prepare_sec = time.perf_counter() - embedding_started
        query_vectors = {task.task_id + f"-{index}": query_values[index].tolist()
                         for index, task in enumerate(tasks)}
        corpus_vectors = {item.memory_id: document_values[index].tolist()
                          for index, item in enumerate(corpus)}

    reports = {}
    for mode in selected_modes:
        started = time.perf_counter()
        protocol = ProtocolAdapter()
        if mode != "text":
            protocol.register(Capability("coordinator", "runtime", ()), {})
            for agent, role, action in (("planner", "planning", "plan"),
                                        ("retriever", "retrieval", "retrieve"),
                                        ("summarizer", "summary", "summarize")):
                protocol.register(Capability(agent, role, (action,)), {action: lambda params: params})
        memory = MemoryAdapter(embedder=None)
        memory_ids = set()
        exchange = StateExchange()
        rows = []
        text_chars = 0
        token_count = 0
        message_count = 0
        state_bytes = 0
        state_transfers = 0
        memory_queries = 0
        memory_hits = 0
        correct = 0

        for index, task in enumerate(tasks):
            frames = []
            if mode == "text":
                frames = _base_frames(task)
                for frame in frames:
                    text_chars += len(frame)
                    token_count += _token_count(tokenizer, frame)
                message_count += len(frames)
            else:
                actions = [
                    Message("coordinator", "planner", "plan", {"task_id": task.task_id, "query": task.query}),
                    Message("planner", "retriever", "retrieve", {"topic": task.topic}),
                    Message("retriever", "summarizer", "summarize", {"evidence_ref": task.expected_memory_id}),
                ]
                for message in actions:
                    result = protocol.dispatch(message)
                frames = list(protocol.frames)
                protocol.frames.clear()
                message_count += len(frames)
                text_chars += sum(len(frame) for frame in frames)
                token_count += sum(_token_count(tokenizer, frame) for frame in frames)

            if mode in ("protocol_vector", "protocol_vector_memory") and embedder is not None:
                key = task.task_id + f"-{index}"
                packet = exchange.encode(query_vectors[key], state_id=key,
                                         source_agent="planner", target_agent="retriever")
                vector = exchange.decode(packet)
                state_bytes += packet.payload_bytes
                state_transfers += 1
            else:
                vector = None

            if mode == "protocol_vector_memory":
                memory_queries += 1
                results = memory.search(task.query, tags=[task.group], limit=1)
                if results:
                    memory_hits += 1
                    selected = results[0].item
                elif vector is not None and corpus_vectors:
                    selected = max(corpus, key=lambda item: MemoryAdapter._cosine(
                        vector.tolist(), corpus_vectors[item.memory_id]))
                elif vector is not None:
                    selected = _keyword_retrieve(task.query, corpus)
                else:
                    selected = _keyword_retrieve(task.query, corpus)
            elif vector is not None and corpus_vectors:
                selected = max(corpus, key=lambda item: MemoryAdapter._cosine(
                    vector.tolist(), corpus_vectors[item.memory_id]))
            else:
                selected = _keyword_retrieve(task.query, corpus)

            is_correct = selected.memory_id == task.expected_memory_id
            correct += int(is_correct)
            rows.append({
                "task_id": task.task_id, "group": task.group,
                "expected_memory_id": task.expected_memory_id,
                "selected_memory_id": selected.memory_id,
                "correct": is_correct,
            })

            if mode == "protocol_vector_memory" and is_correct and task.expected_memory_id not in memory_ids:
                source = corpus_by_id[task.expected_memory_id]
                memory.put(MemoryItem(
                    source.memory_id, "retriever", source.created_at, task.topic,
                    source.summary, tags=[task.group],
                    embedding=corpus_vectors.get(source.memory_id),
                ))
                memory_ids.add(source.memory_id)

        memory.close()
        elapsed = time.perf_counter() - started
        denominator = len(tasks)
        reports[mode] = {
            "task_count": denominator,
            "correct": correct,
            "accuracy": correct / denominator,
            "message_count": message_count,
            "text_chars": text_chars,
            "qwen_token_count": token_count if tokenizer is not None else None,
            "state_transfers": state_transfers,
            "state_payload_bytes": state_bytes,
            "memory_queries": memory_queries,
            "memory_hits": memory_hits,
            "memory_hit_rate": memory_hits / memory_queries if memory_queries else 0.0,
            "elapsed_sec": round(elapsed, 6),
            "tasks": rows,
        }

    text_reference = reports.get("text")
    if text_reference:
        for metrics in reports.values():
            baseline_tokens = text_reference["qwen_token_count"]
            current_tokens = metrics["qwen_token_count"]
            baseline_chars = text_reference["text_chars"]
            metrics["token_reduction_vs_text_pct"] = (
                round((1 - current_tokens / baseline_tokens) * 100, 2)
                if baseline_tokens and current_tokens is not None else None
            )
            metrics["text_char_reduction_vs_text_pct"] = round(
                (1 - metrics["text_chars"] / baseline_chars) * 100, 2
            ) if baseline_chars else None

    return {
        "experiment": "paired_adapter_ablation",
        "task_groups": ["retrieval", "collaboration"],
        "tasks_per_group_per_repeat": 5,
        "repeat": repeat,
        "corpus_size": len(corpus),
        "embedding_prepare_sec": round(embedding_prepare_sec, 6),
        "modes": reports,
        "limitations": [
            "The corpus and tasks are a deterministic mechanism fixture, not a broad QA benchmark.",
            "Elapsed time excludes model loading and has high variance at this scale.",
            "Text mode counts request and response frames with repeated prompt context; protocol counts request and response JSON envelopes.",
            "Memory-hit rate is a retrieval event rate, not evidence of accepted answer reuse.",
            "No Qwen3-8B generation or answer-quality evaluation is performed.",
        ],
    }


def save_report(report: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
