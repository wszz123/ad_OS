"""Persistent shared memory adapter with pluggable embedding search."""

from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import sqlite3
from typing import Any, Callable, Iterable, List, Optional, Protocol
from uuid import uuid4


Embedder = Callable[[Iterable[str]], Any]


@dataclass
class MemoryItem:
    memory_id: str
    source_agent: str
    created_at: str
    task_topic: str
    summary: str
    memory_type: str = "episodic"
    tags: List[str] = field(default_factory=list)
    evidence: str = ""
    embedding: Optional[List[float]] = None


@dataclass
class MemorySearchResult:
    item: MemoryItem
    score: float
    match_type: str


class MemoryBackend(Protocol):
    """Contract for local or external memory adapters used by graph nodes."""

    def put(self, item: MemoryItem) -> str: ...
    def get(self, memory_id: str) -> Optional[MemoryItem]: ...
    def get_metadata(self, memory_id: str) -> Optional[MemoryItem]: ...
    def items(self) -> List[MemoryItem]: ...
    def count(self) -> int: ...
    def close(self) -> None: ...
    def search(self, query: str, *, tags: Optional[Iterable[str]] = None,
               limit: int = 5) -> List[MemorySearchResult]: ...
    def search_vector(self, vector: Iterable[float], *, limit: int = 5,
                      tags: Optional[Iterable[str]] = None
                      ) -> List[MemorySearchResult]: ...


class MemoryAdapter:
    """SQLite-backed memory collection; embedding is optional and injected."""

    def __init__(self, path: str = ":memory:", embedder: Optional[Embedder] = None,
                 query_embedder: Optional[Embedder] = None, *, connection=None) -> None:
        self.path = path
        self.embedder = embedder
        self.query_embedder = query_embedder or embedder
        self._scan_counters = {"full_scan_calls": 0, "full_scan_rows": 0,
                               "vector_json_bytes_loaded": 0, "point_get_calls": 0,
                               "point_get_rows": 0, "point_vector_json_bytes_loaded": 0,
                               "filtered_scan_calls": 0, "filtered_scan_rows": 0,
                               "filtered_vector_json_bytes_loaded": 0,
                               "metadata_get_calls": 0, "metadata_get_rows": 0}
        self._owns_connection = connection is None
        self.connection = connection if connection is not None else sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                source_agent TEXT NOT NULL,
                created_at TEXT NOT NULL,
                task_topic TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                summary TEXT NOT NULL,
                evidence TEXT NOT NULL,
                embedding_json TEXT
            )
        """)
        self.connection.commit()
        self._write_counters = {"put_many_calls": 0, "transaction_scopes": 0, "savepoint_scopes": 0,
                                "commits": 0, "rolled_back_scopes": 0, "rows_accepted_in_scopes": 0}

    def close(self) -> None:
        if self._owns_connection:
            self.connection.close()

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        self._scan_counters["point_get_calls"] += 1
        row = self.connection.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is not None:
            self._scan_counters["point_get_rows"] += 1
            self._scan_counters["point_vector_json_bytes_loaded"] += len((row["embedding_json"] or "").encode("utf-8"))
        return self._from_row(row) if row else None

    def get_metadata(self, memory_id: str) -> Optional[MemoryItem]:
        """Project source metadata in SQL; never fetch or deserialize the vector."""
        self._scan_counters["metadata_get_calls"] += 1
        row = self.connection.execute(
            """SELECT memory_id, source_agent, created_at, task_topic, memory_type,
                      tags_json, summary, evidence, NULL AS embedding_json
               FROM memories WHERE memory_id = ?""", (memory_id,),
        ).fetchone()
        if row is not None:
            self._scan_counters["metadata_get_rows"] += 1
        return self._from_row(row) if row is not None else None

    def items(self) -> List[MemoryItem]:
        rows = self.connection.execute(
            "SELECT * FROM memories ORDER BY memory_id"
        ).fetchall()
        self._scan_counters["full_scan_calls"] += 1
        self._scan_counters["full_scan_rows"] += len(rows)
        self._scan_counters["vector_json_bytes_loaded"] += sum(
            len((row["embedding_json"] or "").encode("utf-8")) for row in rows)
        return [self._from_row(row) for row in rows]

    def scan_stats(self):
        """Rows/serialized vector bytes materialized, not physical disk I/O."""
        return dict(self._scan_counters)

    def count(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def search_vector(self, vector: Iterable[float], *, limit: int = 5,
                      tags: Optional[Iterable[str]] = None) -> List[MemorySearchResult]:
        """Use a received intermediate directly, without encoding text again."""
        return rank_vectors(self.items(), vector, limit=limit, tags=tags)

    def put(self, item: MemoryItem) -> str:
        return self.put_many([item])[0]

    def _prepare_row(self, item):
        item.memory_id = item.memory_id or str(uuid4())
        item.created_at = item.created_at or datetime.now(timezone.utc).isoformat()
        item.tags = item.tags or []
        if item.embedding is None and self.embedder is not None:
            item.embedding = self._vectors([self._search_text(item)])[0]
        return (item.memory_id, item.source_agent, item.created_at, item.task_topic,
                item.memory_type, json.dumps(item.tags, ensure_ascii=False), item.summary,
                item.evidence, json.dumps(item.embedding) if item.embedding is not None else None)

    @contextmanager
    def _write_scope(self):
        nested = self.connection.in_transaction
        savepoint = "memory_batch_" + uuid4().hex
        if nested:
            self.connection.execute(f"SAVEPOINT {savepoint}")
            self._write_counters["savepoint_scopes"] += 1
        else:
            self.connection.execute("BEGIN")
            self._write_counters["transaction_scopes"] += 1
        try:
            yield
            if nested:
                self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            else:
                self.connection.commit()
                self._write_counters["commits"] += 1
        except BaseException:
            if nested:
                self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            else:
                self.connection.rollback()
            self._write_counters["rolled_back_scopes"] += 1
            raise

    def put_many(self, items: Iterable[MemoryItem]) -> List[str]:
        """Atomic SQLite batch, with last-write-wins for repeated IDs.

        Prepare/encode before locking. A caller-owned transaction uses a nested
        savepoint instead of committing the caller's work. Database rows and tag
        trigger updates roll back together; mutations to caller items/embedding
        callbacks are outside the database transaction contract.
        """
        self._write_counters["put_many_calls"] += 1
        rows = [self._prepare_row(item) for item in items]
        if not rows:
            return []
        with self._write_scope():
            self.connection.executemany(
                """INSERT OR REPLACE INTO memories
                (memory_id, source_agent, created_at, task_topic, memory_type,
                 tags_json, summary, evidence, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)
        self._write_counters["rows_accepted_in_scopes"] += len(rows)
        return [row[0] for row in rows]

    def write_stats(self):
        """Adapter API transactions only, excluding schema/index setup and raw SQL."""
        return dict(self._write_counters)

    def search(
        self, query: str, *, tags: Optional[Iterable[str]] = None, limit: int = 5
    ) -> List[MemorySearchResult]:
        return rank_memories(self.items(), query, limit=limit, tags=tags,
                             query_embedder=self.query_embedder if self.embedder is not None else None)

    @staticmethod
    def _search_text(item: MemoryItem) -> str:
        return " ".join([item.task_topic, item.summary, item.evidence, *(item.tags or [])])

    def _vectors(self, texts: Iterable[str]) -> List[List[float]]:
        values = self.embedder(texts)
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [[float(value) for value in vector] for vector in values]

    @staticmethod
    def _cosine(left: List[float], right: List[float]) -> float:
        if len(left) != len(right):
            raise ValueError("embedding dimensions do not match")
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
            sum(value * value for value in right)
        )
        return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            memory_id=row["memory_id"], source_agent=row["source_agent"],
            created_at=row["created_at"], task_topic=row["task_topic"],
            memory_type=row["memory_type"], tags=json.loads(row["tags_json"]),
            summary=row["summary"], evidence=row["evidence"],
            embedding=json.loads(row["embedding_json"]) if row["embedding_json"] else None,
        )


def _candidates(items, tags):
    required = set(tags or [])
    return sorted((item for item in items if required.issubset(item.tags or [])),
                  key=lambda item: item.memory_id)


def rank_vectors(items, vector, *, limit=5, tags=None):
    """Common exact cosine policy: storage backends cannot change ranking rules."""
    if limit < 1:
        raise ValueError("limit must be positive")
    vector = [float(value) for value in vector]
    results = [MemorySearchResult(item, MemoryAdapter._cosine(vector, item.embedding), "semantic")
               for item in _candidates(items, tags) if item.embedding is not None]
    return sorted(results, key=lambda result: result.score, reverse=True)[:limit]


def rank_memories(items, query, *, limit=5, tags=None, query_embedder=None):
    """Same keyword/tag and optional query-embedding policy for all adapters."""
    if limit < 1:
        raise ValueError("limit must be positive")
    items = _candidates(items, tags)
    query_terms = set(query.lower().split())
    results = [MemorySearchResult(item, float(len(query_terms & set(
        MemoryAdapter._search_text(item).lower().split()))), "keyword") for item in items]
    if query_embedder is not None and items:
        vectors = query_embedder([query])
        vector = vectors.tolist()[0] if hasattr(vectors, "tolist") else vectors[0]
        for index, result in enumerate(results):
            if result.item.embedding is not None:
                results[index] = MemorySearchResult(result.item, MemoryAdapter._cosine(vector, result.item.embedding), "semantic")
    results.sort(key=lambda result: result.score, reverse=True)
    return [result for result in results[:limit] if result.match_type == "semantic" or result.score > 0]
