"""External LangGraph Store adapter, backend factory and operation profiling."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import time
from uuid import uuid4

from .memory import MemoryAdapter, MemoryItem, rank_memories, rank_vectors


class LangGraphMemoryAdapter:
    """MemoryBackend over an injected synchronous LangGraph BaseStore.

    Default storage is LangGraph InMemoryStore. Exact cosine ranking stays in the
    adapter so received vectors need no text re-encoding. This is not native ANN
    search. The caller owns the store lifecycle; closing this adapter does not
    delete other namespaces or shut down an injected shared store.
    """

    def __init__(self, store=None, *, namespace=("latentmas", "memories"),
                 embedder=None, query_embedder=None, page_size=256):
        if not namespace or not all(isinstance(part, str) and part for part in namespace):
            raise ValueError("namespace must be a nonempty tuple of nonempty strings")
        if not isinstance(namespace, tuple) or page_size < 1:
            raise ValueError("use a tuple namespace and positive page size")
        if store is None:
            from langgraph.store.memory import InMemoryStore

            store = InMemoryStore()
        self.store = store
        self.namespace = namespace
        self.embedder = embedder
        self.query_embedder = query_embedder or embedder
        self.page_size = page_size
        self.closed = False

    def _open(self):
        if self.closed:
            raise RuntimeError("memory adapter is closed")

    def put(self, item):
        self._open()
        item.memory_id = item.memory_id or str(uuid4())
        item.created_at = item.created_at or datetime.now(timezone.utc).isoformat()
        item.tags = item.tags or []
        if item.embedding is None and self.embedder is not None:
            vectors = self.embedder([MemoryAdapter._search_text(item)])
            vector = vectors.tolist()[0] if hasattr(vectors, "tolist") else vectors[0]
            item.embedding = [float(value) for value in vector]
        value = json.loads(json.dumps(asdict(item), ensure_ascii=False))
        self.store.put(self.namespace, item.memory_id, value, index=False)
        return item.memory_id

    def get(self, memory_id):
        self._open()
        item = self.store.get(self.namespace, memory_id)
        # Detach values: some stores expose their stored dict by reference.
        return MemoryItem(**json.loads(json.dumps(item.value))) if item is not None else None

    def get_metadata(self, memory_id):
        # BaseStore has no field projection contract: this strips an already-read
        # vector and must not be described as reducing provider I/O.
        item = self.get(memory_id)
        if item is not None:
            item.embedding = None
        return item

    def items(self):
        self._open()
        values, offset = [], 0
        while True:
            page = self.store.search(self.namespace, limit=self.page_size, offset=offset)
            for item in page:
                # Store.search is prefix-based; exclude children explicitly.
                if tuple(item.namespace) == self.namespace:
                    values.append(MemoryItem(**json.loads(json.dumps(item.value))))
            if len(page) < self.page_size:
                break
            offset += len(page)
        return sorted(values, key=lambda item: item.memory_id)

    def count(self):
        return len(self.items())

    def search(self, query, *, tags=None, limit=5):
        return rank_memories(self.items(), query, tags=tags, limit=limit,
                             query_embedder=self.query_embedder if self.embedder is not None else None)

    def search_vector(self, vector, *, tags=None, limit=5):
        return rank_vectors(self.items(), vector, tags=tags, limit=limit)

    def close(self):
        self.closed = True


class MeasuredMemoryBackend:
    """Profile public adapter operations; model/embedding calls remain separate."""

    def __init__(self, backend):
        self.backend = backend
        self.operations = {}

    def _call(self, operation, *args, **kwargs):
        counter = self.operations.setdefault(operation, {"calls": 0, "errors": 0, "elapsed_sec": 0.0})
        counter["calls"] += 1
        started = time.perf_counter()
        try:
            return getattr(self.backend, operation)(*args, **kwargs)
        except Exception:
            counter["errors"] += 1
            raise
        finally:
            counter["elapsed_sec"] += time.perf_counter() - started

    def put(self, item): return self._call("put", item)
    def put_many(self, items): return self._call("put_many", items)
    def get(self, memory_id): return self._call("get", memory_id)
    def get_metadata(self, memory_id): return self._call("get_metadata", memory_id)
    def items(self): return self._call("items")
    def count(self): return self._call("count")
    def search(self, query, **kwargs): return self._call("search", query, **kwargs)
    def search_vector(self, vector, **kwargs): return self._call("search_vector", vector, **kwargs)
    def close(self): self.backend.close()

    def stats(self):
        return json.loads(json.dumps(self.operations))


def create_memory_backend(name, *, path=None, namespace=("latentmas", "memories"), **kwargs):
    if name == "sqlite":
        return MemoryAdapter(str(path) if path is not None else ":memory:", **kwargs)
    if name == "langgraph":
        return LangGraphMemoryAdapter(namespace=namespace, **kwargs)
    raise ValueError(f"unknown memory backend: {name}")
