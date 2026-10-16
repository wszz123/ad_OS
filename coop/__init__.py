"""Competition prototype adapters."""

from importlib import import_module

_EXPORTS = {
    "LocalQwenEmbedding": "embedding", "last_token_pool": "embedding",
    "MemoryAdapter": "memory", "MemoryItem": "memory", "MemorySearchResult": "memory",
    "StateExchange": "state_exchange", "StatePacket": "state_exchange",
    "SQLiteStateStore": "state_store",
}


def __getattr__(name):
    # Keep diagnostics/packaging usable before torch/numpy are installed.
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "LocalQwenEmbedding",
    "MemoryAdapter",
    "MemoryItem",
    "MemorySearchResult",
    "StateExchange",
    "StatePacket",
    "SQLiteStateStore",
    "last_token_pool",
]
