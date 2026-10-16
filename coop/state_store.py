"""Durable binary state exchange through a namespaced SQLite BLOB store."""

from collections.abc import MutableMapping
from dataclasses import asdict
import json
import sqlite3
from threading import RLock

from .state_exchange import StateExchange, StatePacket


class SQLiteStateStore(MutableMapping):
    """Mapping-compatible packet backend usable by independent Linux processes.

    State rows are immutable: the same key/content is an idempotent write;
    conflicting reuse of a state ID is rejected rather than corrupting a saved
    checkpoint's reference. Connections must be closed by the owning process.
    """

    def __init__(self, path: str, *, namespace: str):
        if not namespace:
            raise ValueError("a state namespace is required")
        self.namespace = namespace
        self.connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.lock = RLock()
        self.exchange = StateExchange()
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS state_packets (
            namespace TEXT NOT NULL, state_id TEXT NOT NULL,
            metadata TEXT NOT NULL, payload BLOB NOT NULL,
            PRIMARY KEY (namespace, state_id)
        )""")
        self.connection.commit()

    def __getitem__(self, state_id: str) -> StatePacket:
        with self.lock:
            row = self.connection.execute(
                "SELECT metadata, payload FROM state_packets WHERE namespace=? AND state_id=?",
                (self.namespace, state_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"state {state_id!r} missing in namespace {self.namespace!r}")
        metadata = json.loads(row[0])
        metadata["shape"] = tuple(metadata["shape"])
        packet = StatePacket(**metadata, payload=bytes(row[1]))
        if packet.state_id != state_id:
            raise ValueError("stored state ID does not match lookup key")
        self.exchange.decode(packet)
        return packet

    def __setitem__(self, state_id: str, packet: StatePacket):
        if state_id != packet.state_id:
            raise ValueError("state key must equal packet ID")
        self.exchange.decode(packet)
        metadata = asdict(packet)
        metadata.pop("payload")
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO state_packets VALUES (?, ?, ?, ?)",
                (self.namespace, state_id, encoded, packet.payload),
            )
            stored = self.connection.execute(
                "SELECT metadata, payload FROM state_packets WHERE namespace=? AND state_id=?",
                (self.namespace, state_id),
            ).fetchone()
            if stored != (encoded, packet.payload):
                raise ValueError("conflicting write to an immutable state ID")

    def __delitem__(self, state_id):
        with self.lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM state_packets WHERE namespace=? AND state_id=?",
                (self.namespace, state_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(state_id)

    def __iter__(self):
        with self.lock:
            rows = self.connection.execute(
                "SELECT state_id FROM state_packets WHERE namespace=? ORDER BY state_id",
                (self.namespace,),
            ).fetchall()
        return iter(row[0] for row in rows)

    def __len__(self):
        with self.lock:
            return self.connection.execute(
                "SELECT COUNT(*) FROM state_packets WHERE namespace=?", (self.namespace,),
            ).fetchone()[0]

    def close(self):
        with self.lock:
            self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
