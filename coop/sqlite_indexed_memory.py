"""SQLite tag-index pushdown with the same exact scoring as the reference."""

import time

from .memory import MemoryAdapter, rank_memories, rank_vectors


class IndexedSQLiteMemoryAdapter(MemoryAdapter):
    """Materialize only candidates with all required tags, before decoding vectors.

    The derived tag table is maintained by SQLite triggers, including writes from
    the reference adapter on the same file. Source records stay in the original
    schema; upgrades rebuild the derived index atomically without loading vectors.
    This is a metadata index, not an approximate vector index.
    """

    def __init__(self, *args, **kwargs):
        started = time.perf_counter()
        super().__init__(*args, **kwargs)
        try:
            with self.connection:
                self.connection.execute("""CREATE TABLE IF NOT EXISTS coop_memory_tags (
                    tag TEXT NOT NULL, memory_id TEXT NOT NULL,
                    PRIMARY KEY (tag, memory_id)
                )""")
                for event, clear_id in (("INSERT", "NEW.memory_id"),
                                        ("UPDATE OF tags_json, memory_id", "OLD.memory_id")):
                    name = "insert" if event == "INSERT" else "update"
                    self.connection.execute(f"""CREATE TRIGGER IF NOT EXISTS coop_tags_{name}
                        AFTER {event} ON memories BEGIN
                            DELETE FROM coop_memory_tags WHERE memory_id = {clear_id};
                            INSERT OR IGNORE INTO coop_memory_tags(tag, memory_id)
                            SELECT value, NEW.memory_id FROM json_each(NEW.tags_json) WHERE type='text';
                        END""")
                self.connection.execute("""CREATE TRIGGER IF NOT EXISTS coop_tags_delete
                    AFTER DELETE ON memories BEGIN
                        DELETE FROM coop_memory_tags WHERE memory_id = OLD.memory_id;
                    END""")
                self.connection.execute("DELETE FROM coop_memory_tags")
                self.connection.execute("""INSERT OR IGNORE INTO coop_memory_tags(tag, memory_id)
                    SELECT j.value, m.memory_id FROM memories m, json_each(m.tags_json) j WHERE j.type='text'""")
        except Exception:
            self.close()
            raise
        self.index_setup_sec = time.perf_counter() - started

    def _tagged_items(self, tags):
        values = list(tags or [])
        if any(not isinstance(tag, str) for tag in values):
            raise ValueError("tags must be strings")
        required = sorted(set(values))
        if not required:
            return self.items()
        placeholders = ",".join("?" for _ in required)
        # Only placeholder count is interpolated; tag values are bound parameters.
        rows = self.connection.execute(f"""
            SELECT m.* FROM memories m JOIN (
                SELECT memory_id FROM coop_memory_tags
                WHERE tag IN ({placeholders}) GROUP BY memory_id
                HAVING COUNT(DISTINCT tag) = ?
            ) matched ON matched.memory_id = m.memory_id ORDER BY m.memory_id
        """, (*required, len(required))).fetchall()
        self._scan_counters["filtered_scan_calls"] += 1
        self._scan_counters["filtered_scan_rows"] += len(rows)
        self._scan_counters["filtered_vector_json_bytes_loaded"] += sum(
            len((row["embedding_json"] or "").encode("utf-8")) for row in rows)
        return [self._from_row(row) for row in rows]

    def search(self, query, *, tags=None, limit=5):
        required = tuple(tags or ())
        return rank_memories(self._tagged_items(required), query, tags=required, limit=limit,
                             query_embedder=self.query_embedder if self.embedder is not None else None)

    def search_vector(self, vector, *, tags=None, limit=5):
        required = tuple(tags or ())
        return rank_vectors(self._tagged_items(required), vector, tags=required, limit=limit)
