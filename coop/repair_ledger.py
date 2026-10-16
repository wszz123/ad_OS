"""Durable, at-most-budget reservations for repair across process restarts."""

from datetime import datetime, timezone
import json
import sqlite3
from uuid import uuid4


class RepairLedger:
    """Commit reservations BEFORE dispatch; never replay an uncertain attempt.

    A process can die after reservation or after generation but before finish.
    Both cases spend the reservation. This bounds attempts, not exactly-once
    execution, and does not infer token usage for unfinished calls.
    """

    def __init__(self, path=":memory:"):
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS repair_sessions (
            task_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, max_repairs INTEGER NOT NULL
        )""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS repair_attempts (
            task_id TEXT NOT NULL, attempt INTEGER NOT NULL, remaining INTEGER NOT NULL,
            status TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY (task_id, attempt)
        )""")
        self.connection.commit()

    def attempts(self, task_id):
        rows = self.connection.execute(
            "SELECT attempt, remaining, status, payload FROM repair_attempts WHERE task_id=? ORDER BY attempt",
            (task_id,),
        ).fetchall()
        # Canonical database columns cannot be overridden by the payload.
        return [{**json.loads(payload), "attempt": number, "remaining": remaining, "status": status}
                for number, remaining, status, payload in rows]

    def _session(self, task_id, max_repairs, fingerprint):
        row = self.connection.execute(
            "SELECT fingerprint, max_repairs FROM repair_sessions WHERE task_id=?", (task_id,),
        ).fetchone()
        if row is None:
            if self.attempts(task_id):
                raise ValueError("legacy ledger has attempts without a session manifest; cannot safely resume")
            self.connection.execute("INSERT INTO repair_sessions VALUES (?, ?, ?)",
                                    (task_id, fingerprint, max_repairs))
        elif row != (fingerprint, max_repairs):
            raise ValueError("repair ledger policy fingerprint mismatch")
        previous = self.attempts(task_id)
        if any(record["attempt"] != index or record["remaining"] != max_repairs - index
               or record["status"] not in ("started", "rejected", "accepted")
               for index, record in enumerate(previous)) or len(previous) > max_repairs + 1:
            raise ValueError("invalid or discontinuous ledger history")
        return previous

    def begin(self, task_id, *, max_repairs, prompt_fingerprint):
        if not task_id or isinstance(max_repairs, bool) or not isinstance(max_repairs, int) or not 1 <= max_repairs <= 5:
            raise ValueError("task_id and a repair limit in [1,5] are required")
        with self.connection:
            previous = self._session(task_id, max_repairs, prompt_fingerprint)
        return previous, max_repairs - max(len(previous) - 1, 0)

    def reserve(self, task_id, *, max_repairs, prompt_fingerprint, payload):
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            previous = self._session(task_id, max_repairs, prompt_fingerprint)
            if any(record["status"] == "accepted" for record in previous):
                raise ValueError("accepted plan must be replayed, not allocated another attempt")
            number = len(previous)
            if number > max_repairs:
                raise ValueError("repair reservation budget exhausted")
            token = uuid4().hex
            data = {**json.loads(encoded), "reservation_token": token}
            record = {**data, "attempt": number, "remaining": max_repairs - number, "status": "started"}
            self.connection.execute("INSERT INTO repair_attempts VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, number, record["remaining"], "started", json.dumps(data, ensure_ascii=False, sort_keys=True),
                 datetime.now(timezone.utc).isoformat()))
            self.connection.commit()
            return record
        except BaseException:
            self.connection.rollback()
            raise

    def finish(self, task_id, reservation, *, payload, accepted):
        data = {**payload, "reservation_token": reservation["reservation_token"]}
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self.connection:
            stored = self.connection.execute(
                "SELECT status, payload FROM repair_attempts WHERE task_id=? AND attempt=?",
                (task_id, reservation["attempt"]),
            ).fetchone()
            if stored is None or stored[0] != "started" or json.loads(stored[1]).get("reservation_token") != reservation["reservation_token"]:
                raise ValueError("attempt is not the caller's unfinished reservation")
            self.connection.execute("UPDATE repair_attempts SET status=?, payload=? WHERE task_id=? AND attempt=? AND status='started'",
                ("accepted" if accepted else "rejected", encoded, task_id, reservation["attempt"]))

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
