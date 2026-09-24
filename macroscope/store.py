"""Append-only SQLite event store + immutable derived artifacts.

Events carry monotonically increasing seq; derived artifacts record algorithm
version, parameter fingerprint and direct sources so any value can be replayed
and staleness can be judged locally. Nothing is ever overwritten.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Callable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY,
  id TEXT UNIQUE NOT NULL,
  project TEXT NOT NULL,
  type TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS derived(
  id TEXT PRIMARY KEY,
  project TEXT NOT NULL,
  kind TEXT NOT NULL,
  algo_version TEXT NOT NULL,
  params_fingerprint TEXT NOT NULL,
  sources TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_seq INTEGER NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_id(prefix: str, obj) -> str:
    digest = hashlib.sha1(canonical(obj).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


class EventStore:
    def __init__(self, path: str, clock: Callable[[], float]):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.clock = clock

    def close(self) -> None:
        self.conn.close()

    def _next_seq(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(seq),0)+1 AS s FROM events").fetchone()
        return row["s"]

    def append(self, project: str, type_: str, payload: dict,
               event_id: Optional[str] = None, seq: Optional[int] = None,
               created_at: Optional[float] = None) -> dict:
        if seq is None:
            seq = self._next_seq()
        if event_id is None:
            event_id = content_id("ev", {"seq": seq, "type": type_, "payload": payload})
        if created_at is None:
            created_at = self.clock()
        self.conn.execute(
            "INSERT INTO events(seq,id,project,type,payload,created_at) "
            "VALUES(?,?,?,?,?,?)",
            (seq, event_id, project, type_, canonical(payload), created_at))
        self.conn.commit()
        return {"seq": seq, "id": event_id, "project": project, "type": type_,
                "payload": payload, "created_at": created_at}

    def events(self, project: str, upto_seq: Optional[int] = None) -> List[dict]:
        q = "SELECT * FROM events WHERE project=?"
        args: list = [project]
        if upto_seq is not None:
            q += " AND seq<=?"
            args.append(upto_seq)
        q += " ORDER BY seq"
        return [self._row_to_event(r) for r in self.conn.execute(q, args)]

    @staticmethod
    def _row_to_event(r) -> dict:
        return {"seq": r["seq"], "id": r["id"], "project": r["project"],
                "type": r["type"], "payload": json.loads(r["payload"]),
                "created_at": r["created_at"]}

    def put_derived(self, project: str, kind: str, artifact_id: str,
                    algo_version: str, params_fingerprint: str,
                    sources: dict, payload: dict, created_seq: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO derived"
            "(id,project,kind,algo_version,params_fingerprint,sources,payload,"
            " created_seq,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (artifact_id, project, kind, algo_version, params_fingerprint,
             canonical(sources), canonical(payload), created_seq, self.clock()))
        self.conn.commit()

    def get_derived(self, artifact_id: str) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM derived WHERE id=?",
                              (artifact_id,)).fetchone()
        return self._row_to_derived(r) if r else None

    def derived_for(self, project: str, kind: Optional[str] = None,
                    upto_seq: Optional[int] = None) -> List[dict]:
        q = "SELECT * FROM derived WHERE project=?"
        args: list = [project]
        if kind:
            q += " AND kind=?"
            args.append(kind)
        if upto_seq is not None:
            q += " AND created_seq<=?"
            args.append(upto_seq)
        q += " ORDER BY created_seq, id"
        return [self._row_to_derived(r) for r in self.conn.execute(q, args)]

    @staticmethod
    def _row_to_derived(r) -> dict:
        return {"id": r["id"], "project": r["project"], "kind": r["kind"],
                "algo_version": r["algo_version"],
                "params_fingerprint": r["params_fingerprint"],
                "sources": json.loads(r["sources"]),
                "payload": json.loads(r["payload"]),
                "created_seq": r["created_seq"], "created_at": r["created_at"]}

    def state_hash(self, project: str) -> str:
        events = self.events(project)
        derived = self.derived_for(project)
        return hashlib.sha1(canonical({"events": events,
                                       "derived": derived}).encode()).hexdigest()
