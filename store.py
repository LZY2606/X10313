"""SQLite-backed immutable event store.

Identity model:
  inputs   - original submitted inputs, content-addressed, never overwritten
  versions - revisions (source/params) linked by parent_id
  events   - derived values and human decisions, each carrying
             algo_version + parameter fingerprint + direct source ids
Stale events remain viewable; they never overwrite history.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Callable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY, v TEXT
);
CREATE TABLE IF NOT EXISTS inputs (
    id TEXT PRIMARY KEY, content TEXT NOT NULL, sha TEXT NOT NULL, seq INTEGER
);
CREATE TABLE IF NOT EXISTS versions (
    id TEXT PRIMARY KEY, parent_id TEXT, kind TEXT NOT NULL,
    label TEXT NOT NULL, payload TEXT NOT NULL, seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, seq INTEGER NOT NULL, kind TEXT NOT NULL,
    status TEXT NOT NULL, algo_version TEXT, params_fp TEXT,
    depends_on TEXT NOT NULL, payload TEXT NOT NULL, clock REAL NOT NULL
);
"""


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def params_fingerprint(algo_version: str, params: dict) -> str:
    blob = json.dumps({"algo_version": algo_version, "params": params},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class Store:
    def __init__(self, path: str = "gsb.sqlite3",
                 now: Optional[Callable[[], float]] = None):
        self.path = path
        self.now = now or time.time
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def reset(self):
        self.conn.executescript(
            "DELETE FROM events; DELETE FROM versions; DELETE FROM inputs;")
        self.conn.commit()

    # ---- low level --------------------------------------------------------
    def _next_seq(self, table: str) -> int:
        row = self.conn.execute(f"SELECT COALESCE(MAX(seq),0)+1 AS s FROM {table}").fetchone()
        return int(row["s"])

    def add_input(self, content: str) -> str:
        input_id = "in-" + _sha(content)
        exists = self.conn.execute(
            "SELECT id FROM inputs WHERE id=?", (input_id,)).fetchone()
        if exists is None:
            self.conn.execute(
                "INSERT INTO inputs(id,content,sha,seq) VALUES(?,?,?,?)",
                (input_id, content, _sha(content), self._next_seq("inputs")))
            self.conn.commit()
        return input_id

    def add_version(self, parent_id: Optional[str], kind: str, label: str,
                    payload: dict) -> str:
        seq = self._next_seq("versions")
        version_id = f"v{seq:03d}"
        self.conn.execute(
            "INSERT INTO versions(id,parent_id,kind,label,payload,seq) "
            "VALUES(?,?,?,?,?,?)",
            (version_id, parent_id, kind, label,
             json.dumps(payload, ensure_ascii=False, sort_keys=True), seq))
        self.conn.commit()
        return version_id

    def record_event(self, kind: str, payload: dict,
                     algo_version: Optional[str] = None,
                     params: Optional[dict] = None,
                     depends_on: Optional[list[str]] = None,
                     status: str = "current") -> str:
        seq = self._next_seq("events")
        event_id = f"e{seq:03d}"
        fp = params_fingerprint(algo_version or "", params or {})
        self.conn.execute(
            "INSERT INTO events(id,seq,kind,status,algo_version,params_fp,"
            "depends_on,payload,clock) VALUES(?,?,?,?,?,?,?,?,?)",
            (event_id, seq, kind, status, algo_version, fp,
             json.dumps(depends_on or [], ensure_ascii=False),
             json.dumps(payload, ensure_ascii=False, sort_keys=True),
             self.now()))
        self.conn.commit()
        return event_id

    # ---- queries ----------------------------------------------------------
    def list_versions(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM versions ORDER BY seq").fetchall()
        return [_version_row(r) for r in rows]

    def list_events(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY seq").fetchall()
        return [_event_row(r) for r in rows]

    def get_version(self, version_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM versions WHERE id=?", (version_id,)).fetchone()
        return _version_row(row) if row else None

    def get_event(self, event_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return _event_row(row) if row else None

    def supersede(self, version_id: str) -> list[str]:
        """Mark current events directly depending on this version stale."""
        affected: list[str] = []
        for event in self.list_events():
            if (event["status"] == "current"
                    and version_id in event["depends_on"]):
                affected.append(event["id"])
        for event_id in affected:
            self.conn.execute(
                "UPDATE events SET status='stale' WHERE id=?", (event_id,))
        self.conn.commit()
        return affected

    # ---- decisions --------------------------------------------------------
    def decide(self, target: str, verdict: str, note: str = "",
               params: Optional[dict] = None) -> str:
        return self.record_event(
            "decision",
            {"target": target, "verdict": verdict, "note": note,
             "params": params or {}},
            depends_on=[target])

    def undo(self, decision_id: str) -> str:
        original = self.get_event(decision_id)
        if original is None or original["kind"] != "decision":
            raise ValueError(f"{decision_id} is not a decision event")
        return self.record_event(
            "undo", {"undoes": decision_id}, depends_on=[decision_id])

    def active_decisions(self, as_of_seq: Optional[int] = None) -> list[dict]:
        """Decisions that are not reverted by an undo at/before as_of_seq."""
        undone: set[str] = set()
        out: list[dict] = []
        for event in self.list_events():
            if as_of_seq is not None and event["seq"] > as_of_seq:
                break
            if event["kind"] == "undo":
                undone.add(event["payload"]["undoes"])
            elif event["kind"] == "decision" and event["id"] not in undone:
                out.append(event)
        return out

    def conflicts(self) -> list[dict]:
        groups: dict[str, list[dict]] = {}
        for event in self.active_decisions():
            target = event["payload"]["target"]
            groups.setdefault(target, []).append(event)
        result = []
        for target, decisions in groups.items():
            verdicts = {d["payload"]["verdict"] for d in decisions}
            if len(verdicts) > 1:
                result.append({"target": target,
                               "decision_ids": [d["id"] for d in decisions],
                               "verdicts": sorted(verdicts)})
        return result

    def event_view(self, event_id: str) -> dict:
        """Reconstruct the view as of an event: its dependency versions and
        the human decisions visible at that point in history."""
        event = self.get_event(event_id)
        if event is None:
            raise KeyError(event_id)
        versions = [self.get_version(vid)
                    for vid in event["depends_on"]
                    if vid and vid.startswith("v")]
        upstream_events = [self.get_event(eid)
                           for eid in event["depends_on"]
                           if eid and eid.startswith("e")]
        return {"event": event,
                "dependency_versions": [v for v in versions if v],
                "upstream_events": [e for e in upstream_events if e],
                "decisions_visible": self.active_decisions(event["seq"]),
                "historical": event["seq"]
                < self.conn.execute("SELECT MAX(seq) AS s FROM events")
                    .fetchone()["s"]}

    # ---- export / import --------------------------------------------------
    def export_db(self) -> dict:
        def rows(table):
            return [dict(r) for r in self.conn.execute(
                f"SELECT * FROM {table} ORDER BY seq").fetchall()]
        data = {"format": "gsb/1",
                "inputs": rows("inputs"),
                "versions": rows("versions"),
                "events": rows("events")}
        body = json.dumps(data, sort_keys=True, ensure_ascii=False)
        data["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return data

    def import_db(self, data: dict, verify: bool = True) -> dict:
        data = dict(data)
        digest = data.pop("sha256", None)
        body = json.dumps(data, sort_keys=True, ensure_ascii=False)
        if verify:
            if digest != hashlib.sha256(body.encode("utf-8")).hexdigest():
                raise ValueError("export checksum mismatch")
            for table in ("inputs", "versions", "events"):
                ids = [r["id"] for r in data[table]]
                if len(ids) != len(set(ids)):
                    raise ValueError(f"duplicate ids in {table}")
        self.reset()
        for row in data["inputs"]:
            self.conn.execute(
                "INSERT INTO inputs(id,content,sha,seq) VALUES(?,?,?,?)",
                (row["id"], row["content"], row["sha"], row["seq"]))
        for row in data["versions"]:
            self.conn.execute(
                "INSERT INTO versions(id,parent_id,kind,label,payload,seq) "
                "VALUES(?,?,?,?,?,?)",
                (row["id"], row["parent_id"], row["kind"], row["label"],
                 row["payload"], row["seq"]))
        for row in data["events"]:
            self.conn.execute(
                "INSERT INTO events(id,seq,kind,status,algo_version,"
                "params_fp,depends_on,payload,clock) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (row["id"], row["seq"], row["kind"], row["status"],
                 row["algo_version"], row["params_fp"], row["depends_on"],
                 row["payload"], row["clock"]))
        self.conn.commit()
        return {"inputs": len(data["inputs"]),
                "versions": len(data["versions"]),
                "events": len(data["events"])}


def _event_row(row) -> dict:
    return {"id": row["id"], "seq": row["seq"], "kind": row["kind"],
            "status": row["status"], "algo_version": row["algo_version"],
            "params_fp": row["params_fp"],
            "depends_on": json.loads(row["depends_on"]),
            "payload": json.loads(row["payload"]),
            "clock": row["clock"]}


def _version_row(row) -> dict:
    return {"id": row["id"], "parent_id": row["parent_id"],
            "kind": row["kind"], "label": row["label"],
            "payload": json.loads(row["payload"]), "seq": row["seq"]}


def replace_database(path: str, data: dict,
                     now: Optional[Callable[[], float]] = None) -> Store:
    """Delete the local database file and rebuild it from an export."""
    if path != ":memory:" and os.path.exists(path):
        os.remove(path)
    rebuilt = Store(path, now=now)
    rebuilt.import_db(data)
    return rebuilt
