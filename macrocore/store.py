"""SQLite event-sourced store for the macro-scope mirror.

- raw inputs are immutable once submitted (insert-only)
- revisions are expressed as new version rows with an explicit parent
- derived values carry algo version, parameter fingerprint and direct sources
- every human action is an append-only event; undo is itself an event
- historical views are replayed from events, never overwritten by latest state
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple


def fixed_clock(seed_ts: str = "2026-09-25T00:00:00+09:00") -> Callable[[], str]:
    """Virtual time: deterministic clock for tests and replay."""
    state = {"i": 0}

    def now() -> str:
        # fixed seed + monotonic integer seconds so every event has a stable ts
        state["i"] += 1
        base = datetime.fromisoformat(seed_ts).timestamp()
        return datetime.fromtimestamp(base + state["i"], timezone.utc).isoformat()

    return now


def fingerprint(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS raw_inputs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  submitted_ts TEXT NOT NULL,
  submitted_event TEXT NOT NULL,
  representation TEXT NOT NULL DEFAULT 'text'
);
CREATE TABLE IF NOT EXISTS versions (
  id TEXT PRIMARY KEY,
  lineage TEXT NOT NULL,
  parent_id TEXT,
  kind TEXT NOT NULL,
  label TEXT NOT NULL,
  raw_id TEXT NOT NULL,
  params TEXT NOT NULL,
  created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS derived (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  subject_id TEXT NOT NULL,
  subject_version TEXT,
  algo_version TEXT NOT NULL,
  params_fp TEXT NOT NULL,
  params_json TEXT NOT NULL,
  source_ids TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT UNIQUE NOT NULL,
  type TEXT NOT NULL,
  payload TEXT NOT NULL,
  ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS case_pack (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  loaded_ts TEXT NOT NULL
);
"""


class OverwriteError(Exception):
    pass


class ConflictError(Exception):
    pass


class Store:
    def __init__(self, path: str, clock: Optional[Callable[[], str]] = None):
        self.path = path
        self.clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------- identity
    def _counter(self, kind: str) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (kind,)).fetchone()
        n = int(row["value"]) + 1 if row else 1
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (kind, str(n)),
        )
        return n

    def _new_id(self, kind: str, prefix: str) -> str:
        return f"{prefix}_{self._counter(kind):04d}"

    def _seq(self) -> int:
        row = self.conn.execute("SELECT MAX(seq) AS s FROM events").fetchone()
        return int(row["s"] or 0)

    # --------------------------------------------------------------- events
    def add_event(self, etype: str, payload: dict) -> dict:
        eid = self._new_id("evt", "evt")
        ts = self.clock()
        cur = self.conn.execute(
            "INSERT INTO events(id,type,payload,ts) VALUES(?,?,?,?)",
            (eid, etype, json.dumps(payload, ensure_ascii=False), ts),
        )
        self.conn.commit()
        return {"seq": int(cur.lastrowid), "id": eid, "type": etype,
                "payload": payload, "ts": ts}

    def events(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT seq,id,type,payload,ts FROM events ORDER BY seq").fetchall()
        return [{"seq": r["seq"], "id": r["id"], "type": r["type"],
                 "payload": json.loads(r["payload"]), "ts": r["ts"]} for r in rows]

    def event_by_id(self, eid: str) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
        if not r:
            return None
        return {"seq": r["seq"], "id": r["id"], "type": r["type"],
                "payload": json.loads(r["payload"]), "ts": r["ts"]}

    # ------------------------------------------------------------------ raw
    def add_raw(self, kind: str, content: str) -> dict:
        """Submit immutable raw input; identical resubmission is idempotent."""
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        existing = self.conn.execute(
            "SELECT id FROM raw_inputs WHERE kind=? AND content_hash=?",
            (kind, h)).fetchone()
        if existing:
            return {"id": existing["id"], "created": False, "immutable": True}
        rid = self._new_id("raw", "raw")
        evt = self.add_event("raw-imported", {"raw_id": rid, "kind": kind, "sha256": h})
        self.conn.execute(
            "INSERT INTO raw_inputs(id,kind,content,content_hash,submitted_ts,"
            "submitted_event) VALUES(?,?,?,?,?,?)",
            (rid, kind, content, h, self.clock(), evt["id"]),
        )
        self.conn.commit()
        return {"id": rid, "created": True, "event": evt["id"], "immutable": True}

    def get_raw(self, rid: str) -> dict:
        r = self.conn.execute("SELECT * FROM raw_inputs WHERE id=?", (rid,)).fetchone()
        if not r:
            raise KeyError(f"unknown raw input {rid}")
        return dict(r)

    def list_raw(self) -> List[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT id,kind,content_hash,submitted_ts FROM raw_inputs ORDER BY id")]

    # -------------------------------------------------------------- versions
    def add_version(self, parent_id: Optional[str], kind: str, label: str,
                    raw_id: str, params: Optional[dict] = None) -> dict:
        ancestors: List[str] = []
        if parent_id:
            prow = self.conn.execute(
                "SELECT lineage FROM versions WHERE id=?", (parent_id,)).fetchone()
            if not prow:
                raise KeyError(f"unknown parent version {parent_id}")
            lineage = prow["lineage"]
            ancestors = [r["id"] for r in self.conn.execute(
                "SELECT id FROM versions WHERE lineage=? ORDER BY created_seq",
                (lineage,)).fetchall()]
        else:
            lineage = self._new_id("lin", "lin")
        vid = self._new_id("ver", "ver")
        self.conn.execute(
            "INSERT INTO versions(id,lineage,parent_id,kind,label,raw_id,params,created_seq)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (vid, lineage, parent_id, kind, label, raw_id,
             json.dumps(params or {}, ensure_ascii=False), self._seq()),
        )
        self.conn.commit()
        evt = self.add_event("version-added", {
            "version_id": vid, "parent_id": parent_id, "lineage": lineage,
            "kind": kind, "label": label, "raw_id": raw_id, "params": params or {},
        })
        staled: List[str] = []
        if ancestors:
            affected = set(ancestors)
            staled = [r["id"] for r in self.conn.execute(
                "SELECT id,subject_version FROM derived").fetchall()
                if r["subject_version"] in affected]
            if staled:
                self.add_event("derived-staled", {
                    "derived_ids": staled, "by_version": vid,
                    "reason": "direct source superseded by new version",
                })
        return {"id": vid, "lineage": lineage, "event": evt["id"], "staled": staled}

    def get_version(self, vid: str) -> dict:
        r = self.conn.execute("SELECT * FROM versions WHERE id=?", (vid,)).fetchone()
        if not r:
            raise KeyError(f"unknown version {vid}")
        d = dict(r)
        d["params"] = json.loads(d["params"])
        return d

    def list_versions(self) -> List[dict]:
        out = []
        for r in self.conn.execute(
                "SELECT id,lineage,parent_id,kind,label,raw_id,created_seq"
                " FROM versions ORDER BY created_seq,id"):
            out.append(dict(r))
        return out

    # --------------------------------------------------------------- derived
    def add_derived(self, kind: str, subject_id: str, subject_version: Optional[str],
                    algo_version: str, params: dict, source_ids: List[str],
                    payload: dict) -> dict:
        did = self._new_id("der", "der")
        fp = fingerprint({"algo": algo_version, "params": params})
        self.conn.execute(
            "INSERT INTO derived(id,kind,subject_id,subject_version,algo_version,"
            "params_fp,params_json,source_ids,payload,created_seq)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (did, kind, subject_id, subject_version, algo_version, fp,
             json.dumps(params, ensure_ascii=False), json.dumps(source_ids),
             json.dumps(payload, ensure_ascii=False), self._seq()),
        )
        self.conn.commit()
        evt = self.add_event("derived-added", {
            "derived_id": did, "kind": kind, "subject_id": subject_id,
            "subject_version": subject_version, "algo_version": algo_version,
            "params_fp": fp, "source_ids": list(source_ids),
        })
        return {"id": did, "params_fp": fp, "event": evt["id"]}

    def get_derived(self, did: str) -> dict:
        r = self.conn.execute("SELECT * FROM derived WHERE id=?", (did,)).fetchone()
        if not r:
            raise KeyError(f"unknown derived {did}")
        return self._derived_row(r)

    def _derived_row(self, r: sqlite3.Row, stale: Optional[Dict[str, str]] = None) -> dict:
        d = dict(r)
        d["params"] = json.loads(d.pop("params_json"))
        d["source_ids"] = json.loads(d["source_ids"])
        d["payload"] = json.loads(d["payload"])
        stale = stale if stale is not None else self._stale_map()
        d["status"] = "stale" if d["id"] in stale else "current"
        d["staled_by_version"] = stale.get(d["id"])
        return d

    def list_derived(self, kind: Optional[str] = None, subject_id: Optional[str] = None,
                     include_stale: bool = False, at_seq: Optional[int] = None) -> List[dict]:
        stale = self._stale_map(at_seq)
        out = []
        for r in self.conn.execute("SELECT * FROM derived ORDER BY created_seq,id"):
            if at_seq is not None and r["created_seq"] > at_seq:
                continue
            if kind and r["kind"] != kind:
                continue
            if subject_id and r["subject_id"] != subject_id:
                continue
            d = self._derived_row(r, stale)
            if d["status"] == "stale" and not include_stale:
                continue
            out.append(d)
        return out

    def _stale_map(self, at_seq: Optional[int] = None) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for evt in self.events():
            if at_seq is not None and evt["seq"] > at_seq:
                break
            if evt["type"] == "derived-staled":
                for did in evt["payload"]["derived_ids"]:
                    out[did] = evt["payload"]["by_version"]
        return out

    # -------------------------------------------------------------- decisions
    def add_decision(self, target: str, verdict: str, rationale: str, reviewer: str,
                     derived_id: Optional[str] = None,
                     params: Optional[dict] = None) -> dict:
        if verdict not in ("accept", "reject", "adjust"):
            raise ValueError("verdict must be accept|reject|adjust")
        return self.add_event("decision", {
            "target": target, "verdict": verdict, "rationale": rationale,
            "reviewer": reviewer, "derived_id": derived_id, "params": params or {},
        })

    def resolve_conflict(self, target: str, verdict: str, rationale: str,
                         reviewer: str, derived_id: Optional[str] = None) -> dict:
        return self.add_event("conflict-resolution", {
            "target": target, "verdict": verdict, "rationale": rationale,
            "reviewer": reviewer, "derived_id": derived_id,
        })

    def undo_event(self, target_event_id: str, reviewer: str, note: str = "") -> dict:
        target = self.event_by_id(target_event_id)
        if target is None:
            raise KeyError(f"unknown event {target_event_id}")
        return self.add_event("undo", {
            "target_event": target_event_id, "target_seq": target["seq"],
            "target_type": target["type"], "reviewer": reviewer, "note": note,
        })

    def effective_decisions(self, at_seq: Optional[int] = None) -> List[dict]:
        """Replay decision/conflict/undo events; conflicts stay explicit."""
        undone: set = set()
        rows: List[dict] = []
        for evt in self.events():
            if at_seq is not None and evt["seq"] > at_seq:
                break
            p = evt["payload"]
            if evt["type"] == "undo":
                undone.add(p["target_event"])
            elif evt["type"] in ("decision", "conflict-resolution"):
                rows.append({"event_id": evt["id"], "seq": evt["seq"], "ts": evt["ts"],
                             "type": evt["type"], **p})
        active = [r for r in rows if r["event_id"] not in undone]
        by_target: Dict[str, List[dict]] = {}
        for r in active:
            by_target.setdefault(r["target"], []).append(r)
        out: List[dict] = []
        for target, group in sorted(by_target.items()):
            resolutions = [g for g in group if g["type"] == "conflict-resolution"]
            plain = [g for g in group if g["type"] == "decision"]
            verdicts = {g["verdict"] for g in plain}
            if resolutions:
                last = resolutions[-1]
                out.append({"target": target, "state": "resolved",
                            "verdict": last["verdict"], "by": last["reviewer"],
                            "event_id": last["event_id"],
                            "history": [g["event_id"] for g in group]})
            elif len(verdicts) > 1:
                out.append({"target": target, "state": "conflict",
                            "verdict": None,
                            "conflicting": [
                                {"verdict": g["verdict"], "reviewer": g["reviewer"],
                                 "event_id": g["event_id"]} for g in plain],
                            "history": [g["event_id"] for g in group]})
            elif plain:
                last = plain[-1]
                out.append({"target": target, "state": "decided",
                            "verdict": last["verdict"], "by": last["reviewer"],
                            "event_id": last["event_id"],
                            "history": [g["event_id"] for g in group]})
        return out

    # ---------------------------------------------------------- history view
    def view_at(self, event_id: str) -> dict:
        """Reconstruct visible state as of an event: dependency versions and
        conclusions of that moment, replayed from the log (not overwritten by
        newer computation)."""
        target = self.event_by_id(event_id)
        if target is None:
            raise KeyError(f"unknown event {event_id}")
        seq = target["seq"]
        versions = [v for v in self.list_versions() if v["created_seq"] <= seq]
        derived = self.list_derived(include_stale=True, at_seq=seq)
        return {
            "as_of_event": target,
            "versions": versions,
            "derived": derived,
            "decisions": self.effective_decisions(at_seq=seq),
            "events": [e for e in self.events() if e["seq"] <= seq],
        }

    # ---------------------------------------------------------- export/import
    def export_dump(self) -> dict:
        return {
            "format": "macro-scope-mirror/dump/v1",
            "meta": {r["key"]: r["value"] for r in
                     self.conn.execute("SELECT key,value FROM meta")},
            "raw_inputs": [dict(r) for r in self.conn.execute(
                "SELECT * FROM raw_inputs ORDER BY id")],
            "versions": [dict(r) for r in self.conn.execute(
                "SELECT * FROM versions ORDER BY created_seq,id")],
            "derived": [dict(r) for r in self.conn.execute(
                "SELECT * FROM derived ORDER BY created_seq,id")],
            "events": [dict(r) for r in self.conn.execute(
                "SELECT * FROM events ORDER BY seq")],
            "case_pack": [dict(r) for r in self.conn.execute(
                "SELECT * FROM case_pack ORDER BY id")],
        }

    def import_dump(self, dump: dict) -> dict:
        if dump.get("format") != "macro-scope-mirror/dump/v1":
            raise ValueError("unsupported dump format")
        if self._seq() > 0 or self.list_raw():
            raise ConflictError("import requires an empty database; delete first")
        with self.conn:
            for r in dump["raw_inputs"]:
                self.conn.execute(
                    "INSERT INTO raw_inputs(id,kind,content,content_hash,submitted_ts,"
                    "submitted_event,representation) VALUES(?,?,?,?,?,?,?)",
                    (r["id"], r["kind"], r["content"], r["content_hash"],
                     r["submitted_ts"], r["submitted_event"],
                     r.get("representation", "text")))
            for r in dump["versions"]:
                self.conn.execute(
                    "INSERT INTO versions(id,lineage,parent_id,kind,label,raw_id,params,"
                    "created_seq) VALUES(?,?,?,?,?,?,?,?)",
                    (r["id"], r["lineage"], r["parent_id"], r["kind"], r["label"],
                     r["raw_id"], r["params"], r["created_seq"]))
            for r in dump["derived"]:
                self.conn.execute(
                    "INSERT INTO derived(id,kind,subject_id,subject_version,algo_version,"
                    "params_fp,params_json,source_ids,payload,created_seq)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (r["id"], r["kind"], r["subject_id"], r["subject_version"],
                     r["algo_version"], r["params_fp"], r["params_json"],
                     r["source_ids"], r["payload"], r["created_seq"]))
            for r in dump["events"]:
                self.conn.execute(
                    "INSERT INTO events(seq,id,type,payload,ts) VALUES(?,?,?,?,?)",
                    (r["seq"], r["id"], r["type"], r["payload"], r["ts"]))
            for r in dump.get("case_pack", []):
                self.conn.execute(
                    "INSERT INTO case_pack(id,title,loaded_ts) VALUES(?,?,?)",
                    (r["id"], r["title"], r["loaded_ts"]))
            for k, v in dump.get("meta", {}).items():
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, v))
        return {"imported": True,
                "events": len(dump["events"]),
                "derived": len(dump["derived"]),
                "versions": len(dump["versions"])}

    def delete_database(self) -> dict:
        self.conn.close()
        if os.path.exists(self.path):
            os.remove(self.path)
        return {"deleted": True, "path": self.path}

    def register_case_pack(self, pack_id: str, title: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO case_pack(id,title,loaded_ts) VALUES(?,?,?)",
            (pack_id, title, self.clock()))
        self.conn.commit()
