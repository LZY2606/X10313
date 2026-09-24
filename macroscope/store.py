"""Local-first SQLite store: immutable inputs, append-only events, derived
artifacts with algorithm version + parameter fingerprint + direct sources.

- Original inputs are immutable once submitted; revisions are new rows with
  a parent pointer (same family).
- Every derived value carries algo_version, param fingerprint and sources,
  so any state can be replayed step by step after reopening the project.
- A cursor over the event log gives time travel: moving back to an old
  event restores the dependency versions and visible conclusions of that
  moment instead of overwriting history with fresh results.
- Staleness is computed, not destructive: stale artifacts stay viewable.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Callable, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS inputs(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT UNIQUE NOT NULL,
  family TEXT NOT NULL,
  parent TEXT,
  content TEXT NOT NULL,
  sha TEXT NOT NULL,
  vt INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT UNIQUE NOT NULL,
  vt INTEGER NOT NULL,
  type TEXT NOT NULL,
  payload TEXT NOT NULL,
  algo TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  sources TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT UNIQUE NOT NULL,
  kind TEXT NOT NULL,
  algo TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  sources TEXT NOT NULL,
  payload TEXT NOT NULL,
  event_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""

DEFAULT_PARAMS = {"budget": 64}
ARCHIVE_KIND = "macroscope-archive"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(algo: str, params: dict) -> str:
    return hashlib.sha256(canonical({"algo": algo, "params": params}).encode()).hexdigest()[:16]


class Store:
    def __init__(self, path: str, clock: Optional[Callable[[], int]] = None,
                 algo_version: str = "macroscope-expander/1.2.0"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._clock = clock or self._default_clock
        self.algo_version = algo_version
        if self._get_meta("cursor") is None:
            self._set_meta("cursor", "0")

    @staticmethod
    def _default_clock() -> int:
        import time
        return int(time.time() * 1000)

    def close(self):
        self.conn.close()

    # -- meta ---------------------------------------------------------------
    def _get_meta(self, k: str) -> Optional[str]:
        row = self.conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return row["v"] if row else None

    def _set_meta(self, k: str, v: str):
        self.conn.execute("INSERT INTO meta(k,v) VALUES(?,?) "
                          "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))
        self.conn.commit()

    @property
    def params(self) -> dict:
        """Current params are derived by replaying param_revision events up to
        the cursor, so time travel restores the dependency versions of that
        moment instead of overwriting history."""
        params = dict(DEFAULT_PARAMS)
        for e in self.list_events(visible_only=True):
            if e["type"] == "param_revision":
                params.update(e["payload"]["params"])
        return params

    def set_params(self, params: dict) -> dict:
        return self.add_event("param_revision", {"params": params}, sources=[])

    @property
    def cursor(self) -> int:
        return int(self._get_meta("cursor"))

    def set_cursor(self, seq: int):
        self._set_meta("cursor", str(seq))

    # -- inputs (immutable) ---------------------------------------------------
    def add_input(self, content: str, parent: Optional[str] = None) -> dict:
        sha = hashlib.sha256(content.encode()).hexdigest()
        iid = f"in-{sha[:12]}"
        existing = self.conn.execute("SELECT * FROM inputs WHERE id=?", (iid,)).fetchone()
        if existing:
            if existing["content"] != content:  # defensive; sha makes this impossible
                raise ValueError("input id collision with different content")
            return dict(existing)  # idempotent re-submit, never overwritten
        family = iid
        if parent:
            prow = self.conn.execute("SELECT family FROM inputs WHERE id=?", (parent,)).fetchone()
            if not prow:
                raise KeyError(f"unknown parent input {parent}")
            family = prow["family"]
        self.conn.execute(
            "INSERT INTO inputs(id,family,parent,content,sha,vt) VALUES(?,?,?,?,?,?)",
            (iid, family, parent, content, sha, self._clock()))
        self.conn.commit()
        return self.get_input(iid)

    def get_input(self, iid: str) -> dict:
        row = self.conn.execute("SELECT * FROM inputs WHERE id=?", (iid,)).fetchone()
        if not row:
            raise KeyError(iid)
        return dict(row)

    def list_inputs(self) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM inputs ORDER BY seq").fetchall()
        heads = self._family_heads()
        return [{**dict(r), "is_head": r["id"] in heads} for r in rows]

    def _family_heads(self) -> set:
        rows = self.conn.execute(
            "SELECT family, MAX(seq) AS m FROM inputs GROUP BY family").fetchall()
        heads = set()
        for r in rows:
            row = self.conn.execute("SELECT id FROM inputs WHERE family=? AND seq=?",
                                    (r["family"], r["m"])).fetchone()
            heads.add(row["id"])
        return heads

    # -- events (append-only) -------------------------------------------------
    def add_event(self, etype: str, payload: dict, sources: List[str],
                  fingerprint_: str = "") -> dict:
        seq_row = self.conn.execute("SELECT COALESCE(MAX(seq),0)+1 AS n FROM events").fetchone()
        seq = seq_row["n"]
        eid = f"ev-{seq:06d}"
        if fingerprint_:
            fp = fingerprint_
        elif etype == "param_revision":
            fp = fingerprint(self.algo_version, payload["params"])
        else:
            fp = fingerprint(self.algo_version, self.params)
        self.conn.execute(
            "INSERT INTO events(id,vt,type,payload,algo,fingerprint,sources) "
            "VALUES(?,?,?,?,?,?,?)",
            (eid, self._clock(), etype, canonical(payload), self.algo_version, fp,
             canonical(sources)))
        self.conn.commit()
        self.set_cursor(seq)  # new events move the cursor to the present
        return self.get_event(seq)

    def get_event(self, seq: int) -> dict:
        row = self.conn.execute("SELECT * FROM events WHERE seq=?", (seq,)).fetchone()
        if not row:
            raise KeyError(seq)
        d = dict(row)
        d["payload"] = json.loads(d["payload"])
        d["sources"] = json.loads(d["sources"])
        return d

    def list_events(self, visible_only: bool = True) -> List[dict]:
        rows = self.conn.execute("SELECT seq FROM events ORDER BY seq").fetchall()
        events = [self.get_event(r["seq"]) for r in rows]
        if visible_only:
            events = [e for e in events if e["seq"] <= self.cursor]
        return events

    # -- artifacts --------------------------------------------------------------
    def add_artifact(self, kind: str, payload: dict, sources: List[str],
                     event_seq: int, artifact_id: Optional[str] = None) -> dict:
        fp = fingerprint(self.algo_version, self.params)
        if artifact_id is None:
            seq_row = self.conn.execute(
                "SELECT COALESCE(MAX(seq),0)+1 AS n FROM artifacts").fetchone()
            artifact_id = f"art-{seq_row['n']:04d}"
        self.conn.execute(
            "INSERT INTO artifacts(id,kind,algo,fingerprint,sources,payload,event_seq) "
            "VALUES(?,?,?,?,?,?,?)",
            (artifact_id, kind, self.algo_version, fp, canonical(sources),
             canonical(payload), event_seq))
        self.conn.commit()
        return self.get_artifact(artifact_id)

    def get_artifact(self, aid: str) -> dict:
        row = self.conn.execute("SELECT * FROM artifacts WHERE id=?", (aid,)).fetchone()
        if not row:
            raise KeyError(aid)
        return self._artifact_dict(row)

    def _artifact_dict(self, row) -> dict:
        d = dict(row)
        d["sources"] = json.loads(d["sources"])
        d["payload"] = json.loads(d["payload"])
        d["stale"], d["stale_reasons"] = self._staleness(d)
        return d

    def list_artifacts(self) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM artifacts ORDER BY seq").fetchall()
        return [self._artifact_dict(r) for r in rows]

    def _staleness(self, art: dict) -> (bool, List[str]):
        reasons = []
        heads = self._family_heads()
        for src in art["sources"]:
            if src.startswith("in-") and src not in heads:
                reasons.append(f"input {src} superseded by a newer revision")
        if art["fingerprint"] != fingerprint(self.algo_version, self.params):
            reasons.append("parameter fingerprint differs from current params")
        return (len(reasons) > 0, reasons)

    # -- decisions / undo (derived from visible events) --------------------------
    def decisions(self) -> Dict[str, dict]:
        """Current decision per target, honoring cursor and undo events."""
        out: Dict[str, dict] = {}
        undone_targets = set()
        for e in self.list_events(visible_only=True):
            if e["type"] == "decision":
                t = e["payload"]["target"]
                out[t] = {"verdict": e["payload"]["verdict"],
                          "rationale": e["payload"].get("rationale", ""),
                          "event_seq": e["seq"], "undone": False}
                undone_targets.discard(t)
            elif e["type"] == "undo":
                tgt_seq = e["payload"]["event_seq"]
                try:
                    tgt = self.get_event(tgt_seq)
                except KeyError:
                    continue
                if tgt["type"] == "decision":
                    t = tgt["payload"]["target"]
                    if t in out:
                        out[t]["undone"] = True
        return out

    def conflicts(self) -> List[dict]:
        """Targets with more than one non-undone, differing visible verdict."""
        by_target: Dict[str, List[dict]] = {}
        dec = self.decisions()
        for e in self.list_events(visible_only=True):
            if e["type"] != "decision":
                continue
            t = e["payload"]["target"]
            cur = dec.get(t)
            if cur and not cur["undone"]:
                by_target.setdefault(t, []).append(
                    {"verdict": e["payload"]["verdict"], "event_seq": e["seq"]})
        out = []
        for t, verdicts in by_target.items():
            distinct = {v["verdict"] for v in verdicts}
            if len(distinct) > 1:
                out.append({"target": t, "verdicts": verdicts,
                            "resolved_as": dec[t]["verdict"]})
        return out

    # -- export / import ---------------------------------------------------------
    def export(self) -> dict:
        tables = {}
        for table in ("inputs", "events", "artifacts", "meta"):
            rows = self.conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            tables[table] = [dict(r) for r in rows]
        return {"kind": ARCHIVE_KIND, "version": 1, "tables": tables}

    def import_archive(self, archive: dict, merge: bool = False):
        if archive.get("kind") != ARCHIVE_KIND:
            raise ValueError("not a macroscope archive")
        if not merge:
            for table in ("inputs", "events", "artifacts", "meta"):
                self.conn.execute(f"DELETE FROM {table}")
        counts = {}
        for table in ("inputs", "events", "artifacts", "meta"):
            rows = archive["tables"].get(table, [])
            counts[table] = len(rows)
            for row in rows:
                cols = list(row.keys())
                if table == "meta":
                    self.conn.execute(
                        "INSERT INTO meta(k,v) VALUES(?,?) "
                        "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                        (row["k"], row["v"]))
                else:
                    placeholders = ",".join("?" for _ in cols)
                    self.conn.execute(
                        f"INSERT OR IGNORE INTO {table}({','.join(cols)}) "
                        f"VALUES({placeholders})", [row[c] for c in cols])
        self.conn.commit()
        return counts
