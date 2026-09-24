"""SQLite append-only project store.

Tables are append-only: original inputs can never be overwritten, a revision
is a new row with a parent pointer; derivations are immutable snapshots with
algorithm version + parameter fingerprint + direct sources; human actions
(accept/reject clause, parameter change, conflict resolution, undo) are
events that never mutate earlier rows.

Export/import round-trips the exact row sets, so stable identities, ordering
and provenance survive deleting the local database.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3

from .clock import Clock
from .derive import run_derivation, fingerprint, ALGORITHM_VERSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
  project_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  clock_start_ms INTEGER NOT NULL,
  clock_step_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS inputs (
  input_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  version_no INTEGER NOT NULL,
  parent_input_id TEXT,
  label TEXT NOT NULL,
  source_text TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  immutable_note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS param_revisions (
  param_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  version_no INTEGER NOT NULL,
  parent_param_id TEXT,
  params_json TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  reason TEXT NOT NULL,
  created_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  ts_ms INTEGER NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  undone_by_event_id TEXT,
  supersedes_event_id TEXT
);
CREATE TABLE IF NOT EXISTS derivations (
  derivation_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  version_no INTEGER NOT NULL,
  input_id TEXT NOT NULL,
  param_id TEXT NOT NULL,
  event_version_id TEXT NOT NULL,
  algorithm_version TEXT NOT NULL,
  param_fingerprint TEXT NOT NULL,
  direct_sources_json TEXT NOT NULL,
  artifact_json TEXT NOT NULL,
  created_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
  project_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value_json TEXT NOT NULL,
  PRIMARY KEY (project_id, key)
);
"""

EXPORT_TABLES = ["projects", "inputs", "param_revisions", "events",
                 "derivations", "meta"]


def _hid(prefix: str, *parts) -> str:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{h[:20]}"


class Store:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ----- helpers ----------------------------------------------------
    def _clock_for(self, project_id: str) -> Clock:
        row = self.conn.execute(
            "SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"no project {project_id}")
        return Clock(row["clock_start_ms"], row["clock_step_ms"], row["seed"])

    def _event_seq(self, project_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS m FROM events WHERE project_id=?",
            (project_id,)).fetchone()
        return int(row["m"])

    def _next_version(self, project_id: str, table: str, col: str) -> int:
        row = self.conn.execute(
            f"SELECT COALESCE(MAX({col}), 0) AS m FROM {table} WHERE project_id=?",
            (project_id,)).fetchone()
        return int(row["m"]) + 1

    def _add_event(self, project_id: str, kind: str, payload: dict,
                   supersedes=None, clock=None) -> dict:
        seq = self._event_seq(project_id) + 1
        ts = clock.tick() if clock else 0
        event_id = _hid("ev", project_id, seq, kind,
                        json.dumps(payload, sort_keys=True, ensure_ascii=False))
        self.conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
            (event_id, project_id, seq, ts, kind,
             json.dumps(payload, ensure_ascii=False), None, supersedes))
        return {"event_id": event_id, "seq": seq, "ts_ms": ts, "kind": kind,
                "payload": payload}

    # ----- project lifecycle -----------------------------------------
    def create_project(self, title: str, *, seed: int = 20260925,
                       clock_start_ms: int = 1_700_000_000_000,
                       clock_step_ms: int = 1) -> str:
        project_id = _hid("proj", title, seed, clock_start_ms)
        self.conn.execute(
            "INSERT OR IGNORE INTO projects VALUES (?,?,?,?,?,?)",
            (project_id, title, clock_start_ms, seed, clock_start_ms, clock_step_ms))
        self.conn.commit()
        clock = self._clock_for(project_id)
        self._add_event(project_id, "project_created", {"title": title}, clock=clock)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta VALUES (?,?,?)",
            (project_id, "initial_params", json.dumps({"budget": 64})))
        self.conn.commit()
        return project_id

    def list_projects(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM projects ORDER BY created_ms").fetchall()
        return [dict(r) for r in rows]

    def get_project(self, project_id: str) -> dict:
        r = self.conn.execute("SELECT * FROM projects WHERE project_id=?",
                              (project_id,)).fetchone()
        return dict(r) if r else None

    # ----- inputs -----------------------------------------------------
    def add_input(self, project_id: str, label: str, source_text: str,
                  parent_input_id=None, diagnostics=None) -> str:
        clock = self._clock_for(project_id)
        if not source_text or not source_text.strip():
            raise ValueError("source text must not be empty")
        if parent_input_id is not None:
            par = self.conn.execute("SELECT input_id FROM inputs WHERE input_id=?",
                                    (parent_input_id,)).fetchone()
            if par is None:
                raise KeyError(f"unknown parent input {parent_input_id}")
        existing = self.conn.execute(
            "SELECT input_id FROM inputs WHERE project_id=? AND source_text=?",
            (project_id, source_text)).fetchone()
        if existing is not None:
            return existing["input_id"]
        version_no = self._next_version(project_id, "inputs", "version_no")
        input_id = _hid("in", project_id, version_no, source_text)
        ts = clock.tick()
        self.conn.execute(
            "INSERT INTO inputs VALUES (?,?,?,?,?,?,?,?)",
            (input_id, project_id, version_no, parent_input_id, label, source_text, ts,
             "original input is immutable; revisions use parent_input_id"))
        ev = self._add_event(project_id, "input_committed",
                             {"input_id": input_id, "version_no": version_no,
                              "label": label, "parent_input_id": parent_input_id,
                              "diagnostics": diagnostics or []}, clock=clock)
        self.conn.commit()
        return input_id

    def list_inputs(self, project_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT input_id, project_id, version_no, parent_input_id, label, "
            "source_text, created_ms FROM inputs WHERE project_id=? ORDER BY version_no",
            (project_id,)).fetchall()
        return [dict(r) for r in rows]

    # ----- parameters -------------------------------------------------
    def revise_params(self, project_id: str, params: dict, reason: str) -> str:
        clock = self._clock_for(project_id)
        cur = self.current_params(project_id)
        version_no = self._next_version(project_id, "param_revisions", "version_no")
        fp = fingerprint(params)
        parent_row = self.conn.execute(
            "SELECT param_id FROM param_revisions WHERE project_id=? ORDER BY version_no "
            "DESC LIMIT 1", (project_id,)).fetchone()
        parent_id = parent_row["param_id"] if parent_row else None
        same = self.conn.execute(
            "SELECT param_id FROM param_revisions WHERE project_id=? AND fingerprint=?",
            (project_id, fp)).fetchone()
        if same is not None and parent_id is None:
            return same["param_id"]
        param_id = _hid("pr", project_id, version_no, fp)
        ts = clock.tick()
        self.conn.execute(
            "INSERT INTO param_revisions VALUES (?,?,?,?,?,?,?,?,?)",
            (param_id, project_id, version_no, parent_id,
             json.dumps(params, ensure_ascii=False), fp, reason, ts))
        self._add_event(project_id, "param_revision",
                        {"param_id": param_id, "version_no": version_no,
                         "params": params, "fingerprint": fp, "reason": reason,
                         "parent_param_id": parent_id}, clock=clock)
        self.conn.commit()
        return param_id

    def current_params(self, project_id: str) -> dict:
        row = self.conn.execute(
            "SELECT params_json FROM param_revisions WHERE project_id=? ORDER BY version_no "
            "DESC LIMIT 1", (project_id,)).fetchone()
        if row:
            return json.loads(row["params_json"])
        meta = self.conn.execute(
            "SELECT value_json FROM meta WHERE project_id=? AND key='initial_params'",
            (project_id,)).fetchone()
        return json.loads(meta["value_json"]) if meta else {"budget": 64}

    def _param_id_effective(self, project_id: str, up_to_seq: int | None) -> tuple[str, dict]:
        q = ("SELECT param_id, params_json FROM param_revisions WHERE project_id=? "
             "ORDER BY version_no DESC")
        rows = self.conn.execute(q, (project_id,)).fetchall()
        if up_to_seq is None:
            if rows:
                return rows[0]["param_id"], json.loads(rows[0]["params_json"])
            return "", self.current_params(project_id)
        # find the revision committed at or before up_to_seq
        for r in rows:
            ev = self.conn.execute(
                "SELECT seq FROM events WHERE project_id=? AND kind='param_revision' "
                "AND json_extract(payload_json,'$.param_id')=?",
                (project_id, r["param_id"])).fetchone()
            if ev and ev["seq"] <= up_to_seq:
                return r["param_id"], json.loads(r["params_json"])
        return "", self.current_params(project_id)

    # ----- human decisions -------------------------------------------
    def add_decision(self, project_id: str, kind: str, payload: dict,
                     *, conflicts_with=None, note="") -> dict:
        """Record an accept/reject/parameter/conflict/undo decision.

        Conflict resolution: if conflicts_with is supplied and the earlier
        decision is still active, both rows are kept; the new one references
        the old via supersedes_event_id. Undo appends an ``undo`` event and
        stamps undone_by_event_id on the target.
        """
        clock = self._clock_for(project_id)
        payload = dict(payload)
        payload.setdefault("note", note)
        supersedes = None
        if kind == "undo":
            target_id = payload.get("target_event_id")
            target = self.conn.execute("SELECT * FROM events WHERE event_id=?",
                                       (target_id,)).fetchone()
            if target is None:
                raise KeyError(f"cannot undo unknown event {target_id}")
            if target["undone_by_event_id"]:
                raise ValueError("event is already undone")
            ev = self._add_event(project_id, "undo", payload, clock=clock)
            self.conn.execute(
                "UPDATE events SET undone_by_event_id=? WHERE event_id=?",
                (ev["event_id"], target_id))
            self.conn.commit()
            return ev
        if conflicts_with:
            prior = self._active_decision(project_id, conflicts_with)
            if prior is not None:
                supersedes = prior["event_id"]
                payload["conflict"] = {
                    "with_event_id": prior["event_id"],
                    "with_payload": prior["payload"],
                    "resolution": payload.get("resolution", "newer decision wins"),
                }
        ev = self._add_event(project_id, kind, payload, supersedes=supersedes,
                             clock=clock)
        self.conn.commit()
        return ev

    def _active_decision(self, project_id: str, decision_key: str):
        """decision_key: a stable action key such as clause override call_oid."""
        rows = self.conn.execute(
            "SELECT * FROM events WHERE project_id=? ORDER BY seq",
            (project_id,)).fetchall()
        latest = None
        for r in rows:
            if r["undone_by_event_id"]:
                continue
            payload = json.loads(r["payload_json"])
            if payload.get("decision_key") == decision_key:
                latest = {"event_id": r["event_id"], "seq": r["seq"],
                          "kind": r["kind"], "payload": payload}
        return latest

    def clause_overrides_active(self, project_id: str) -> dict:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE project_id=? ORDER BY seq",
            (project_id,)).fetchall()
        overrides: dict[str, int] = {}
        for r in rows:
            if r["undone_by_event_id"]:
                continue
            if r["kind"] in ("clause_decision", "conflict_resolution"):
                p = json.loads(r["payload_json"])
                key = p.get("decision_key")
                if key and key.startswith("clause:"):
                    overrides[key.split(":", 1)[1]] = int(p["clause"])
        return overrides

    def timeline(self, project_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE project_id=? ORDER BY seq",
            (project_id,)).fetchall()
        return [{"event_id": r["event_id"], "seq": r["seq"], "ts_ms": r["ts_ms"],
                 "kind": r["kind"], "payload": json.loads(r["payload_json"]),
                 "undone_by_event_id": r["undone_by_event_id"],
                 "supersedes_event_id": r["supersedes_event_id"]} for r in rows]

    # ----- derivations ------------------------------------------------
    def compute_derivation(self, project_id: str, input_id: str, *,
                           label: str = "current", up_to_event_seq: int | None = None,
                           record: bool = True) -> dict:
        inrow = self.conn.execute("SELECT * FROM inputs WHERE input_id=?",
                                  (input_id,)).fetchone()
        if inrow is None:
            raise KeyError(f"unknown input {input_id}")
        param_id, params = self._param_id_effective(project_id, up_to_event_seq)
        overrides = self.clause_overrides_active(project_id)
        if up_to_event_seq is not None:
            overrides = self._overrides_at(project_id, up_to_event_seq)
        event_version_id = self._event_version(project_id, up_to_event_seq)
        clock = self._clock_for(project_id)
        dver = self._next_version(project_id, "derivations", "version_no")
        tag = f"{project_id[:6]}-v{dver}"
        artifact = run_derivation(inrow["source_text"], params, clock=clock,
                                  clause_overrides=overrides, tag=tag)
        sources = {
            "input_id": input_id,
            "input_version_no": inrow["version_no"],
            "param_id": param_id,
            "event_version_id": event_version_id,
            "up_to_event_seq": up_to_event_seq,
        }
        derivation_id = _hid(
            "dv", project_id, input_id, param_id, event_version_id,
            artifact["param_fingerprint"], ALGORITHM_VERSION)
        record_row = {
            "derivation_id": derivation_id,
            "project_id": project_id,
            "version_no": dver,
            "input_id": input_id,
            "param_id": param_id,
            "event_version_id": event_version_id,
            "algorithm_version": ALGORITHM_VERSION,
            "param_fingerprint": artifact["param_fingerprint"],
            "direct_sources": sources,
            "artifact": artifact,
        }
        if record:
            self.conn.execute(
                "INSERT OR IGNORE INTO derivations VALUES (?,?,?,?,?,?,?,?,?,?)",
                (derivation_id, project_id, dver, input_id, param_id, event_version_id,
                 ALGORITHM_VERSION, artifact["param_fingerprint"],
                 json.dumps(sources, ensure_ascii=False),
                 json.dumps(artifact, ensure_ascii=False), clock.tick()))
            self.conn.commit()
        record_row["current"] = self._is_current(project_id, input_id, param_id,
                                                 event_version_id)
        return record_row

    def _event_version(self, project_id: str, up_to_seq: int | None) -> str:
        rows = self.conn.execute(
            "SELECT event_id, seq, undone_by_event_id FROM events WHERE project_id=? "
            "ORDER BY seq", (project_id,)).fetchall()
        kept = [r["event_id"] for r in rows
                if (up_to_seq is None or r["seq"] <= up_to_seq)
                and not r["undone_by_event_id"]]
        return _hid("evv", project_id, "\x1e".join(kept))

    def _overrides_at(self, project_id: str, up_to_seq: int) -> dict:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE project_id=? AND seq<=? ORDER BY seq",
            (project_id, up_to_seq)).fetchall()
        overrides = {}
        undone = {r["undone_by_event_id"] for r in rows if r["undone_by_event_id"]}
        for r in rows:
            if r["event_id"] in undone:
                continue
            if r["kind"] in ("clause_decision", "conflict_resolution"):
                p = json.loads(r["payload_json"])
                key = p.get("decision_key", "")
                if key.startswith("clause:"):
                    overrides[key.split(":", 1)[1]] = int(p["clause"])
        return overrides

    def _is_current(self, project_id, input_id, param_id, event_version_id) -> bool:
        latest_input = self.conn.execute(
            "SELECT input_id FROM inputs WHERE project_id=? ORDER BY version_no DESC "
            "LIMIT 1", (project_id,)).fetchone()
        current_events = self._event_version(project_id, None)
        cur_param, _ = self._param_id_effective(project_id, None)
        return (latest_input and latest_input["input_id"] == input_id
                and current_events == event_version_id and cur_param == param_id)

    def list_derivations(self, project_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT derivation_id, version_no, input_id, param_id, event_version_id, "
            "algorithm_version, param_fingerprint, direct_sources_json, created_ms "
            "FROM derivations WHERE project_id=? ORDER BY version_no",
            (project_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["direct_sources"] = json.loads(d.pop("direct_sources_json"))
            d["current"] = self._is_current(project_id, d["input_id"], d["param_id"],
                                            d["event_version_id"])
            out.append(d)
        return out

    def get_derivation(self, derivation_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM derivations WHERE derivation_id=?",
                              (derivation_id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["direct_sources"] = json.loads(d.pop("direct_sources_json"))
        d["artifact"] = json.loads(d.pop("artifact_json"))
        d["current"] = self._is_current(d["project_id"], d["input_id"], d["param_id"],
                                        d["event_version_id"])
        return d

    def state_at_event(self, project_id: str, event_id: str) -> dict:
        """Replay the project as of an old event (history is never overwritten)."""
        row = self.conn.execute("SELECT seq FROM events WHERE event_id=?",
                                (event_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown event {event_id}")
        up_to = row["seq"]
        inputs = [i for i in self.list_inputs(project_id) if True]
        chosen_input = None
        # input committed at/before that moment: latest such input
        ev_rows = self.conn.execute(
            "SELECT payload_json FROM events WHERE project_id=? AND kind='input_committed' "
            "AND seq<=? ORDER BY seq DESC LIMIT 1", (project_id, up_to)).fetchone()
        if ev_rows:
            chosen_input = json.loads(ev_rows["payload_json"])["input_id"]
        param_id, params = self._param_id_effective(project_id, up_to)
        if chosen_input is None:
            return {"event_id": event_id, "seq": up_to, "available": False,
                    "params": params, "param_id": param_id}
        rec = self.compute_derivation(project_id, chosen_input,
                                      up_to_event_seq=up_to, record=False)
        rec["historical"] = True
        rec["as_of_event_id"] = event_id
        rec["as_of_seq"] = up_to
        return rec

    # ----- staleness --------------------------------------------------
    def staleness(self, project_id: str) -> dict:
        """Which stored derivations are stale w.r.t. direct sources.

        Local invalidation: a derivation is stale only if its exact input /
        parameter / event-version triple is no longer the current one. The
        fine-grained affected set (which displayed objects change) comes from
        the NumPy dependency closure in derive.affected_by.
        """
        latest_input = self.conn.execute(
            "SELECT input_id FROM inputs WHERE project_id=? ORDER BY version_no DESC "
            "LIMIT 1", (project_id,)).fetchone()
        cur_input = latest_input["input_id"] if latest_input else None
        cur_events = self._event_version(project_id, None)
        cur_param, _ = self._param_id_effective(project_id, None)
        rows = self.conn.execute(
            "SELECT derivation_id, input_id, param_id, event_version_id, "
            "direct_sources_json FROM derivations WHERE project_id=?",
            (project_id,)).fetchall()
        items = []
        for r in rows:
            stale_reasons = []
            if r["input_id"] != cur_input:
                stale_reasons.append("input revised (parent-version chain)")
            if r["param_id"] != cur_param:
                stale_reasons.append("parameter revision")
            if r["event_version_id"] != cur_events:
                stale_reasons.append("new human decisions / undo after derivation")
            items.append({"derivation_id": r["derivation_id"],
                          "is_current": not stale_reasons,
                          "stale_reasons": stale_reasons,
                          "viewable": True,
                          "note": "stale results remain viewable but never pose as "
                                  "current" if stale_reasons else ""})
        return {"current_input_id": cur_input, "current_event_version_id": cur_events,
                "current_param_id": cur_param, "derivations": items}

    # ----- export / import -------------------------------------------
    def export_project(self, project_id: str) -> dict:
        out = {"format": "macro-scope-mirror-archive", "format_version": 1,
               "project_id": project_id, "tables": {}}
        for table in EXPORT_TABLES:
            rows = self.conn.execute(
                f"SELECT * FROM {table} WHERE project_id=?", (project_id,)).fetchall()
            out["tables"][table] = [dict(r) for r in rows]
        out["row_counts"] = {t: len(out["tables"][t]) for t in EXPORT_TABLES}
        return out

    def import_archive(self, archive: dict, *, replace: bool = False) -> str:
        if archive.get("format") != "macro-scope-mirror-archive":
            raise ValueError("not a macro-scope-mirror archive")
        project_id = archive["project_id"]
        if replace:
            for table in reversed(EXPORT_TABLES):
                self.conn.execute(f"DELETE FROM {table} WHERE project_id=?",
                                  (project_id,))
        for table in EXPORT_TABLES:
            cols = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for row in archive["tables"].get(table, []):
                keys = [k for k in row.keys() if k in cols]
                placeholders = ",".join("?" for _ in keys)
                sql = (f"INSERT OR IGNORE INTO {table} ({','.join(keys)}) VALUES "
                       f"({placeholders})")
                self.conn.execute(sql, tuple(row[k] for k in keys))
        self.conn.commit()
        return project_id

    def delete_project(self, project_id: str) -> None:
        for table in reversed(EXPORT_TABLES):
            self.conn.execute(f"DELETE FROM {table} WHERE project_id=?",
                              (project_id,))
        self.conn.commit()

    def counterexample_events(self, project_id: str) -> list[dict]:
        """Machine-readable judgments required by the counterexample archive."""
        out = []
        for d in self.list_derivations(project_id):
            full = self.get_derivation(d["derivation_id"])
            art = full["artifact"]
            out.append({
                "judgment": "capture_decided_by_scope_relation_not_rename",
                "derivation_id": d["derivation_id"],
                "evidence": [{
                    "edge_id": e["edge_id"], "symbol": e["symbol"],
                    "scopes": e["scopes"], "status": e["status"],
                    "binding_id": e["binding_id"],
                } for e in art["binding_edges"]],
                "rule": "membership + creation order over scope sets; spelling is "
                        "display-only",
            })
            out.append({
                "judgment": "datum_to_syntax_context_choice_must_be_visible",
                "derivation_id": d["derivation_id"],
                "evidence": art["context_choices"],
            })
            out.append({
                "judgment": "recursive_expansion_is_budgeted_failure_keeps_partial_graph",
                "derivation_id": d["derivation_id"],
                "evidence": {
                    "status": art["status"], "budget": art["budget"],
                    "calls_used": art["calls_used"], "failure": art["failure"],
                    "partial_tree_present": bool(art["tree"]),
                },
            })
        return out
