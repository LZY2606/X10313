"""Local-first SQLite persistence.

Everything is stored as canonical JSON in text columns so an export is a
plain JSON document and a delete+import round trip reproduces every stable
id, ordering and provenance relation.
"""
from __future__ import annotations

import json

from .jsonio import dumps as jdumps, dumps_pretty
import os
import sqlite3
from typing import Any, Optional

from .ids import VirtualClock, digest
from .pipeline import import_source, run_expansion

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_vt TEXT NOT NULL,
  detail_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS input_versions (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  seq_index INTEGER NOT NULL,
  label TEXT NOT NULL,
  parent_version_id TEXT,
  raw_text TEXT NOT NULL,
  detail_json TEXT NOT NULL,
  created_vt TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT NOT NULL,
  project_id TEXT NOT NULL,
  input_version_id TEXT NOT NULL,
  params_fingerprint TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (id)
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  vt TEXT NOT NULL,
  kind TEXT NOT NULL,
  summary TEXT NOT NULL,
  undo_of TEXT,
  detail_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  target_type TEXT NOT NULL,
  target_id TEXT NOT NULL,
  verdict TEXT NOT NULL,
  note TEXT NOT NULL,
  superseded_by TEXT,
  event_id TEXT NOT NULL,
  vt TEXT NOT NULL,
  detail_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  detail_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  event_seq INTEGER NOT NULL,
  label TEXT NOT NULL,
  detail_json TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str, seed: int = 20260925):
        self.path = path
        first = not os.path.exists(path)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.clock = VirtualClock(seed)
        if first:
            self._set_meta("seed", str(seed))
        else:
            saved = self._get_meta("seed")
            if saved and int(saved) != seed:
                # replay with the persisted seed to keep timeline stable
                self.clock = VirtualClock(int(saved))

    # ------------------------------------------------------------------
    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))
        self.conn.commit()

    def _get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?",
                                (key,)).fetchone()
        return row["value"] if row else None

    # ---- projects -----------------------------------------------------
    def create_project(self, name: str) -> dict[str, Any]:
        pid = digest(["project", name], "proj", 16)
        vt = self.clock.now()
        detail = {"name": name, "seed": self.clock.seed}
        self.conn.execute(
            "INSERT OR IGNORE INTO projects(id,name,created_vt,detail_json)"
            " VALUES(?,?,?,?)",
            (pid, name, vt, jdumps(detail)))
        self.conn.commit()
        if self.conn.total_changes:
            self.add_event(pid, "project-created", f"项目创建：{name}",
                           {"name": name})
        return {"id": pid, "name": name, "created_vt": vt}

    def list_projects(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM projects ORDER BY created_vt")
        return [{"id": r["id"], "name": r["name"],
                 "created_vt": r["created_vt"],
                 **json.loads(r["detail_json"])} for r in rows]

    def get_project(self, pid: str) -> Optional[dict[str, Any]]:
        r = self.conn.execute("SELECT * FROM projects WHERE id=?",
                              (pid,)).fetchone()
        if not r:
            return None
        return {"id": r["id"], "name": r["name"],
                "created_vt": r["created_vt"], **json.loads(r["detail_json"])}

    # ---- inputs (immutable versions) ----------------------------------
    def submit_input(self, project_id: str, text: str, label: str,
                     parent_version_id: Optional[str] = None) -> dict[str, Any]:
        seq_index = self.conn.execute(
            "SELECT COALESCE(MAX(seq_index)+1,0) FROM input_versions "
            "WHERE project_id=?", (project_id,)).fetchone()[0]
        payload = import_source(project_id=project_id, text=text,
                                parent_version_id=parent_version_id,
                                label=label, clock=self.clock,
                                seq_index=seq_index)
        self.conn.execute(
            "INSERT INTO input_versions(id,project_id,seq_index,label,"
            "parent_version_id,raw_text,detail_json,created_vt) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (payload["id"], project_id, seq_index, label,
             parent_version_id, text,
             jdumps(payload), payload["created_vt"]))
        self.conn.commit()
        self.add_event(
            project_id, "input-submitted",
            f"提交输入版本 v{seq_index}（{label}）",
            {"input_version_id": payload["id"], "seq_index": seq_index,
             "parent_version_id": parent_version_id,
             "tokens": len(payload["tokens"]),
             "diagnostics": len(payload["diagnostics"]),
             "immutable": True})
        self._mark_dependents_stale(project_id, payload["id"])
        return payload

    def list_inputs(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT detail_json FROM input_versions WHERE project_id=? "
            "ORDER BY seq_index", (project_id,))
        return [json.loads(r["detail_json"]) for r in rows]

    def get_input(self, input_id: str) -> Optional[dict[str, Any]]:
        r = self.conn.execute("SELECT detail_json FROM input_versions "
                              "WHERE id=?", (input_id,)).fetchone()
        return json.loads(r["detail_json"]) if r else None

    def latest_input_id(self, project_id: str) -> Optional[str]:
        r = self.conn.execute(
            "SELECT id FROM input_versions WHERE project_id=? "
            "ORDER BY seq_index DESC LIMIT 1", (project_id,)).fetchone()
        return r["id"] if r else None

    # ---- artifacts (derived) ------------------------------------------
    def compute_artifact(self, project_id: str, input_version_id: str,
                         params: Optional[dict[str, Any]] = None,
                         attempt_seq: Optional[int] = None) -> dict[str, Any]:
        iv = self.get_input(input_version_id)
        if iv is None:
            raise KeyError(input_version_id)
        if params is None:
            params = {"macro_budget": 64, "seed": self.clock.seed}
        if attempt_seq is None:
            attempt_seq = self.conn.execute(
                "SELECT COALESCE(MAX(json_extract(payload_json,'$.attempt_seq'))+1,0) "
                "FROM artifacts WHERE project_id=? AND input_version_id=?",
                (project_id, input_version_id)).fetchone()[0]
        payload = run_expansion(project_id=project_id, input_version=iv,
                                params=params, clock=self.clock,
                                attempt_seq=attempt_seq)
        self.conn.execute(
            "INSERT INTO artifacts(id,project_id,input_version_id,"
            "params_fingerprint,status,payload_json) VALUES(?,?,?,?,?,?)",
            (payload["id"], project_id, input_version_id,
             payload["params_fingerprint"], "current",
             jdumps(payload)))
        self.conn.commit()
        self.add_event(
            project_id, "artifact-computed",
            f"展开 v{iv['seq_index']}（预算 {payload['macro_budget']}）→ "
            f"{payload['expansion_status']}",
            {"artifact_id": payload["id"],
             "input_version_id": input_version_id,
             "params_fingerprint": payload["params_fingerprint"],
             "algorithm_version": payload["algorithm_version"],
             "direct_sources": payload["direct_sources"],
             "expansion_status": payload["expansion_status"],
             "call_count": payload["call_count"]})
        return payload

    def list_artifacts(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT status,payload_json FROM artifacts WHERE project_id=? "
            "ORDER BY rowid", (project_id,))
        out = []
        for r in rows:
            p = json.loads(r["payload_json"])
            p["status"] = r["status"]
            out.append(p)
        return out

    def get_artifact(self, artifact_id: str) -> Optional[dict[str, Any]]:
        r = self.conn.execute("SELECT status,payload_json FROM artifacts "
                              "WHERE id=?", (artifact_id,)).fetchone()
        if not r:
            return None
        p = json.loads(r["payload_json"])
        p["status"] = r["status"]
        return p

    def current_artifact(self, project_id: str,
                         input_version_id: str) -> Optional[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT payload_json FROM artifacts WHERE project_id=? "
            "AND input_version_id=? ORDER BY rowid DESC",
            (project_id, input_version_id)).fetchall()
        for r in reversed(rows):
            p = json.loads(r["payload_json"])
            return p
        return None

    def _mark_dependents_stale(self, project_id: str,
                               changed_input_id: str) -> None:
        """Only artifacts directly derived from the changed input go stale.

        Stale artifacts remain readable; the UI must not present them as
        current.
        """
        rows = self.conn.execute(
            "SELECT id,payload_json FROM artifacts WHERE project_id=? "
            "AND status='current'", (project_id,)).fetchall()
        for r in rows:
            p = json.loads(r["payload_json"])
            if changed_input_id in p.get("direct_sources", []):
                self.conn.execute(
                    "UPDATE artifacts SET status='stale' WHERE id=?",
                    (r["id"],))
        self.conn.commit()

    # ---- events / decisions -------------------------------------------
    def add_event(self, project_id: str, kind: str, summary: str,
                  detail: dict[str, Any], *, undo_of: Optional[str] = None
                  ) -> dict[str, Any]:
        seq = self.conn.execute(
            "SELECT COALESCE(MAX(seq)+1,0) FROM events WHERE project_id=?",
            (project_id,)).fetchone()[0]
        eid = digest([project_id, "event", seq, kind], "evt", 16)
        vt = self.clock.now()
        self.conn.execute(
            "INSERT INTO events(id,project_id,seq,vt,kind,summary,undo_of,"
            "detail_json) VALUES(?,?,?,?,?,?,?,?)",
            (eid, project_id, seq, vt, kind, summary, undo_of,
             jdumps(detail)))
        self.conn.commit()
        return {"id": eid, "seq": seq, "vt": vt, "kind": kind,
                "summary": summary, "undo_of": undo_of, "detail": detail}

    def list_events(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE project_id=? ORDER BY seq",
            (project_id,))
        return [{"id": r["id"], "seq": r["seq"], "vt": r["vt"],
                 "kind": r["kind"], "summary": r["summary"],
                 "undo_of": r["undo_of"],
                 "detail": json.loads(r["detail_json"])} for r in rows]

    def record_decision(self, project_id: str, target_type: str,
                        target_id: str, verdict: str, note: str,
                        detail: Optional[dict[str, Any]] = None
                        ) -> dict[str, Any]:
        did = digest([project_id, target_type, target_id, verdict, note],
                     "dec", 16)
        # conflicting decision: mark previous decision on same target stale
        prev = self.conn.execute(
            "SELECT id FROM decisions WHERE project_id=? AND target_id=? "
            "AND superseded_by IS NULL ORDER BY vt DESC LIMIT 1",
            (project_id, target_id)).fetchone()
        event = self.add_event(
            project_id, f"decision-{verdict}",
            f"人工裁定（{verdict}）{target_type} {target_id[:18]}…：{note}",
            {"target_type": target_type, "target_id": target_id,
             "verdict": verdict, "note": note,
             "supersedes": prev["id"] if prev else None,
             **(detail or {})})
        self.conn.execute(
            "INSERT INTO decisions(id,project_id,target_type,target_id,"
            "verdict,note,superseded_by,event_id,vt,detail_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (did, project_id, target_type, target_id, verdict, note,
             None, event["id"], event["vt"],
             jdumps(detail or {})))
        if prev:
            self.conn.execute(
                "UPDATE decisions SET superseded_by=? WHERE id=?",
                (did, prev["id"]))
        self.conn.commit()
        return {"id": did, "event_id": event["id"], "verdict": verdict}

    def undo_last_decision(self, project_id: str) -> Optional[dict[str, Any]]:
        r = self.conn.execute(
            "SELECT * FROM decisions WHERE project_id=? "
            "ORDER BY vt DESC LIMIT 1", (project_id,)).fetchone()
        if not r:
            return None
        self.conn.execute("UPDATE decisions SET superseded_by='<undone>' "
                          "WHERE id=?", (r["id"],))
        self.conn.commit()
        ev = self.add_event(
            project_id, "decision-undone",
            f"撤销裁定：{r['note']}",
            {"decision_id": r["id"], "verdict": r["verdict"],
             "target_id": r["target_id"]}, undo_of=r["event_id"])
        return {"decision_id": r["id"], "event": ev}

    def list_decisions(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM decisions WHERE project_id=? ORDER BY vt",
            (project_id,))
        return [{"id": r["id"], "target_type": r["target_type"],
                 "target_id": r["target_id"], "verdict": r["verdict"],
                 "note": r["note"], "superseded_by": r["superseded_by"],
                 "event_id": r["event_id"], "vt": r["vt"],
                 "detail": json.loads(r["detail_json"])} for r in rows]

    # ---- experiments / checkpoints ------------------------------------
    def save_experiment(self, record: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO experiments(id,project_id,detail_json)"
            " VALUES(?,?,?)",
            (record["id"], record["project_id"],
             jdumps(record)))
        self.conn.commit()
        self.add_event(record["project_id"], "experiment-recorded",
                       f"反例实验 {record['key']}："
                       f"{'采纳' if record['adopted'] else '未采纳'}",
                       {"experiment_id": record["id"], "key": record["key"],
                        "adopted": record["adopted"],
                        "locality_ok": record["locality_ok"]})

    def list_experiments(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT detail_json FROM experiments WHERE project_id=? ORDER BY id",
            (project_id,))
        return [json.loads(r["detail_json"]) for r in rows]

    def add_checkpoint(self, project_id: str, label: str) -> dict[str, Any]:
        last = self.conn.execute(
            "SELECT COALESCE(MAX(seq),-1) FROM events "
            "WHERE project_id=?", (project_id,)).fetchone()[0]
        cid = digest([project_id, "checkpoint", label, last], "cp", 16)
        state = {
            "inputs": self.list_inputs(project_id),
            "artifacts": [{"id": a["id"], "status": a["status"],
                           "input_version_id": a["input_version_id"],
                           "params_fingerprint": a["params_fingerprint"]}
                          for a in self.list_artifacts(project_id)],
            "decisions": self.list_decisions(project_id),
            "experiments": [e["id"] for e in
                            self.list_experiments(project_id)],
        }
        self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints(id,project_id,event_seq,"
            "label,detail_json) VALUES(?,?,?,?,?)",
            (cid, project_id, last, label,
             jdumps(state)))
        self.conn.commit()
        self.add_event(project_id, "checkpoint", f"检查点：{label}",
                       {"checkpoint_id": cid, "event_seq": last})
        return {"id": cid, "event_seq": last, "state": state}

    def list_checkpoints(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM checkpoints WHERE project_id=? ORDER BY event_seq",
            (project_id,))
        return [{"id": r["id"], "event_seq": r["event_seq"],
                 "label": r["label"], "state": json.loads(r["detail_json"])}
                for r in rows]

    # ---- export / import ----------------------------------------------
    def export_project(self, project_id: str) -> dict[str, Any]:
        def rows(table: str) -> list[dict[str, Any]]:
            rs = self.conn.execute(
                f"SELECT * FROM {table} WHERE project_id=? ORDER BY rowid",
                (project_id,)).fetchall()
            return [dict(r) for r in rs]
        data = {
            "format": "macroscope-export/1",
            "project_id": project_id,
            "seed": self.clock.seed,
            "project": self.get_project(project_id),
            "input_versions": rows("input_versions"),
            "artifacts": rows("artifacts"),
            "events": rows("events"),
            "decisions": rows("decisions"),
            "experiments": rows("experiments"),
            "checkpoints": rows("checkpoints"),
        }
        data["content_hash"] = digest(
            {k: v for k, v in data.items() if k != "content_hash"},
            "sha", 32)
        return data

    def import_project(self, data: dict[str, Any], *,
                       verify: bool = True) -> str:
        if verify:
            claimed = data.get("content_hash")
            body = {k: v for k, v in data.items() if k != "content_hash"}
            if claimed != digest(body, "sha", 32):
                raise ValueError("export content hash mismatch")
        pid = data["project_id"]
        p = data["project"]
        self.conn.execute(
            "INSERT OR IGNORE INTO projects(id,name,created_vt,detail_json)"
            " VALUES(?,?,?,?)",
            (pid, p["name"], p["created_vt"],
             jdumps({k: v for k, v in p.items()
                     if k not in ("id", "name", "created_vt")})))
        for r in data["input_versions"]:
            self.conn.execute(
                "INSERT OR REPLACE INTO input_versions(id,project_id,seq_index,"
                "label,parent_version_id,raw_text,detail_json,created_vt)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (r["id"], pid, r["seq_index"], r["label"],
                 r["parent_version_id"], r["raw_text"], r["detail_json"],
                 r["created_vt"]))
        for r in data["artifacts"]:
            self.conn.execute(
                "INSERT OR REPLACE INTO artifacts(id,project_id,"
                "input_version_id,params_fingerprint,status,payload_json)"
                " VALUES(?,?,?,?,?,?)",
                (r["id"], pid, r["input_version_id"], r["params_fingerprint"],
                 r["status"], r["payload_json"]))
        for r in data["events"]:
            self.conn.execute(
                "INSERT OR REPLACE INTO events(id,project_id,seq,vt,kind,"
                "summary,undo_of,detail_json) VALUES(?,?,?,?,?,?,?,?)",
                (r["id"], pid, r["seq"], r["vt"], r["kind"], r["summary"],
                 r["undo_of"], r["detail_json"]))
        for r in data["decisions"]:
            self.conn.execute(
                "INSERT OR REPLACE INTO decisions(id,project_id,target_type,"
                "target_id,verdict,note,superseded_by,event_id,vt,detail_json)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (r["id"], pid, r["target_type"], r["target_id"],
                 r["verdict"], r["note"], r["superseded_by"], r["event_id"],
                 r["vt"], r["detail_json"]))
        for r in data["experiments"]:
            self.conn.execute(
                "INSERT OR REPLACE INTO experiments(id,project_id,detail_json)"
                " VALUES(?,?,?)", (r["id"], pid, r["detail_json"]))
        for r in data["checkpoints"]:
            self.conn.execute(
                "INSERT OR REPLACE INTO checkpoints(id,project_id,event_seq,"
                "label,detail_json) VALUES(?,?,?,?,?)",
                (r["id"], pid, r["event_seq"], r["label"], r["detail_json"]))
        self.conn.commit()
        return pid

    def wipe_project(self, project_id: str) -> None:
        for table in ("checkpoints", "experiments", "decisions", "events",
                      "artifacts", "input_versions"):
            self.conn.execute(f"DELETE FROM {table} WHERE project_id=?",
                              (project_id,))
        self.conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
