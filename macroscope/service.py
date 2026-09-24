"""Domain service: versions, expansion, decisions, replay, export/import.

- Original input is immutable once imported; revisions are new versions with a
  parent pointer.
- Every derived artifact carries algo version, parameter fingerprint and its
  direct sources; stale artifacts stay viewable but are flagged.
- Capture is judged by scope relations; datum->syntax context choices are
  visible events; recursive expansion has a budget and keeps partial graphs.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from . import ALGO_VERSION, ARCHIVE_FORMAT, INPUT_FORMAT
from .expander import Expander
from .language import LanguageError, lex, parse
from .store import EventStore, canonical, content_id

DEFAULT_PARAMS = {"budget": 16}


class ServiceError(Exception):
    def __init__(self, message: str, diagnostics: Optional[list] = None):
        super().__init__(message)
        self.message = message
        self.diagnostics = diagnostics or []


def _frag_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def split_fragments(text: str, file: str) -> Dict[str, dict]:
    forms = parse(text, file=file, sid_prefix="src")
    fragments: Dict[str, dict] = {}
    expr_n = 0
    for form in forms:
        frag_text = text[form.span.start_offset:form.span.end_offset]
        if (form.is_list and form.datum and form.datum[0].is_ident
                and form.datum[0].datum == "defmacro"):
            name = form.datum[1].datum
            fid = f"defmacro:{name}"
            kind = "defmacro"
        elif (form.is_list and form.datum and form.datum[0].is_ident
              and form.datum[0].datum == "def"):
            name = form.datum[1].datum
            fid = f"def:{name}"
            kind = "def"
        else:
            expr_n += 1
            fid = f"expr:{expr_n}"
            kind = "expr"
            name = None
        if fid in fragments:
            raise ServiceError(f"duplicate fragment {fid}")
        fragments[fid] = {"kind": kind, "name": name, "text": frag_text,
                          "hash": _frag_hash(frag_text),
                          "span": form.span.to_obj()}
    return fragments


SPAN_RAW_RE = re.compile(r"^L(\d+):C(\d+)-L(\d+):C(\d+)$")


def normalize_annotations(text: str, annotations: List[dict]) -> List[dict]:
    """Raw span annotations are kept verbatim; normalized offsets are derived.
    Unconvertible values stay in place with a diagnostic."""
    lines = text.splitlines(keepends=True)
    line_starts = [0]
    for ln in lines:
        line_starts.append(line_starts[-1] + len(ln))
    out = []
    for i, ann in enumerate(annotations):
        raw = ann.get("span_raw", "")
        entry = {"aid": f"ann-{i+1}", "note": ann.get("note", ""),
                 "span_raw": raw, "normalized": None, "diagnostic": None}
        m = SPAN_RAW_RE.match(raw)
        if not m:
            entry["diagnostic"] = ("cannot normalize span annotation "
                                   f"{raw!r}; kept in raw form")
        else:
            l1, c1, l2, c2 = (int(g) for g in m.groups())
            if l1 < 1 or l2 > len(lines) or c1 < 1:
                entry["diagnostic"] = (f"span {raw!r} out of range for source; "
                                       "kept in raw form")
            else:
                entry["normalized"] = {
                    "start_offset": line_starts[l1 - 1] + c1 - 1,
                    "end_offset": line_starts[l2 - 1] + c2 - 1,
                    "basis": "offset 0-based exclusive; unit=unicode-codepoint",
                }
        out.append(entry)
    return out


class ProjectService:
    def __init__(self, db_path: str, clock: Optional[Callable[[], float]] = None):
        self.store = EventStore(db_path, clock or time.time)

    def close(self) -> None:
        self.store.close()

    # ------------------------------------------------------------ projects
    def create_project(self, name: str) -> dict:
        existing = self.store.events(name)
        if existing:
            raise ServiceError(f"project {name!r} already exists")
        ev = self.store.append(name, "project_created", {"name": name})
        return {"project": name, "event": ev}

    def _events(self, project: str, seq: Optional[int] = None) -> List[dict]:
        return self.store.events(project, seq)

    def _versions(self, events: List[dict]) -> List[dict]:
        return [e["payload"] | {"event_seq": e["seq"], "event_id": e["id"]}
                for e in events
                if e["type"] in ("input_imported", "version_revised")]

    def get_version(self, project: str, version_id: str) -> dict:
        for v in self._versions(self._events(project)):
            if v["version_id"] == version_id:
                return v
        raise ServiceError(f"unknown version {version_id!r}")

    # --------------------------------------------------------------- import
    def import_input(self, project: str, name: str, text: str,
                     fmt: str = INPUT_FORMAT,
                     annotations: Optional[List[dict]] = None) -> dict:
        if fmt != INPUT_FORMAT:
            raise ServiceError(f"unsupported input format {fmt!r}")
        events = self._events(project)
        if not events:
            self.create_project(project)
            events = self._events(project)
        if any(e["type"] == "input_imported" for e in events):
            raise ServiceError("original input is immutable and already exists; "
                               "use revise() to create a new version")
        try:
            fragments = split_fragments(text, name)
        except LanguageError as e:
            raise ServiceError(f"format error: {e.message}", [e.to_diagnostic()])
        norm_ann = normalize_annotations(text, annotations or [])
        version_id = f"v{self.store._next_seq()}"
        payload = {
            "version_id": version_id, "parent": None, "name": name,
            "format": fmt, "raw_text": text, "fragments": fragments,
            "params": dict(DEFAULT_PARAMS),
            "annotations_raw": annotations or [],
            "annotations": norm_ann,
        }
        ev = self.store.append(project, "input_imported", payload)
        return {"version_id": version_id, "event_id": ev["id"],
                "fragments": fragments, "annotations": norm_ann}

    def revise(self, project: str, parent_version: str,
               fragment_updates: Optional[Dict[str, Optional[str]]] = None,
               param_updates: Optional[dict] = None,
               rationale: str = "") -> dict:
        parent = self.get_version(project, parent_version)
        fragments = {fid: dict(f) for fid, f in parent["fragments"].items()}
        for fid, new_text in (fragment_updates or {}).items():
            if new_text is None:
                fragments.pop(fid, None)
                continue
            try:
                new_frags = split_fragments(new_text, f"{fid}@revise")
            except LanguageError as e:
                raise ServiceError(f"format error in revision for {fid}: "
                                   f"{e.message}", [e.to_diagnostic()])
            if len(new_frags) != 1:
                raise ServiceError(f"revision for {fid} must contain exactly "
                                   "one top-level form")
            kind, frag = next(iter(new_frags.items()))
            frag = dict(frag)
            if fid in fragments:
                frag["kind"] = fragments[fid]["kind"]
                frag["name"] = fragments[fid]["name"]
            fragments[fid] = frag
        params = dict(parent["params"])
        params.update(param_updates or {})
        version_id = f"v{self.store._next_seq()}"
        payload = {
            "version_id": version_id, "parent": parent_version,
            "name": parent["name"], "format": INPUT_FORMAT,
            "raw_text": parent["raw_text"], "fragments": fragments,
            "params": params,
            "annotations_raw": parent.get("annotations_raw", []),
            "annotations": parent.get("annotations", []),
            "changes": {"fragment_updates": fragment_updates or {},
                        "param_updates": param_updates or {}},
            "rationale": rationale,
        }
        ev = self.store.append(project, "version_revised", payload)
        return {"version_id": version_id, "event_id": ev["id"],
                "parent": parent_version, "params": params}

    # ------------------------------------------------------- decisions state
    @staticmethod
    def _undone_ids(events: List[dict]) -> set:
        return {e["payload"]["target_event_id"] for e in events
                if e["type"] == "undo"}

    @staticmethod
    def _superseded_ids(events: List[dict]) -> set:
        out = set()
        for e in events:
            if e["type"] == "decision_made":
                out.update(e["payload"].get("supersedes", []))
        return out

    def active_decisions(self, events: List[dict],
                         target: Optional[str] = None) -> List[dict]:
        undone = self._undone_ids(events)
        superseded = self._superseded_ids(events)
        out = []
        for e in events:
            if e["type"] != "decision_made":
                continue
            if e["id"] in undone or e["id"] in superseded:
                continue
            if target is not None and e["payload"]["target"] != target:
                continue
            out.append({"decision_id": e["id"], "seq": e["seq"], **e["payload"]})
        return out

    def conflicts(self, events: List[dict]) -> List[dict]:
        resolved = {e["payload"]["conflict_id"] for e in events
                    if e["type"] == "conflict_resolved"}
        out = []
        for e in events:
            if e["type"] == "conflict_detected":
                cid = e["payload"]["conflict_id"]
                out.append({**e["payload"], "event_id": e["id"],
                            "status": "resolved" if cid in resolved else "open"})
        return out

    def effective_choices(self, events: List[dict]) -> Dict[str, str]:
        """node_key -> chosen option label, from accepted decisions.
        Targets with an open conflict stay pending (no choice applied)."""
        open_targets = {c["target"] for c in self.conflicts(events)
                        if c["status"] == "open"}
        choices: Dict[str, str] = {}
        for d in self.active_decisions(events):
            if d["kind"] != "accept" or not d["target"].startswith("cand:"):
                continue
            if d["target"] in open_targets:
                continue
            choices[d["target"][len("cand:"):]] = d["choice"]
        return choices

    def decide(self, project: str, target: str, kind: str,
               choice: Optional[str] = None, rationale: str = "",
               actor: str = "reviewer",
               resolves: Optional[str] = None) -> dict:
        if kind not in ("accept", "reject", "adjust"):
            raise ServiceError(f"unknown decision kind {kind!r}")
        events = self._events(project)
        payload = {"target": target, "kind": kind, "choice": choice,
                   "rationale": rationale, "actor": actor,
                   "resolves": resolves, "supersedes": []}
        conflict_id = None
        if resolves is not None:
            open_conf = {c["conflict_id"]: c for c in self.conflicts(events)
                         if c["status"] == "open"}
            if resolves not in open_conf:
                raise ServiceError(f"no open conflict {resolves!r}")
            payload["supersedes"] = open_conf[resolves]["decision_ids"]
        else:
            active = self.active_decisions(events, target)
            if active and any(d["choice"] != choice for d in active):
                conflict_id = content_id("conf", {"target": target,
                                                  "seq": self.store._next_seq()})
        ev = self.store.append(project, "decision_made", payload)
        result = {"decision_id": ev["id"], "event": ev}
        if resolves is not None:
            self.store.append(project, "conflict_resolved",
                              {"conflict_id": resolves,
                               "decision_id": ev["id"]})
            result["resolved"] = resolves
        elif conflict_id is not None:
            ids = [d["decision_id"] for d in self.active_decisions(events, target)]
            ids.append(ev["id"])
            self.store.append(project, "conflict_detected",
                              {"conflict_id": conflict_id, "target": target,
                               "decision_ids": ids})
            result["conflict_id"] = conflict_id
        return result

    def undo(self, project: str, target_event_id: str,
             rationale: str = "") -> dict:
        events = self._events(project)
        if not any(e["id"] == target_event_id for e in events):
            raise ServiceError(f"unknown event {target_event_id!r}")
        if target_event_id in self._undone_ids(events):
            raise ServiceError(f"event {target_event_id!r} already undone")
        ev = self.store.append(project, "undo",
                               {"target_event_id": target_event_id,
                                "rationale": rationale})
        return {"undo_id": ev["id"], "event": ev}
