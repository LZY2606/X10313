"""Local-first macro scope mirror: FastAPI application.

All review state lives in SQLite; reopening the browser or switching
browsers restores the same history. No external services are used.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import casepack
from macrocore import ALGO_VERSION, compare_results, expand_source, layout
from store import Store, replace_database

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _event_summary(event: dict) -> dict:
    payload = event.get("payload") or {}
    return {"id": event["id"], "seq": event["seq"], "kind": event["kind"],
            "status": event["status"],
            "algo_version": event.get("algo_version"),
            "params_fp": event.get("params_fp"),
            "depends_on": event.get("depends_on", []),
            "label": payload.get("label")
            or payload.get("stage") or payload.get("target") or "",
            "has_result": "result" in payload}


def create_app(db_path: Optional[str] = None, now=None) -> FastAPI:
    db_path = db_path or os.environ.get("GSB_DB", "gsb.sqlite3")
    app = FastAPI(title="macro-scope-mirror")
    app.state.db_path = db_path
    app.state.store = Store(db_path, now=now)

    def store() -> Store:
        return app.state.store

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/api/case")
    def get_case():
        case = casepack.load_case()
        return {"title": case["title"],
                "coordinate_basis": case["coordinate_basis"],
                "acceptance": case["acceptance"],
                "replay": case["replay"],
                "sources": {k: v for k, v in case["sources"].items()}}

    @app.post("/api/import-case")
    def import_case():
        case = casepack.load_case()
        ids = casepack.stage_case(store(), case)
        return {"ok": True, "ids": ids}

    class ImportBody(BaseModel):
        source: str
        label: str = "直接导入"

    @app.post("/api/import")
    def import_source(body: ImportBody):
        if not body.source.strip():
            raise HTTPException(422, "empty source")
        input_id = store().add_input(body.source)
        version_id = store().add_version(
            None, "source", body.label,
            {"input_id": input_id, "source": body.source})
        return {"input_id": input_id, "version_id": version_id}

    class ReviseBody(BaseModel):
        parent: str
        kind: str  # 'source' | 'params'
        label: str
        source: Optional[str] = None
        params: Optional[dict] = None

    @app.post("/api/revise")
    def revise(body: ReviseBody):
        parent = store().get_version(body.parent)
        if parent is None:
            raise HTTPException(404, f"unknown parent {body.parent}")
        if body.kind == "source":
            if body.source is None:
                raise HTTPException(422, "source revision requires source")
            input_id = store().add_input(body.source)
            payload = {"input_id": input_id, "source": body.source}
        elif body.kind == "params":
            payload = {"params": body.params or {}}
        else:
            raise HTTPException(422, "kind must be 'source' or 'params'")
        version_id = store().add_version(body.parent, body.kind,
                                         body.label, payload)
        stale = (store().supersede(body.parent)
                 if body.kind == "source" else [])
        return {"version_id": version_id, "stale_events": stale}

    class ExpandBody(BaseModel):
        version_id: str
        budget: int = 128
        macro_version: str = "v1"

    @app.post("/api/expand")
    def expand(body: ExpandBody):
        version = store().get_version(body.version_id)
        if version is None:
            raise HTTPException(404, f"unknown version {body.version_id}")
        source = version["payload"].get("source")
        if source is None:
            input_id = version["payload"].get("input_id")
            raise HTTPException(422, f"version has no source ({input_id})")
        try:
            result, _ = expand_source(source, budget=body.budget,
                                      macro_version=body.macro_version)
        except Exception as exc:  # lexer / format errors stay visible
            raise HTTPException(422, f"format error: {exc}") from exc
        event_id = store().record_event(
            "expansion",
            {"label": f"展开 {body.version_id} "
                      f"(budget={body.budget}, {body.macro_version})",
             "result": result},
            algo_version=ALGO_VERSION,
            params={"budget": body.budget,
                    "macro_version": body.macro_version},
            depends_on=[body.version_id])
        return {"event_id": event_id,
                "status": result["status"],
                "n_edges": len(result["edges"]),
                "n_diagnostics": len(result["diagnostics"])}

    @app.get("/api/state")
    def state():
        db = store()
        return {"versions": db.list_versions(),
                "events": [_event_summary(e) for e in db.list_events()],
                "conflicts": db.conflicts(),
                "active_decisions": [
                    {"id": d["id"], **d["payload"]}
                    for d in db.active_decisions()]}

    @app.get("/api/events/{event_id}")
    def event(event_id: str):
        db = store()
        try:
            return db.event_view(event_id)
        except KeyError:
            raise HTTPException(404, f"unknown event {event_id}")

    @app.get("/api/graph/{event_id}")
    def graph(event_id: str):
        event = store().get_event(event_id)
        if event is None:
            raise HTTPException(404, f"unknown event {event_id}")
        result = event["payload"].get("result")
        if result is None:
            raise HTTPException(422, f"{event_id} has no expansion graph")
        enriched = dict(result)
        enriched["layout"] = layout(result["nodes"], result["roots"])
        enriched["event_id"] = event_id
        enriched["event_status"] = event["status"]
        enriched["params_fp"] = event["params_fp"]
        enriched["algo_version"] = event["algo_version"]
        return enriched

    class DecisionBody(BaseModel):
        target: str
        verdict: str  # accept | reject | adjust
        note: str = ""
        params: dict = {}

    @app.post("/api/decide")
    def decide(body: DecisionBody):
        if body.verdict not in ("accept", "reject", "adjust"):
            raise HTTPException(422, "verdict must be accept/reject/adjust")
        event_id = store().decide(body.target, body.verdict,
                                  body.note, body.params)
        return {"event_id": event_id, "conflicts": store().conflicts()}

    class UndoBody(BaseModel):
        decision_id: str

    @app.post("/api/undo")
    def undo(body: UndoBody):
        try:
            event_id = store().undo(body.decision_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"event_id": event_id, "conflicts": store().conflicts()}

    @app.get("/api/compare")
    def compare(a: str, b: str):
        db = store()
        ea, eb = db.get_event(a), db.get_event(b)
        if ea is None or eb is None:
            raise HTTPException(404, "unknown expansion event")
        ra = ea["payload"].get("result")
        rb = eb["payload"].get("result")
        if ra is None or rb is None:
            raise HTTPException(422, "both events must be expansions")
        diff = compare_results(ra, rb)
        diff["a_event"] = a
        diff["b_event"] = b
        return diff

    @app.get("/api/export")
    def export_db():
        return store().export_db()

    class ImportDbBody(BaseModel):
        data: dict

    @app.post("/api/import-db")
    def import_db(body: ImportDbBody):
        path = app.state.db_path
        app.state.store.close()
        try:
            rebuilt = replace_database(path, body.data)
        except ValueError as exc:
            app.state.store = Store(path)
            raise HTTPException(422, str(exc)) from exc
        app.state.store = rebuilt
        return {"ok": True,
                "counts": {"inputs": len(body.data.get("inputs", [])),
                           "versions": len(body.data.get("versions", [])),
                           "events": len(body.data.get("events", []))}}

    return app


app = create_app()
