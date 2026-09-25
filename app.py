"""FastAPI entrypoint for the local-first macro-scope mirror.

Run:  .venv/bin/uvicorn app:app --host 127.0.0.1 --port 5396
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from macrocore.casepack import CaseRunner, load_case_pack
from macrocore.pipeline import (ImportFormatError, compare_expansions,
                                compute_expansion, import_source)
from macrocore.store import ConflictError, Store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DEFAULT_DB = os.environ.get("MACRO_MIRROR_DB", os.path.join(BASE_DIR, "macro_mirror.db"))


class ImportBody(BaseModel):
    content: str
    kind: str = "source"
    label: Optional[str] = None
    parent_version: Optional[str] = None
    params: Optional[dict] = None


class DecisionBody(BaseModel):
    target: str
    verdict: str
    rationale: str = ""
    reviewer: str = "reviewer"
    derived_id: Optional[str] = None
    params: Optional[dict] = None


class UndoBody(BaseModel):
    event_id: str
    reviewer: str = "reviewer"
    note: str = ""


class RecomputeBody(BaseModel):
    params: dict
class RecomputeBody(BaseModel):
    params: dict


def create_app(db_path: Optional[str] = None, clock=None) -> FastAPI:
    app = FastAPI(title="macro-scope-mirror")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    holder = {"store": Store(db_path or DEFAULT_DB, clock=clock),
              "path": db_path or DEFAULT_DB, "clock": clock}

    def store() -> Store:
        return holder["store"]

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/api/state")
    def state():
        s = store()
        return {
            "versions": s.list_versions(),
            "derived": s.list_derived(include_stale=True),
            "decisions": s.effective_decisions(),
            "events": s.events(),
            "raw": s.list_raw(),
        }

    @app.get("/api/raw/{raw_id}")
    def get_raw(raw_id: str):
        try:
            return store().get_raw(raw_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    @app.post("/api/import")
    def import_body(body: ImportBody):
        try:
            return import_source(store(), body.content, kind=body.kind,
                                 label=body.label,
                                 parent_version=body.parent_version,
                                 params=body.params)
        except ImportFormatError as exc:
            return JSONResponse(status_code=422,
                                content={"error": "format", "diagnostics": exc.diagnostics})
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    @app.post("/api/versions/{version_id}/recompute")
    def recompute(version_id: str, body: RecomputeBody):
        try:
            return compute_expansion(store(), version_id, body.params)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    @app.get("/api/derived")
    def list_derived(include_stale: bool = True, kind: Optional[str] = None):
        return store().list_derived(kind=kind, include_stale=include_stale)

    @app.get("/api/derived/{derived_id}")
    def get_derived(derived_id: str):
        try:
            return store().get_derived(derived_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    @app.get("/api/compare")
    def compare(a: str = Query(...), b: str = Query(...)):
        try:
            return compare_expansions(store(), a, b)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    @app.post("/api/decisions")
    def decide(body: DecisionBody):
        try:
            return store().add_decision(body.target, body.verdict, body.rationale,
                                        body.reviewer, body.derived_id, body.params)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/api/resolve")
    def resolve(body: DecisionBody):
        return store().resolve_conflict(body.target, body.verdict, body.rationale,
                                        body.reviewer, body.derived_id)

    @app.post("/api/undo")
    def undo(body: UndoBody):
        try:
            return store().undo_event(body.event_id, body.reviewer, body.note)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    @app.get("/api/events")
    def events():
        return store().events()

    @app.get("/api/events/{event_id}/view")
    def view_at(event_id: str):
        try:
            return store().view_at(event_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    @app.get("/api/casepack")
    def casepack_info():
        pack = load_case_pack()
        return {"pack_id": pack["pack_id"], "title": pack["title"],
                "narrative": pack["narrative"],
                "fragments": {k: {"title": v["title"], "acceptance": v["acceptance"]}
                              for k, v in pack["fragments"].items()},
                "coordinate_basis": pack["coordinate_basis"], "units": pack["units"]}

    @app.get("/api/casepack/fragment/{name}")
    def casepack_fragment(name: str):
        pack = load_case_pack()
        frag = pack["fragments"].get(name)
        if not frag:
            raise HTTPException(404, f"unknown fragment {name}")
        return {"name": name, **frag}

    @app.post("/api/casepack/run")
    def casepack_run():
        pack = load_case_pack()
        runner = CaseRunner(store(), pack)
        return runner.run()

    @app.get("/api/export")
    def export_dump():
        return store().export_dump()

    @app.post("/api/import-dump")
    def import_dump(dump: dict):
        try:
            return store().import_dump(dump)
        except ConflictError as exc:
            return JSONResponse(status_code=409, content={"error": str(exc)})
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/api/db/delete")
    def delete_db():
        s = store()
        result = s.delete_database()
        holder["store"] = Store(holder["path"], clock=holder["clock"])
        return result

    return app


app = create_app()
