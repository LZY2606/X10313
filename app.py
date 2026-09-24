"""macroscope review server: local-first macro hygiene & provenance workbench."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from macroscope import casepack
from macroscope.expander import expand_program
from macroscope.lexer import LexError, ParseError, lex, parse
from macroscope.store import Store

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"


def create_app(store: Store) -> FastAPI:
    app = FastAPI(title="macroscope")
    app.state.store = store

    def expand_and_record(input_id: str) -> dict:
        inp = store.get_input(input_id)
        forms = parse(inp["content"])
        result = expand_program(forms, budget=store.params.get("budget", 64))
        ev = store.add_event(
            "expand",
            {"input": input_id, "status": result["status"],
             "steps_used": result["steps_used"], "budget": result["budget"]},
            sources=[input_id])
        return store.add_artifact("expansion", result, sources=[input_id],
                                  event_seq=ev["seq"])

    app.state.expand_and_record = expand_and_record

    # -- queries ------------------------------------------------------------
    @app.get("/api/state")
    def state():
        return {
            "inputs": store.list_inputs(),
            "events": store.list_events(visible_only=True),
            "event_count_total": len(store.list_events(visible_only=False)),
            "artifacts": [{k: v for k, v in a.items() if k != "payload"}
                          for a in store.list_artifacts()],
            "params": store.params,
            "cursor": store.cursor,
            "decisions": store.decisions(),
            "conflicts": store.conflicts(),
            "algo_version": store.algo_version,
        }

    @app.get("/api/inputs/{input_id}/tokens")
    def tokens(input_id: str):
        inp = store.get_input(input_id)
        try:
            toks = lex(inp["content"])
        except LexError as e:  # cannot happen: inputs are validated on import
            raise HTTPException(422, e.to_diag())
        return {"input": input_id, "tokens": [t.to_dict() for t in toks]}

    @app.get("/api/artifacts/{artifact_id}")
    def artifact(artifact_id: str):
        try:
            return store.get_artifact(artifact_id)
        except KeyError:
            raise HTTPException(404, "unknown artifact")

    @app.get("/api/compare")
    def compare(a: str, b: str):
        try:
            art_a, art_b = store.get_artifact(a), store.get_artifact(b)
        except KeyError as e:
            raise HTTPException(404, str(e))
        ra = {r["node"]: r for r in art_a["payload"]["resolutions"]}
        rb = {r["node"]: r for r in art_b["payload"]["resolutions"]}
        diff = []
        for node in sorted(set(ra) | set(rb)):
            xa, xb = ra.get(node), rb.get(node)
            entry = {"node": node,
                     "name": (xa or xb)["name"],
                     "a": (xa or {}).get("chosen"), "b": (xb or {}).get("chosen")}
            if xa is None:
                entry["kind"] = "only-in-b"
            elif xb is None:
                entry["kind"] = "only-in-a"
            elif xa["chosen"] != xb["chosen"]:
                entry["kind"] = "chosen-changed"
            else:
                entry["kind"] = "same"
            diff.append(entry)
        return {"a": a, "b": b, "diff": diff}

    # -- commands -------------------------------------------------------------
    class ImportBody(BaseModel):
        content: str
        parent: str | None = None

    @app.post("/api/inputs")
    def import_input(body: ImportBody):
        try:
            lex(body.content)
            parse(body.content)
        except LexError as e:
            raise HTTPException(422, detail={"diagnostics": [e.to_diag()]})
        except ParseError as e:
            raise HTTPException(422, detail={"diagnostics": [e.to_diag()]})
        try:
            inp = store.add_input(body.content, parent=body.parent)
        except KeyError as e:
            raise HTTPException(404, str(e))
        store.add_event("import", {"input": inp["id"], "parent": body.parent,
                                   "sha": inp["sha"]}, sources=[inp["id"]])
        return inp

    class ExpandBody(BaseModel):
        input_id: str

    @app.post("/api/expand")
    def expand(body: ExpandBody):
        try:
            return expand_and_record(body.input_id)
        except KeyError:
            raise HTTPException(404, "unknown input")

    class DecisionBody(BaseModel):
        target: str
        verdict: str
        rationale: str = ""

    @app.post("/api/decisions")
    def decide(body: DecisionBody):
        if body.verdict not in ("accept", "reject", "adjust"):
            raise HTTPException(422, "verdict must be accept|reject|adjust")
        return store.add_event("decision", body.model_dump(), sources=[body.target])

    class UndoBody(BaseModel):
        event_seq: int

    @app.post("/api/undo")
    def undo(body: UndoBody):
        try:
            store.get_event(body.event_seq)
        except KeyError:
            raise HTTPException(404, "unknown event")
        return store.add_event("undo", {"event_seq": body.event_seq},
                               sources=[f"ev-{body.event_seq:06d}"])

    class ParamsBody(BaseModel):
        budget: int

    @app.post("/api/params")
    def set_params(body: ParamsBody):
        if not (1 <= body.budget <= 100000):
            raise HTTPException(422, "budget out of range")
        return store.set_params({"budget": body.budget})

    class CursorBody(BaseModel):
        seq: int

    @app.post("/api/cursor")
    def set_cursor(body: CursorBody):
        total = len(store.list_events(visible_only=False))
        if not (0 <= body.seq <= total):
            raise HTTPException(422, "cursor out of range")
        store.set_cursor(body.seq)
        return {"cursor": store.cursor}

    @app.get("/api/export")
    def export():
        return store.export()

    @app.post("/api/import_archive")
    def import_archive(archive: dict):
        try:
            counts = store.import_archive(archive)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"imported": counts, "cursor": store.cursor}

    @app.post("/api/casepack")
    def load_casepack():
        return casepack.load_casepack(store, expand_and_record)

    @app.get("/api/samples")
    def samples():
        return {"samples": casepack.SAMPLES}

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


def _default_store() -> Store:
    db = os.environ.get("MACROSCOPE_DB", str(BASE_DIR / "macroscope.db"))
    return Store(db)


app = create_app(_default_store())
