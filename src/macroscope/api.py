"""FastAPI application: local-first review workbench.

Endpoints cover the full workflow: create/import projects, submit immutable
input revisions, compute derived artifacts, record/undo decisions, inspect
events/checkpoints, compare two macro versions, run the counterexample
archive, and export/import the whole local database payload.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .cases import CASE_PACK
from .jsonio import dumps_pretty
from .service import replay, seed_case_project
from .store import Store

DB_PATH = os.environ.get("MACROSCOPE_DB",
                         os.path.join(os.getcwd(), "macroscope.db"))
STATIC_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "static"))

app = FastAPI(title="Macroscope 宏作用域镜", version="1.0.0")


def get_store() -> Store:
    return Store(DB_PATH)


class ProjectIn(BaseModel):
    name: str


class InputIn(BaseModel):
    text: str
    label: str = "input.scm"
    parent_version_id: Optional[str] = None


class ComputeIn(BaseModel):
    input_version_id: str
    macro_budget: int = 64
    seed: int = 20260925


class DecisionIn(BaseModel):
    target_type: str = "artifact"
    target_id: str
    verdict: str               # accept | reject | adjust
    note: str
    detail: dict[str, Any] = {}


class CompareIn(BaseModel):
    artifact_a: str
    artifact_b: str


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "db": DB_PATH, "seed": 20260925}


@app.get("/api/case-pack")
def case_pack() -> dict[str, Any]:
    return CASE_PACK


@app.post("/api/projects")
def create_project(body: ProjectIn) -> dict[str, Any]:
    s = get_store()
    try:
        return s.create_project(body.name)
    finally:
        s.close()


@app.post("/api/projects/seed")
def seed() -> dict[str, Any]:
    s = get_store()
    try:
        return seed_case_project(s)
    finally:
        s.close()


@app.get("/api/projects")
def list_projects() -> list[dict[str, Any]]:
    s = get_store()
    try:
        return s.list_projects()
    finally:
        s.close()


@app.get("/api/projects/{pid}")
def project_state(pid: str) -> dict[str, Any]:
    s = get_store()
    try:
        p = s.get_project(pid)
        if not p:
            raise HTTPException(404, "project not found")
        return {
            "project": p,
            "inputs": s.list_inputs(pid),
            "artifacts": s.list_artifacts(pid),
            "events": s.list_events(pid),
            "decisions": s.list_decisions(pid),
            "experiments": s.list_experiments(pid),
            "checkpoints": s.list_checkpoints(pid),
        }
    finally:
        s.close()


@app.post("/api/projects/{pid}/inputs")
def submit_input(pid: str, body: InputIn) -> dict[str, Any]:
    if not body.text.strip():
        raise HTTPException(422, "空输入：请粘贴小型样例后再提交")
    s = get_store()
    try:
        if not s.get_project(pid):
            raise HTTPException(404, "project not found")
        return s.submit_input(pid, body.text, body.label,
                              body.parent_version_id)
    finally:
        s.close()


@app.post("/api/projects/{pid}/compute")
def compute(pid: str, body: ComputeIn) -> dict[str, Any]:
    s = get_store()
    try:
        return s.compute_artifact(
            pid, body.input_version_id,
            {"macro_budget": body.macro_budget, "seed": body.seed})
    except KeyError:
        raise HTTPException(404, "input version not found")
    finally:
        s.close()


@app.get("/api/artifacts/{aid}")
def get_artifact(aid: str) -> dict[str, Any]:
    s = get_store()
    try:
        art = s.get_artifact(aid)
        if not art:
            raise HTTPException(404, "artifact not found")
        return art
    finally:
        s.close()


@app.post("/api/projects/{pid}/decisions")
def decision(pid: str, body: DecisionIn) -> dict[str, Any]:
    if body.verdict not in ("accept", "reject", "adjust"):
        raise HTTPException(422, "verdict must be accept|reject|adjust")
    s = get_store()
    try:
        return s.record_decision(pid, body.target_type, body.target_id,
                                 body.verdict, body.note, body.detail)
    finally:
        s.close()


@app.post("/api/projects/{pid}/decisions/undo")
def undo_decision(pid: str) -> dict[str, Any]:
    s = get_store()
    try:
        out = s.undo_last_decision(pid)
        if not out:
            raise HTTPException(404, "no decision to undo")
        return out
    finally:
        s.close()


@app.post("/api/projects/{pid}/checkpoints")
def checkpoint(pid: str, body: dict[str, Any]) -> dict[str, Any]:
    s = get_store()
    try:
        return s.add_checkpoint(pid, body.get("label", "checkpoint"))
    finally:
        s.close()


@app.post("/api/projects/{pid}/replay")
def do_replay(pid: str) -> dict[str, Any]:
    s = get_store()
    try:
        return replay(s, pid)
    finally:
        s.close()


@app.post("/api/compare")
def compare(body: CompareIn) -> dict[str, Any]:
    s = get_store()
    try:
        a = s.get_artifact(body.artifact_a)
        b = s.get_artifact(body.artifact_b)
        if not a or not b:
            raise HTTPException(404, "artifact not found")
        return compare_artifacts(a, b)
    finally:
        s.close()


@app.get("/api/projects/{pid}/export")
def export(pid: str) -> JSONResponse:
    s = get_store()
    try:
        data = s.export_project(pid)
        return JSONResponse(data,
                            headers={"Content-Disposition":
                                     f'attachment; filename="{pid}.json"'})
    finally:
        s.close()


@app.post("/api/import")
def import_project(body: dict[str, Any]) -> dict[str, Any]:
    s = get_store()
    try:
        try:
            pid = s.import_project(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"project_id": pid, "restored": True}
    finally:
        s.close()


@app.post("/api/projects/{pid}/wipe")
def wipe(pid: str) -> dict[str, Any]:
    s = get_store()
    try:
        s.wipe_project(pid)
        return {"wiped": pid}
    finally:
        s.close()


def compare_artifacts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Stable-id based diff of two macro versions/artifacts."""
    na = {n["id"]: n for n in a["graph"]["nodes"]}
    nb = {n["id"]: n for n in b["graph"]["nodes"]}
    ra, rb = a.get("resolutions", {}), b.get("resolutions", {})
    return {
        "a": {"id": a["id"], "params": a["params"],
              "status": a["expansion_status"],
              "input_version_id": a["input_version_id"]},
        "b": {"id": b["id"], "params": b["params"],
              "status": b["expansion_status"],
              "input_version_id": b["input_version_id"]},
        "same_input": a["input_version_id"] == b["input_version_id"],
        "params_equal": a["params"] == b["params"],
        "nodes_only_in_a": sorted(set(na) - set(nb)),
        "nodes_only_in_b": sorted(set(nb) - set(na)),
        "resolution_changes": sorted(
            {"id": k,
             "a": (ra.get(k, {}).get("status")),
             "b": (rb.get(k, {}).get("status"))}
            for k in set(ra) | set(rb)
            if ra.get(k, {}).get("status") !=
            rb.get(k, {}).get("status")),
        "call_counts": {"a": a["call_count"], "b": b["call_count"]},
        "generated_symbol_diff": {
            "a": a["generated_symbols"], "b": b["generated_symbols"]},
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
