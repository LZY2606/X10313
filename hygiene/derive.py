"""Versioned derivation layer.

Every derived artifact records:
  * algorithm version (hygiene-algo-v1) and parameter fingerprint;
  * direct sources: exact input_version_id, event_version_id and the
    parameter revision event it was computed with;
  * fine-grained source edges from each produced object to the material
    (tokens/scopes/calls/steps) it depends on.

Invalidation is local: changing one direct source only marks the reachable
part of the dependency graph stale. Reachable sets are computed with a NumPy
boolean adjacency matrix (one of the project's numeric workloads).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .clock import Clock
from .syntax import SyntaxRegistry
from .parser import parse, ParseError
from .tokens import LexError
from .expand import Expander, BudgetExceeded, ExpandError
from . import eval_core

ALGORITHM_VERSION = "hygiene-algo-v1"


def fingerprint(params: dict) -> str:
    canonical = json.dumps(params, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return "fp-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def validate_source_text(src: str) -> list[dict]:
    """Pre-validate copy-pasted samples. Returns explicit diagnostics list."""
    diags = []
    if src == "":
        diags.append({"code": "empty", "msg": "source text is empty", "span": None})
    open_paren = 0
    open_brack = 0
    line_nl = []
    for idx, ch in enumerate(src):
        if ch == "\n":
            line_nl.append(idx)
    try:
        from .tokens import tokenize
        toks = tokenize(src)
    except LexError as exc:
        diags.append({"code": "lex", "msg": exc.msg, "span": exc.span.as_dict()})
        return diags
    for t in toks:
        if t.kind == "lparen":
            open_paren += 1
        elif t.kind == "rparen":
            open_paren -= 1
        elif t.kind == "lbracket":
            open_brack += 1
        elif t.kind == "rbracket":
            open_brack -= 1
        if open_paren < 0 or open_brack < 0:
            diags.append({"code": "unbalanced",
                          "msg": "closing bracket without a matching opener",
                          "span": t.span.as_dict()})
            return diags
    if open_paren != 0 or open_brack != 0:
        last = toks[-1] if toks else None
        diags.append({"code": "unbalanced",
                      "msg": "unclosed list before end of input",
                      "span": last.span.as_dict() if last else None})
    return diags


def run_derivation(source_text: str, params: dict, *, clock: Clock,
                   clause_overrides=None, tag: str = "d0") -> dict:
    """Parse -> expand -> evaluate one input version. Pure w.r.t. clock."""
    budget = int(params.get("budget", 64))
    reg = SyntaxRegistry(tag=tag)
    expander = Expander(reg, clock, budget=budget,
                        clause_overrides=clause_overrides or {},
                        algorithm_version=ALGORITHM_VERSION)
    expander.seed_primitives()

    diagnostics = []
    expanded_forms, tree, tokens = [], None, []
    try:
        tokens, forms = parse(source_text, reg, expander.program_scope)
        expanded_forms, tree = expander.expand_program(forms)
    except (LexError, ParseError) as exc:
        span = getattr(exc, "span", None)
        diagnostics.append({"code": "format", "msg": str(exc),
                            "span": span.as_dict() if span else None,
                            "fatal": True})
    except ExpandError as exc:
        diagnostics.append({"code": "expand", "msg": exc.msg,
                            "span": exc.obj.span.as_dict() if exc.obj and exc.obj.span
                            else None, "fatal": True})

    eval_results = []
    eval_error = None
    eval_trace = []
    if not diagnostics and expander.status == "ok":
        try:
            vals, eval_trace = eval_core.evaluate(expanded_forms, expander)
            eval_results = [_render_value(v) for v in vals]
        except eval_core.EvalError as exc:
            eval_error = {
                "msg": exc.msg,
                "oid": exc.obj.oid if exc.obj is not None else None,
                "span": exc.obj.span.as_dict() if exc.obj is not None and
                exc.obj.span else None,
                "birth": exc.obj.birth.as_dict() if exc.obj is not None and
                exc.obj.birth else None,
            }

    if tree is None:
        from .expand import TreeNode
        tree = TreeNode("tr-root", "program (parse failed)", "root", "", None, [],
                        {"status": "parse_failed"})

    artifact = expander.artifact(tokens, source_text, tree)
    artifact["eval"] = {"results": eval_results, "error": eval_error,
                        "trace": eval_trace}
    artifact["diagnostics"] = diagnostics
    artifact["params"] = dict(params)
    artifact["param_fingerprint"] = fingerprint(params)
    artifact["dependency_edges"] = build_dependency_edges(artifact)
    return artifact


def _render_value(v):
    if isinstance(v, eval_core.Closure):
        return {"type": "closure", "params": v.params}
    if isinstance(v, list) and v and v[0] == "pair":
        out = []
        cur = v
        while isinstance(cur, list) and cur and cur[0] == "pair":
            out.append(_render_value(cur[1]))
            cur = cur[2]
        return {"type": "list", "items": out,
                "tail": _render_value(cur) if cur != ["null"] else "()"}
    if isinstance(v, list) and v == ["null"]:
        return {"type": "list", "items": [], "tail": "()"}
    return {"type": "atom", "value": v}


def build_dependency_edges(artifact: dict) -> list[dict]:
    """Fine-grained direct-source edges (object -> material it depends on)."""
    edges: list[dict] = []

    def add(src: str, dst: str, relation: str):
        edges.append({"src": src, "dst": dst, "relation": relation})

    for s in artifact["syntax_objects"]:
        birth = s.get("birth") or {}
        tid = birth.get("token_id")
        if birth.get("kind") == "parsed" and tid is not None:
            add(s["oid"], f"token:{tid}", "syntax<-token")
        if birth.get("call_oid"):
            add(s["oid"], birth["call_oid"], f"syntax<-{birth['kind']}")
    for call in artifact["macro_calls"]:
        add(call["call_oid"], f"macro-def:{call['macro_binding_id']}",
            "call<-definition")
        add(call["call_oid"], call["use_oid"], "call<-use-site")
        for cand in call["candidates"]:
            add(call["call_oid"],
                f"{call['definition_oid']}#clause{cand['clause']}",
                "candidate")
    for edge in artifact["binding_edges"]:
        for s in edge["scopes"]:
            add(edge["edge_id"], f"scope:{s}", "binding<-scope")
        if edge.get("binding_id"):
            add(edge["edge_id"], f"binding:{edge['binding_id']}", "binding<-binding")
    for ev in artifact.get("context_choices", []):
        add(ev["recipient_oid"], ev["donor_oid"], "datum->syntax-context")
    return edges


def reachable_matrix(edges: list[dict]) -> tuple[np.ndarray, list[str]]:
    nodes: list[str] = []
    index = {}

    def nid(x: str) -> int:
        if x not in index:
            index[x] = len(nodes)
            nodes.append(x)
        return index[x]

    pairs = [(nid(e["src"]), nid(e["dst"])) for e in edges]
    n = len(nodes)
    mat = np.zeros((n, n), dtype=bool)
    for a, b in pairs:
        mat[a, b] = True
    # reflexive-transitive closure (Warshall; boolean workload stays local)
    closure = mat.copy()
    for k in range(n):
        closure |= np.outer(closure[:, k], closure[k, :])
    for i in range(n):
        closure[i, i] = True
    return closure, nodes


def affected_by(edges: list[dict], changed_materials: list[str]) -> list[str]:
    """Return every derived object that directly/transitively depends on them."""
    closure, nodes = reachable_matrix(edges)
    index = {x: i for i, x in enumerate(nodes)}
    affected = set()
    for mat in changed_materials:
        if mat not in index:
            continue
        row = closure[index[mat], :]
        for j, flag in enumerate(row):
            if flag:
                affected.add(nodes[j])
    return sorted(affected)
