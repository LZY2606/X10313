"""Syntax objects, source spans, scope sets and the expansion graph.

A :class:`Stx` is an immutable syntax object:

* ``value``    – the datum (id text / number / string / bool) or ``None``
* ``kind``     – seq | id | number | string | bool | raw
* ``lexeme``   – original source text (preserved, never normalised away)
* ``scopes``   – frozenset of scope labels (the *scope set*)
* ``span``     – original :class:`Span` with explicit coordinate basis
* ``parents``  – stable ids of the syntax objects this node was derived from
* ``reason``   – machine readable transform that created this node
* ``flags``    – annotations such as introduced / captured / ambiguous

Every node gets a stable id from the :class:`GraphBuilder` in deterministic
construction order, so a full replay reproduces every identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

from .ids import digest

SPAN_UNIT = "character offsets; line/column are 1-based, counted in Unicode code points"


@dataclass(frozen=True)
class Span:
    file: str
    offset: int
    line: int
    col: int
    length: int
    unit: str = SPAN_UNIT

    @property
    def end_offset(self) -> int:
        return self.offset + self.length

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file, "offset": self.offset, "line": self.line,
            "col": self.col, "length": self.length,
            "end_offset": self.end_offset, "unit": self.unit,
        }


@dataclass(frozen=True)
class Token:
    index: int
    kind: str          # ( ) { } ' ` , ,@ atom
    value: str
    span: Span

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": f"tok{self.index:04d}", "index": self.index,
            "kind": self.kind, "value": self.value, "span": self.span.as_dict(),
        }


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: str = "error"            # error | warning | unknown
    span: Optional[Span] = None
    node_id: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        out = {"code": self.code, "message": self.message,
               "severity": self.severity}
        if self.span is not None:
            out["span"] = self.span.as_dict()
        if self.node_id is not None:
            out["node_id"] = self.node_id
        return out


@dataclass(frozen=True)
class Stx:
    id: str
    kind: str
    value: Any
    lexeme: str
    scopes: frozenset[str]
    span: Optional[Span]
    parents: tuple[str, ...]
    reason: str
    seq: tuple["Stx", ...] = ()
    bracket: str = "()"
    flags: tuple[str, ...] = ()

    # -- basic views -------------------------------------------------------
    def is_id(self, name: Optional[str] = None) -> bool:
        return self.kind == "id" and (name is None or self.value == name)

    def datum_label(self) -> str:
        if self.kind == "seq":
            return f"({ ' '.join(c.datum_label() for c in self.seq) })"
        return str(self.lexeme)

    def to_plain(self) -> Any:
        """Datum view, dropping scopes (kept elsewhere)."""
        if self.kind == "seq":
            return [c.to_plain() for c in self.seq]
        if self.kind == "raw":
            return {"#raw": self.lexeme}
        return self.value

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id, "kind": self.kind, "value": self.value,
            "lexeme": self.lexeme, "scopes": sorted(self.scopes),
            "span": self.span.as_dict() if self.span else None,
            "parents": list(self.parents), "reason": self.reason,
            "bracket": self.bracket, "flags": list(self.flags),
        }
        if self.kind == "seq":
            out["items"] = [c.id for c in self.seq]
        return out


@dataclass
class Edge:
    id: str
    kind: str
    src: str
    dst: str
    detail: str


class GraphBuilder:
    """Allocates stable node/edge ids and records transform steps."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.nodes: dict[str, Stx] = {}
        self.edges: list[Edge] = []
        self.steps: list[dict[str, Any]] = []
        self._n = 0
        self._scope_counter = 0

    # -- nodes -------------------------------------------------------------
    def node(self, *, kind: str, value: Any = None, lexeme: str = "",
             scopes: Any = frozenset(), span: Optional[Span] = None,
             parents: tuple[str, ...] = (), reason: str,
             seq: tuple[Stx, ...] = (), bracket: str = "()",
             flags: tuple[str, ...] = ()) -> Stx:
        self._n += 1
        nid = f"{self.run_id}-n{self._n:05d}"
        stx = Stx(id=nid, kind=kind, value=value, lexeme=lexeme or str(value),
                  scopes=frozenset(scopes), span=span,
                  parents=tuple(parents), reason=reason, seq=tuple(seq),
                  bracket=bracket, flags=tuple(flags))
        self.nodes[nid] = stx
        for i, pid in enumerate(parents):
            detail = reason if len(parents) == 1 else f"{reason} [parent {i}]"
            eid = digest(["edge", nid, pid, reason, i], "e")
            self.edges.append(Edge(eid, "derive", pid, nid, detail))
        return stx

    def clone(self, src: Stx, *, scopes: Optional[Any] = None,
              parents: Optional[tuple[str, ...]] = None,
              reason: Optional[str] = None, span: Optional[Span] = ...,
              flags: Optional[tuple[str, ...]] = None,
              value: Any = ..., seq: Optional[tuple[Stx, ...]] = ...) -> Stx:
        """Derive a new node from ``src`` (a transform step)."""
        new_scopes = src.scopes if scopes is None else frozenset(scopes)
        new_parents = (src.id,) if parents is None else tuple(parents)
        new_reason = reason or "transform"
        new_span = src.span if span is ... else span
        new_flags = src.flags if flags is None else tuple(flags)
        new_value = src.value if value is ... else value
        new_seq = src.seq if seq is ... else tuple(seq or ())
        return self.node(kind=src.kind, value=new_value, lexeme=src.lexeme,
                         scopes=new_scopes, span=new_span,
                         parents=new_parents, reason=new_reason,
                         seq=new_seq, bracket=src.bracket, flags=new_flags)

    # -- scopes ------------------------------------------------------------
    def fresh_binding(self, symbol: str) -> str:
        self._scope_counter += 1
        return f"b:{symbol}.{self._scope_counter:03d}"

    def fresh_intro(self, tag: str) -> str:
        self._scope_counter += 1
        return f"i:{tag}.{self._scope_counter:03d}"

    def fresh_mark(self, tag: str) -> str:
        self._scope_counter += 1
        return f"m:{tag}.{self._scope_counter:03d}"

    # -- steps / events ----------------------------------------------------
    def log_step(self, kind: str, **payload: Any) -> dict[str, Any]:
        step = {"seq": len(self.steps), "kind": kind, **payload}
        self.steps.append(step)
        return step

    def graph_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "nodes": [n.as_dict() for n in self.nodes.values()],
            "edges": [{"id": e.id, "kind": e.kind, "src": e.src,
                       "dst": e.dst, "detail": e.detail} for e in self.edges],
            "steps": self.steps,
        }


def add_scope(builder: GraphBuilder, stx: Stx, scope: str, reason: str,
              deep: bool = False) -> Stx:
    """Return ``stx`` with ``scope`` toggled on (shallow unless ``deep``)."""
    if not deep:
        return builder.clone(stx, scopes=stx.scopes | {scope}, reason=reason)

    def walk(node: Stx) -> Stx:
        if node.kind == "seq":
            items = tuple(walk(c) for c in node.seq)
            return builder.clone(node, scopes=node.scopes | {scope},
                                 seq=items, reason=reason)
        return builder.clone(node, scopes=node.scopes | {scope}, reason=reason)

    return walk(stx)


def flip_scope(builder: GraphBuilder, stx: Stx, mark: str) -> Stx:
    """Toggle a macro mark over the whole subtree (hygiene algorithm)."""
    def walk(node: Stx) -> Stx:
        scopes = node.scopes ^ {mark}
        if node.kind == "seq":
            items = tuple(walk(c) for c in node.seq)
            return builder.clone(node, scopes=scopes, seq=items,
                                 reason=f"flip-mark {mark}")
        return builder.clone(node, scopes=scopes, reason=f"flip-mark {mark}")

    return walk(stx)
