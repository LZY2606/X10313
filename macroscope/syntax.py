"""Syntax objects: datum + scope set + source span + provenance origin.

A syntax object never loses its original token identity: every node carries
`nid` (stable node id), `span` (source span, may be None for generated
nodes), `origin` ("src" or the expansion step id that introduced it) and a
set of lexical scopes. Capture is decided purely by scope-set relations,
never by string renaming.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    line: int
    col: int

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "line": self.line, "col": self.col}


@dataclass(frozen=True)
class SIdent:
    name: str
    scopes: frozenset
    span: Optional[Span]
    origin: str
    nid: str


@dataclass(frozen=True)
class SInt:
    value: int
    scopes: frozenset
    span: Optional[Span]
    origin: str
    nid: str


@dataclass(frozen=True)
class SList:
    items: tuple
    scopes: frozenset
    span: Optional[Span]
    origin: str
    nid: str


Syn = Union[SIdent, SInt, SList]


def with_scope(s: Syn, scope: str) -> Syn:
    """Add a scope to a whole syntax tree (Flatt-style scope propagation)."""
    if isinstance(s, SList):
        return replace(
            s,
            scopes=s.scopes | {scope},
            items=tuple(with_scope(i, scope) for i in s.items),
        )
    return replace(s, scopes=s.scopes | {scope})


def syn_to_json(s: Syn) -> dict:
    base = {
        "nid": s.nid,
        "scopes": sorted(s.scopes),
        "span": s.span.to_dict() if s.span else None,
        "origin": s.origin,
    }
    if isinstance(s, SIdent):
        return {**base, "type": "ident", "name": s.name}
    if isinstance(s, SInt):
        return {**base, "type": "int", "value": s.value}
    return {**base, "type": "list", "items": [syn_to_json(i) for i in s.items]}


def syn_to_str(s: Syn) -> str:
    if isinstance(s, SIdent):
        return s.name
    if isinstance(s, SInt):
        return str(s.value)
    return "(" + " ".join(syn_to_str(i) for i in s.items) + ")"
