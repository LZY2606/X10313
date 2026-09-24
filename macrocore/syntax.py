"""Syntax objects: tokens, spans, scope sets, deterministic id generation.

Every value shown in the UI carries a stable identity.  Identities are
deterministic: the same input expanded with the same algorithm version and
parameters always yields the same node ids, so exported archives can be
re-imported without identity drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


@dataclass(frozen=True)
class Span:
    """A source span.  ``basis`` is the coordinate basis of the raw
    measurement; importers keep the raw representation and only the derived
    layer normalises (see macrocore.quantities)."""

    start: int
    end: int
    line: int
    col: int
    basis: str = "char-offset"

    def to_json(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "line": self.line,
            "col": self.col,
            "basis": self.basis,
        }

    @staticmethod
    def from_json(d: dict) -> "Span":
        return Span(d["start"], d["end"], d["line"], d["col"], d.get("basis", "char-offset"))


@dataclass(frozen=True)
class Token:
    kind: str  # lparen | rparen | ident | int | string
    text: str
    span: Span

    def to_json(self) -> dict:
        return {"kind": self.kind, "text": self.text, "span": self.span.to_json()}


class IdGen:
    """Deterministic id generator; one per expansion run."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.counter = 0

    def next(self, tag: str) -> str:
        self.counter += 1
        return f"{self.prefix}:{tag}{self.counter}"


@dataclass
class Syntax:
    """A syntax object: datum + scope set + source span + provenance."""

    sid: str
    datum: Any  # str (ident) | int | str-literal | list[Syntax]
    scopes: frozenset
    span: Optional[Span] = None
    origin: Optional[str] = None  # sid of the syntax this node derives from
    step: Optional[str] = None  # expansion step id that produced this node
    kind: str = "datum"  # datum | ident | int | string | list

    def is_ident(self) -> bool:
        return self.kind == "ident"

    def to_json(self) -> dict:
        if isinstance(self.datum, list):
            datum: Any = [c.sid for c in self.datum]
        else:
            datum = self.datum
        return {
            "sid": self.sid,
            "datum": datum,
            "kind": self.kind,
            "scopes": sorted(self.scopes),
            "span": self.span.to_json() if self.span else None,
            "origin": self.origin,
            "step": self.step,
        }


def flip_scope(stx: Syntax, scope: str, idgen: IdGen) -> Syntax:
    """Flip ``scope`` on every node of the tree (add if absent, remove if
    present).  Produces new nodes whose ``origin`` points at the source node."""

    def go(s: Syntax) -> Syntax:
        datum = s.datum
        if isinstance(datum, list):
            datum = [go(c) for c in datum]
        new = Syntax(
            sid=idgen.next("n"),
            datum=datum,
            scopes=frozenset(set(s.scopes) ^ {scope}),
            span=s.span,
            origin=s.sid,
            step=s.step,
            kind=s.kind,
        )
        return new

    return go(stx)


def add_scope(stx: Syntax, scope: str, idgen: IdGen) -> Syntax:
    """Add ``scope`` to every node of the tree (no-op where already present)."""

    def go(s: Syntax) -> Syntax:
        datum = s.datum
        if isinstance(datum, list):
            datum = [go(c) for c in datum]
        return Syntax(
            sid=idgen.next("n"),
            datum=datum,
            scopes=frozenset(set(s.scopes) | {scope}),
            span=s.span,
            origin=s.sid,
            step=s.step,
            kind=s.kind,
        )

    return go(stx)


def datum_repr(stx: Syntax) -> Any:
    """Plain datum without scope/context information (for display)."""
    if isinstance(stx.datum, list):
        return [datum_repr(c) for c in stx.datum]
    return stx.datum
