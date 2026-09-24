"""Syntax objects: datum + scope set + source span + provenance chain.

Every identifier/literal/list the expander touches is a Syntax object. The
scope set is the hygienic identity of an identifier; the provenance chain
records which transformation step (or input token) produced the object, which
is what lets errors and source maps point back to original source spans.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

SOURCE_BASIS = {
    "offset_unit": "utf8-codepoint",
    "line_base": 1,
    "col_base": 1,
    "coordinate_system": "source-text",
}


@dataclass
class Span:
    start: int
    end: int
    line: int
    col: int
    basis: dict = field(default_factory=lambda: dict(SOURCE_BASIS))

    def to_json(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "line": self.line,
            "col": self.col,
            "basis": dict(self.basis),
        }


@dataclass
class Syntax:
    datum: Any                      # str (identifier) | int | float | bool | list[Syntax]
    scopes: frozenset               # frozenset[str] of scope ids
    span: Optional[Span]
    prov: tuple = ()                # tuple of (origin_step_id, role)

    def is_ident(self) -> bool:
        return isinstance(self.datum, str)

    def is_list(self) -> bool:
        return isinstance(self.datum, list)

    def to_json(self) -> dict:
        return {
            "datum": _datum_json(self.datum),
            "scopes": sorted(self.scopes),
            "span": self.span.to_json() if self.span else None,
            "prov": [list(p) for p in self.prov],
        }


def _datum_json(datum: Any) -> Any:
    if isinstance(datum, list):
        return [_datum_json(d) for d in datum]
    return datum


def add_scope(stx: Syntax, scope: str) -> Syntax:
    """Return a new Syntax with `scope` added recursively (never mutates)."""
    scopes = frozenset(set(stx.scopes) | {scope})
    if isinstance(stx.datum, list):
        datum: Any = [add_scope(x, scope) for x in stx.datum]
    else:
        datum = stx.datum
    return Syntax(datum, scopes, stx.span, stx.prov)


def datum_to_source(datum: Any) -> str:
    if isinstance(datum, list):
        return "(" + " ".join(datum_to_source(d) for d in datum) + ")"
    if isinstance(datum, bool):
        return "#t" if datum else "#f"
    return str(datum)
