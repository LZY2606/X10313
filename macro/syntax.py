"""Syntax objects: datum + scope set + source span + origin chain.

A Syntax object pairs a datum (symbol / number / string / list) with:
  - scopes:  frozenset of scope labels (ints). Binding is decided by
             subset relations between scope sets, never by string renaming.
  - span:    source location, carrying its coordinate basis (unit).
  - origin:  chain of (step_id, note) recording which expansion steps
             produced this node (the source-mapping edge list).
  - sid:     stable identity, deterministic for a given expansion run.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Tuple


BASES = ("utf8-offset", "utf16-code-unit", "byte-offset", "pixel-x")


@dataclass(frozen=True)
class Span:
    source: str            # source document id
    start: int
    end: int
    line: int
    col: int
    basis: str = "utf8-offset"   # coordinate basis / unit of start/end

    def to_dict(self) -> dict:
        return {
            "source": self.source, "start": self.start, "end": self.end,
            "line": self.line, "col": self.col, "basis": self.basis,
        }

    @staticmethod
    def from_dict(d: dict) -> "Span":
        return Span(d["source"], d["start"], d["end"], d["line"], d["col"],
                    d.get("basis", "utf8-offset"))


@dataclass(frozen=True)
class Syntax:
    datum: Any                       # str | int | float | bool | list[Syntax]
    scopes: frozenset = field(default_factory=frozenset)
    span: Optional[Span] = None
    origin: Tuple[tuple, ...] = ()   # ((step_id, note), ...)
    sid: str = ""

    @property
    def is_list(self) -> bool:
        return isinstance(self.datum, list)

    @property
    def is_symbol(self) -> bool:
        return isinstance(self.datum, str)

    def with_scopes(self, scopes: frozenset) -> "Syntax":
        return replace(self, scopes=scopes)

    def add_scope(self, scope: int) -> "Syntax":
        return replace(self, scopes=self.scopes | {scope})

    def add_origin(self, step_id: str, note: str) -> "Syntax":
        return replace(self, origin=self.origin + ((step_id, note),))

    def map_children(self, fn) -> "Syntax":
        """Apply fn to every node in the tree (post-order rebuild)."""
        if self.is_list:
            new_children = [c.map_children(fn) for c in self.datum]
            return fn(replace(self, datum=new_children))
        return fn(self)

    def to_dict(self) -> dict:
        datum = [c.to_dict() for c in self.datum] if self.is_list else self.datum
        return {
            "sid": self.sid,
            "datum": datum,
            "scopes": sorted(self.scopes),
            "span": self.span.to_dict() if self.span else None,
            "origin": [list(o) for o in self.origin],
        }

    def datum_str(self) -> str:
        if self.is_list:
            return "(" + " ".join(c.datum_str() for c in self.datum) + ")"
        return repr(self.datum) if isinstance(self.datum, str) and not self.is_symbol else str(self.datum)
