"""Syntax objects: datum + source span + scope set + provenance.

A syntax object pairs a datum (identifier / literal / list of syntax) with
- span:   source coordinates, always carrying a coordinate basis,
- scopes: a set of scope identities (hygiene is judged by these, never by
          string renaming),
- origin/step: provenance back to the syntax object it derives from and the
          expansion step that created it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SPAN_BASIS = "line:col 1-based inclusive; offset 0-based exclusive; unit=unicode-codepoint"


@dataclass(frozen=True)
class Span:
    file: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_offset: int
    end_offset: int
    basis: str = SPAN_BASIS

    def to_obj(self) -> dict:
        return {
            "file": self.file,
            "start_line": self.start_line,
            "start_col": self.start_col,
            "end_line": self.end_line,
            "end_col": self.end_col,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "basis": self.basis,
        }

    @staticmethod
    def from_obj(o: dict) -> "Span":
        return Span(
            file=o["file"],
            start_line=o["start_line"],
            start_col=o["start_col"],
            end_line=o["end_line"],
            end_col=o["end_col"],
            start_offset=o["start_offset"],
            end_offset=o["end_offset"],
            basis=o.get("basis", SPAN_BASIS),
        )


@dataclass
class Syntax:
    sid: str
    datum: Any  # str (identifier) | int | bool | list[Syntax]
    span: Optional[Span] = None
    scopes: frozenset = field(default=frozenset)
    origin: Optional[str] = None  # sid of the syntax this one derives from
    step: Optional[str] = None    # expansion step that created it

    @property
    def is_ident(self) -> bool:
        return isinstance(self.datum, str)

    @property
    def is_list(self) -> bool:
        return isinstance(self.datum, list)

    def to_obj(self) -> dict:
        if self.is_list:
            datum = {"list": [c.sid for c in self.datum]}
        else:
            datum = {"atom": self.datum}
        return {
            "sid": self.sid,
            "datum": datum,
            "span": self.span.to_obj() if self.span else None,
            "scopes": sorted(self.scopes),
            "origin": self.origin,
            "step": self.step,
        }

    def datum_text(self) -> str:
        if self.is_list:
            return "(" + " ".join(c.datum_text() for c in self.datum) + ")"
        if self.datum is True:
            return "#t"
        if self.datum is False:
            return "#f"
        return str(self.datum)
