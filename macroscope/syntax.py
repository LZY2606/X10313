"""Tokens, source spans and syntax objects.

A syntax object is a wrapped S-expression. Every atom carries:
- a stable id (deterministic for source nodes, generator-keyed for produced
  nodes so replays reproduce identity),
- a source span referencing the original input (span=None only for generated
  nodes, which still carry a provenance chain back to some source span),
- scope tags (frozenset[str]),
- the input/version id it came from.

Representation
--------------
Syntax values form an S-expression tree with raw Python types:
- str            -> symbol (identifier)
- int/float      -> number
- bool           -> literal
- list[SyntaxV]  -> form
Each value is wrapped by dataclass Syntax(v, ...).
"""

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Tuple


# Token kinds ---------------------------------------------------------------
T_LPAREN = "LPAREN"
T_RPAREN = "RPAREN"
T_SYMBOL = "SYMBOL"
T_NUMBER = "NUMBER"
T_STRING = "STRING"
T_QUOTE = "QUOTE"
T_BOOLEAN = "BOOLEAN"
T_UNKNOWN = "UNKNOWN"  # retained for unconvertible values (soft diagnostics)


@dataclass(frozen=True)
class Span:
    """Byte offsets and 1-based line/column, both ends inclusive.

    Coordinate basis: UTF-8 byte offsets from the start of the raw input.
    Original textual representation is kept on the token; normalisation never
    mutates it.
    """

    start: int
    end: int
    line: int
    column: int
    end_line: int
    end_column: int

    def as_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    span: Span
    input_id: str
    diagnostic: Optional[str] = None

    def as_dict(self):
        d = {"kind": self.kind, "text": self.text, "span": self.span.as_dict(),
             "input_id": self.input_id}
        if self.diagnostic:
            d["diagnostic"] = self.diagnostic
        return d


class FormatError(ValueError):
    """Hard format error: input cannot be read at all."""

    def __init__(self, message, line, column, text=""):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column
        self.text = text

    def as_dict(self):
        return {"message": self.message, "line": self.line,
                "column": self.column, "text": self.text}


# ---------------------------------------------------------------------------
# Syntax objects
# ---------------------------------------------------------------------------

@dataclass
class Syntax:
    """Annotated s-expression node.

    value: list[Syntax] | str | int | float | bool
    scopes: frozenset of scope tag strings
    span: original Span or None for generated nodes
    input_id: provenance to the raw input
    sid: stable identity (survives re-read and export/import)
    origin: provenance chain for generated nodes
        {"step": str, "macro_call_sid": str, "via": str,
         "template_sid": str|None, "from_sids": [...], "depth": int}
    notes: machine-readable per-node annotations (capture kind, etc.)
    """

    value: Any
    scopes: frozensenset = frozenset()
    span: Optional[Span] = None
    input_id: Optional[str] = None
    sid: Optional[str] = None
    origin: Optional[dict] = field(default_factory=dict)
    notes: list = field(default_factory=list)

    # -- shape helpers -----------------------------------------------------
    @property
    def is_list(self):
        return isinstance(self.value, list)

    @property
    def is_symbol(self):
        return isinstance(self.value, str)

    def head(self):
        if self.is_list and self.value:
            return self.value[0]
        return None

    def as_dict(self):
        if isinstance(self.value, list):
            raw = [c.as_dict() for c in self.value]
        else:
            raw = self.value
        d = {"sid": self.sid, "value": raw,
             "scopes": sorted(self.scopes)}
        if self.span is not None:
            d["span"] = self.span.as_dict()
            d["input_id"] = self.input_id
        if self.origin:
            d["origin"] = self.origin
        if self.notes:
            d["notes"] = list(self.notes)
        return d


def clone(node, value=..., scopes=..., span=..., input_id=..., sid=...,
          origin=..., notes=...):
    return replace(
        node,
        value=node.value if value is ... else value,
        scopes=node.scopes if scopes is ... else scopes,
        span=node.span if span is ... else span,
        input_id=node.input_id if input_id is ... else input_id,
        sid=node.sid if sid is ... else sid,
        origin=node.origin if origin is ... else origin,
        notes=node.notes if notes is ... else notes,
    )


def same_binding(a: Syntax, b: Syntax) -> bool:
    """bound-identifier=? : identity is scope sets + same raw name.

    Capture decisions are made by scope relations, never by string renaming.
    """
    return a.is_symbol and b.is_symbol and a.value == b.value and \
        a.scopes == b.scopes
