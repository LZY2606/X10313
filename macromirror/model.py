"""Core data model: tokens, spans, scopes, syntax objects.

Scopes are named strings carried by every syntax object as a sorted tuple.
During a single macro invocation a fresh *use scope* is added with an
anti-mark on input-derived nodes (the classical marks/anti-marks discipline),
which is what makes capture a question of scope relations, never of names.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Span:
    """Coordinates are kept verbatim in the original representation.

    Basis: line is 1-based, column and offset are 0-based character counts.
    """
    start: int
    end: int
    line: int
    col: int

    def as_json(self):
        return {"start": self.start, "end": self.end, "line": self.line,
                "col": self.col, "unit": "char",
                "basis": "line:1based/col:0based/offset:0based"}


@dataclass
class Token:
    kind: str            # lparen | rparen | quote | atom | boolean | integer
    text: str
    span: Span
    raw: str             # exact original slice (original representation kept)

    def as_json(self):
        return {"kind": self.kind, "text": self.text, "raw": self.raw,
                "span": self.span.as_json()}


@dataclass
class Syntax:
    sid: str
    kind: str                      # list | atom
    value: Optional[str] = None    # atom text, symbol names verbatim
    span: Optional[Span] = None
    children: list["Syntax"] = field(default_factory=list)
    scopes: tuple[str, ...] = ()
    anti: bool = False             # anti-mark for the live use scope
    generated: bool = False
    gen_name: Optional[str] = None  # gensym symbol when generated
    origin_sid: Optional[str] = None  # template-author sid for fresh nodes
    datum_diagnostic: Optional[str] = None

    # ---- construction helpers -------------------------------------------
    def atom_clone(self, value, *, scopes=None, anti=False, generated=False,
                   gen_name=None, origin_sid=None, span=None):
        node = Syntax(sid=self._syntax_ids.next_atom(self),
                      kind="atom", value=value,
                      span=span, children=[],
                      scopes=tuple(sorted(scopes)) if scopes is not None
                      else self.scopes,
                      anti=anti, generated=generated, gen_name=gen_name,
                      origin_sid=origin_sid)
        return node

    def with_scopes(self, scopes):
        self.scopes = tuple(sorted(set(scopes)))
        return self

    def add_scope(self, scope):
        if scope not in self.scopes:
            self.scopes = tuple(sorted((*self.scopes, scope)))
        return self

    def deep_add_scope(self, scope, only_anti=None):
        if only_anti is None or self.anti is only_anti:
            self.add_scope(scope)
        for ch in self.children:
            ch.deep_add_scope(scope, only_anti)

    def flip(self, use_scope):
        """Post-transformer flip: anti-marked nodes lose the use scope,
        every other node gains it."""
        if self.anti:
            self.scopes = tuple(s for s in self.scopes if s != use_scope)
            self.anti = False
        else:
            self.add_scope(use_scope)
        for ch in self.children:
            ch.flip(use_scope)

    def every(self):
        yield self
        for ch in self.children:
            yield from ch.every()

    def as_json(self):
        out = {"sid": self.sid, "kind": self.kind,
               "scopes": list(self.scopes), "anti": self.anti,
               "generated": self.generated}
        if self.value is not None:
            out["value"] = self.value
        if self.gen_name:
            out["genName"] = self.gen_name
        if self.span:
            out["span"] = self.span.as_json()
        if self.origin_sid:
            out["originSid"] = self.origin_sid
        if self.datum_diagnostic:
            out["datumDiagnostic"] = self.datum_diagnostic
        if self.kind == "list":
            out["children"] = [c.as_json() for c in self.children]
        return out


class IdFactory:
    """Deterministic identity allocation so replay reproduces every sid."""

    def __init__(self):
        self.n = 0

    def next(self, prefix="s"):
        sid = f"{prefix}{self.n}"
        self.n += 1
        return sid

    def next_atom(self, template_node):
        return self.next("g" if template_node.generated else "x")
