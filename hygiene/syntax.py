"""Syntax objects, scope sets and macro-introduction marks.

Hygiene model (sets-of-scopes, Flatt-style "The Racket Way" mechanics):

* every identifier carries a *scope set* (a set of scope ids);
* a binding lives in one binding scope (created by a lambda/let/define);
* an identifier resolves to the binding whose scope is the most recently
  created scope present in the identifier's set;
* when a macro is invoked a fresh *mark* (introduction scope) is added to
  every syntax object of the input, and the same mark is toggled again on
  every object that came from the input after the transformer runs. Objects
  freshly created by the transformer template keep the mark, objects that
  merely flowed through lose it. Capture is therefore decided entirely by
  scope-set relations, never by string renaming.

Each syntax object also carries an immutable *birth* certificate describing
where it came from (parsed token / template instance / datum->syntax context
choice / generated temporary name) and a parent-edge origin chain used for
source reflection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .tokens import Span


@dataclass
class Birth:
    """Birth certificate of a syntax object (immutable provenance)."""
    kind: str                  # parsed | template | datum_to_syntax | gensym | copied
    detail: str = ""           # human readable role, e.g. "template:it", "clause:1"
    token_id: Optional[int] = None     # originating original token, if any
    template_serial: Optional[int] = None
    call_oid: Optional[str] = None     # macro call inside which it was born
    macro_name: Optional[str] = None
    context_donor: Optional[str] = None  # for datum->syntax: donor object id
    adopts_use_context: bool = False   # datum->syntax: recipient behaves use-site

    def as_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "token_id": self.token_id,
                "template_serial": self.template_serial, "call_oid": self.call_oid,
                "macro_name": self.macro_name, "context_donor": self.context_donor,
                "adopts_use_context": self.adopts_use_context}


@dataclass
class OriginEdge:
    """One edge in the provenance DAG: this object was derived from src_oid."""
    src_oid: str
    relation: str             # token | flip | template-slot | datum->syntax | gensym | copy
    step: str = ""

    def as_dict(self) -> dict:
        return {"src_oid": self.src_oid, "relation": self.relation, "step": self.step}


class Syntax:
    """A syntax object: datum (symbol/value or list of Syntax) + scope set.

    Identity: ``oid`` is assigned by the registry when the object is recorded
    and is stable across export/delete/import (deterministic content of the
    birth certificate within a derivation).
    """

    __slots__ = ("oid", "val", "is_list", "span", "scopes", "birth", "origins",
                 "serial", "mark_touches")

    def __init__(self, val, is_list: bool, span: Optional[Span], scopes=None,
                 birth: Optional[Birth] = None, serial: Optional[int] = None):
        self.val = val
        self.is_list = is_list
        self.span = span
        self.scopes = set(scopes) if scopes else set()
        self.birth = birth
        self.origins: list[OriginEdge] = []
        self.serial = serial
        self.mark_touches: list[str] = []  # mark ids toggled, in order (visibility)
        self.oid: str = ""

    @property
    def symbol(self) -> Optional[str]:
        return self.val if (not self.is_list and isinstance(self.val, str)) else None

    def add_scope(self, scope_id: str) -> None:
        self.scopes.add(scope_id)

    def toggle_scope(self, mark: str) -> bool:
        """Toggle a macro-introduction mark on this single object.

        Returns True if the mark is now present. The pair of toggles is what
        distinguishes introduced (kept mark) from passed-through (lost mark).
        """
        if mark in self.scopes:
            self.scopes.discard(mark)
            present = False
        else:
            self.scopes.add(mark)
            present = True
        self.mark_touches.append(f"{mark}:{'in' if present else 'out'}")
        return present

    def datum(self):
        if self.is_list:
            return [s.datum() for s in self.val]
        return self.val

    def clone(self, birth_kind: str = "copied", detail: str = "",
              call_oid: Optional[str] = None, macro_name: Optional[str] = None,
              keep_origins: bool = False):
        b = Birth(birth_kind, detail=detail or (self.birth.detail if self.birth else ""),
                  token_id=self.birth.token_id if self.birth else None,
                  template_serial=self.birth.template_serial if self.birth else None,
                  call_oid=call_oid or (self.birth.call_oid if self.birth else None),
                  macro_name=macro_name or (self.birth.macro_name if self.birth else None),
                  context_donor=self.birth.context_donor if self.birth else None,
                  adopts_use_context=(self.birth.adopts_use_context
                                      if self.birth else False))
        s = Syntax(self.val, self.is_list, self.span, set(self.scopes), b,
                   serial=self.serial)
        if keep_origins:
            s.origins = list(self.origins)
        return s

    def as_dict(self) -> dict:
        return {
            "oid": self.oid,
            "is_list": self.is_list,
            "value": self.datum() if self.is_list else self.val,
            "symbol": self.symbol,
            "span": self.span.as_dict() if self.span else None,
            "scopes": sorted(self.scopes),
            "birth": self.birth.as_dict() if self.birth else None,
            "origins": [e.as_dict() for e in self.origins],
            "mark_touches": list(self.mark_touches),
        }


class SyntaxRegistry:
    """Assigns stable oids and keeps every created syntax object."""

    def __init__(self, tag: str = "d0"):
        self.tag = tag
        self._seq = 0
        self.objs: dict[str, Syntax] = {}

    def record(self, s: Syntax) -> Syntax:
        self._seq += 1
        s.oid = f"sx-{self.tag}-{self._seq:05d}"
        self.objs[s.oid] = s
        return s

    def all_identifiers(self) -> list[Syntax]:
        out = []
        for s in self.objs.values():
            if s.symbol is not None:
                out.append(s)
        return out
