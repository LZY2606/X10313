"""Domain primitives: tokens, source spans, scope sets and syntax objects.

Coordinate/unit conventions (see cases README):
* character offsets are 0-based ``char`` indices into the submitted source;
* line/column are 1-based;
* scope identifiers are abstract integers with no string-renaming meaning;
* expansion budgets are counted in ``macro-call`` units;
* every derived numeric value keeps the raw representation of its origin.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

UNIT_OFFSET = "char"
UNIT_LINE = "line(1-based)"
UNIT_SCOPE = "scope-id"
UNIT_BUDGET = "macro-call"


@dataclass(frozen=True)
class Span:
    file_id: str
    start: int          # 0-based char offset
    end: int
    line: int           # 1-based
    col: int            # 1-based

    def to_json(self) -> dict:
        return {
            "file": self.file_id, "start": self.start, "end": self.end,
            "line": self.line, "col": self.col, "unit": UNIT_OFFSET,
        }


@dataclass(frozen=True)
class Token:
    kind: str           # lparen|rparen|sugar|symbol|number|string|boolean|dot
    text: str
    span: Span
    raw: Optional[str] = None  # raw representation as submitted (numbers/strings)

    def to_json(self) -> dict:
        return {"kind": self.kind, "text": self.text, "span": self.span.to_json(),
                "raw": self.raw}


@dataclass(frozen=True)
class Origin:
    """Provenance of a syntax object: parse site, macro copy, gensym or datum->syntax."""
    kind: str                       # parse|sugar|macro|gensym|d2s|splice
    macro: Optional[str] = None
    def_site: Optional[str] = None
    call_site: Optional[str] = None
    call_stack: tuple[str, ...] = ()
    template_sid: Optional[str] = None
    gen_serial: Optional[int] = None
    basis: Optional[str] = None     # source sid copied by datum->syntax / macro arg

    def to_json(self) -> dict:
        return {
            "kind": self.kind, "macro": self.macro, "def_site": self.def_site,
            "call_site": self.call_site, "call_stack": list(self.call_stack),
            "template_sid": self.template_sid, "gen_serial": self.gen_serial,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class Node:
    kind: str                       # 'list' or 'atom'
    sid: str                        # stable identity, path-derived or expansion-derived
    scopes: frozenset[int]
    span: Optional[Span] = None
    children: tuple["Node", ...] = ()
    value: Optional[str] = None     # atom text (symbol name / raw number / string body)
    lit: str = "symbol"             # symbol|number|string|boolean
    origin: Origin = field(default_factory=lambda: Origin("parse"))
    fresh: bool = False             # True for gensym-produced symbols
    datum: bool = False             # True for syntax-less datum reintroduced via datum->syntax
    dot: bool = False                 # True for reader dotted pair

    def with_scopes(self, add: tuple[int, ...] = (), remove: tuple[int, ...] = (),
                    **kw: Any) -> "Node":
        scopes = (set(self.scopes) | set(add)) - set(remove)
        return replace(self, scopes=frozenset(scopes), **kw)

    def atom_text(self) -> str:
        return self.value if self.value is not None else ""

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def to_json(self) -> dict:
        return {
            "kind": self.kind, "sid": self.sid,
            "scopes": sorted(self.scopes), "scope_unit": UNIT_SCOPE,
            "span": self.span.to_json() if self.span else None,
            "value": self.value, "lit": self.lit,
            "origin": self.origin.to_json(), "fresh": self.fresh, "datum": self.datum,
            "dot": self.dot,
            "children": [c.to_json() for c in self.children],
        }


@dataclass(frozen=True)
class Binding:
    name: str
    scopes: frozenset[int]          # scope set installed at the binding site
    site: str                       # sid of the binding occurrence
    kind: str                       # lambda|let|define|defmacro|top
    span: Optional[Span]
    unit: str = "scope-set"

    def to_json(self) -> dict:
        return {
            "name": self.name, "scopes": sorted(self.scopes),
            "scope_unit": UNIT_SCOPE, "site": self.site, "kind": self.kind,
            "span": self.span.to_json() if self.span else None, "unit": self.unit,
        }
