"""Syntax objects: tokens, source spans, scope sets, labels, origin chains.

The scope-set hygiene algorithm follows Flatt, "Binding as Sets of Scopes"
(POPL 2016). Each identifier carries a frozenset of scope ids; a binding is
installed on an identifier (symbol + scope set) and reference resolution picks
the most specific applicable binding (largest scope set, scope-id order as
tie breaker). Every object additionally carries:

* ``sid``    - stable per-derivation preorder id (display identity)
* ``labels`` - set of macro-call steps at which this object was introduced
               (used to make the add/remove scope "flip" origin aware)
* ``origin`` - chain of provenance links back to original tokens
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

# coordinate basis documented with every span:
# offsets are 0-based UTF-8? no - Python codepoint offsets; line/col 1-based
SPAN_BASIS = "codepoint-offset:0;line:1;column:1"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    line: int
    col: int
    basis: str = SPAN_BASIS
    synthetic: bool = False
    note: str = ""

    def to_json(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "line": self.line,
            "col": self.col,
            "basis": self.basis,
            "synthetic": self.synthetic,
            "note": self.note,
        }

    @staticmethod
    def synthetic_at(note: str, anchor: Optional["Span"] = None) -> "Span":
        if anchor is None:
            return Span(0, 0, 0, 0, synthetic=True, note=note)
        return Span(
            anchor.start, anchor.end, anchor.line, anchor.col,
            synthetic=True, note=note,
        )


@dataclass(frozen=True)
class OriginLink:
    """One edge in a provenance chain.

    ``what``   - human/ machine-readable reason (e.g. "macro-template",
                 "datum->syntax", "syntax-gensym", "reader")
    ``call``   - macro call stack step id, if any
    ``parent`` - previous link
    """

    what: str
    call: Optional[int] = None
    parent: Optional["OriginLink"] = None

    def chain(self) -> list[dict]:
        out: list[dict] = []
        link: Optional[OriginLink] = self
        seen: set[int] = set()
        while link is not None and id(link) not in seen:
            seen.add(id(link))
            out.append({"what": link.what, "call": link.call})
            link = link.parent
        return out


@dataclass
class Token:
    kind: str           # id | int | float | string | bool | punct
    text: str
    span: Span
    unit: str = "lexeme"  # domain quantity carries its unit/basis
    raw: str = ""        # original representation preserved on import

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "text": self.text,
            "unit": self.unit,
            "raw": self.raw or self.text,
            "span": self.span.to_json(),
        }


@dataclass
class SObj:
    """A syntax object. Leaf when ``atom`` is not None."""

    atom: Any = None                  # symbol str | int | float | str(string)| bool
    kind: str = "id"                  # id | int | float | string | bool
    span: Optional[Span] = None
    items: list["SObj"] = field(default_factory=list)
    tail: Optional["SObj"] = None     # for dotted lists
    scopes: frozenset[int] = frozenset()
    labels: frozenset[int] = frozenset()  # macro steps at which introduced
    origin: Optional[OriginLink] = None
    sid: int = -1                     # stable display id, assigned post order
    gen_name: Optional[str] = None    # present when this is a generated symbol

    # ---- construction helpers -----------------------------------------
    @staticmethod
    def leaf(atom: Any, kind: str, span: Optional[Span],
             origin: Optional[OriginLink] = None) -> "SObj":
        return SObj(atom=atom, kind=kind, span=span, origin=origin)

    @staticmethod
    def form(items: list["SObj"], span: Optional[Span],
             origin: Optional[OriginLink] = None,
             tail: Optional["SObj"] = None) -> "SObj":
        return SObj(kind="form", span=span, items=items, tail=tail, origin=origin)

    @property
    def is_form(self) -> bool:
        return self.kind == "form"

    @property
    def symbol(self) -> str:
        return self.atom

    def head_symbol(self) -> Optional[str]:
        if self.is_form and self.items and self.items[0].kind == "id":
            return self.items[0].atom
        return None

    # ---- scope / label algebra ----------------------------------------
    def add_scope(self, scope: int) -> "SObj":
        """Origin unaware: used before the transformer (whole input)."""
        if self.is_form:
            return SObj(
                atom=None, kind="form", span=self.span,
                items=[c.add_scope(scope) for c in self.items],
                tail=self.tail.add_scope(scope) if self.tail else None,
                scopes=self.scopes | {scope}, labels=self.labels,
                origin=self.origin, gen_name=self.gen_name,
            )
        return replace(self, scopes=self.scopes | {scope})

    def flip_after(self, scope: int, step: int) -> "SObj":
        """After the transformer: remove ``scope`` from use-derived pieces,
        keep it on pieces introduced at ``step``."""
        keep = scope in self.scopes and step in self.labels
        scopes = self.scopes
        if scope in scopes and not keep:
            scopes = scopes - {scope}
        if self.is_form:
            return SObj(
                atom=None, kind="form", span=self.span,
                items=[c.flip_after(scope, step) for c in self.items],
                tail=self.tail.flip_after(scope, step) if self.tail else None,
                scopes=scopes, labels=self.labels, origin=self.origin,
                gen_name=self.gen_name,
            )
        return replace(self, scopes=scopes)

    def mark_introduced(self, step: int, scope: int) -> "SObj":
        """Used when a template piece is instantiated: labels it with the
        current step and stamps the (already pre-added) use scope."""
        if self.is_form:
            return SObj(
                atom=None, kind="form", span=self.span,
                items=[c.mark_introduced(step, scope) for c in self.items],
                tail=self.tail.mark_introduced(step, scope) if self.tail else None,
                scopes=self.scopes | {scope},
                labels=self.labels | {step}, origin=self.origin,
                gen_name=self.gen_name,
            )
        return replace(
            self, scopes=self.scopes | {scope}, labels=self.labels | {step},
        )

    def clone(self, origin: Optional[OriginLink] = None) -> "SObj":
        if self.is_form:
            return SObj(
                atom=None, kind="form", span=self.span,
                items=[c.clone(origin) for c in self.items],
                tail=self.tail.clone(origin) if self.tail else None,
                scopes=self.scopes, labels=self.labels,
                origin=origin or self.origin, gen_name=self.gen_name,
                sid=self.sid,
            )
        return replace(self, origin=origin or self.origin)

    def add_scope_inplace(self, scope: int) -> None:
        """Origin unaware pre-transformer add; mutates this tree."""
        if scope not in self.scopes:
            self.scopes = self.scopes | {scope}
        if self.is_form:
            for c in self.items:
                c.add_scope_inplace(scope)
            if self.tail:
                self.tail.add_scope_inplace(scope)

    def flip_inplace(self, scope: int, step: int) -> None:
        """Post-transformer flip: ``scope`` stays only on pieces that were
        introduced at ``step`` (labels), use-derived pieces lose it."""
        if scope in self.scopes and step not in self.labels:
            self.scopes = self.scopes - {scope}
        if self.is_form:
            for c in self.items:
                c.flip_inplace(scope, step)
            if self.tail:
                self.tail.flip_inplace(scope, step)

    def mark_introduced_inplace(self, step: int, scope: int) -> None:
        self.scopes = self.scopes | {scope}
        self.labels = self.labels | {step}
        if self.is_form:
            for c in self.items:
                c.mark_introduced_inplace(step, scope)
            if self.tail:
                self.tail.mark_introduced_inplace(step, scope)

    def mark_self_inplace(self, step: int, scope: int) -> None:
        """Mark only this node (children are already marked individually)."""
        self.scopes = self.scopes | {scope}
        self.labels = self.labels | {step}

    def assign_sids(self, counter: list[int]) -> None:
        """Preorder sid assignment for newly created subtrees."""
        stack = [self]
        while stack:
            node = stack.pop()
            if node.sid < 0:
                counter[0] += 1
                node.sid = counter[0]
            if node.is_form:
                if node.tail:
                    stack.append(node.tail)
                stack.extend(reversed(node.items))

    def all_scopes(self) -> frozenset[int]:
        s = set(self.scopes)
        if self.is_form:
            for c in self.items:
                s |= c.all_scopes()
            if self.tail:
                s |= self.tail.all_scopes()
        return frozenset(s)

    def to_datum_json(self) -> Any:
        if not self.is_form:
            return self.atom
        out: list[Any] = [c.to_datum_json() for c in self.items]
        if self.tail is not None:
            out.append({"__dot__": self.tail.to_datum_json()})
        return out
