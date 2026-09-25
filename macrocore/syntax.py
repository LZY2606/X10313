"""Syntax objects: datum + source span + scope set + origin identity.

Every syntax node carries a Ctx with:
  - span:   source coordinates (coordinate basis declared on the Span)
  - scopes: frozenset of scope identities (sets-of-scopes hygiene)
  - origin: stable identity of the token / generator that produced the node
  - step:   expansion step that introduced the node (0 = original source)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

COORD_BASIS = "utf8-offset/1-based-line-col"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    line: int
    col: int
    basis: str = COORD_BASIS

    def to_data(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "line": self.line,
            "col": self.col,
            "basis": self.basis,
        }

    @staticmethod
    def from_data(d: dict) -> "Span":
        return Span(d["start"], d["end"], d["line"], d["col"], d.get("basis", COORD_BASIS))


@dataclass(frozen=True)
class Ctx:
    span: Optional[Span]
    scopes: frozenset
    origin: str
    step: int = 0

    def to_data(self) -> dict:
        return {
            "span": self.span.to_data() if self.span else None,
            "scopes": sorted(self.scopes),
            "origin": self.origin,
            "step": self.step,
        }

    @staticmethod
    def from_data(d: dict) -> "Ctx":
        return Ctx(
            span=Span.from_data(d["span"]) if d.get("span") else None,
            scopes=frozenset(d.get("scopes", [])),
            origin=d.get("origin", "?"),
            step=d.get("step", 0),
        )


class Syn:
    ctx: Ctx


@dataclass(frozen=True)
class Id(Syn):
    name: str
    ctx: Ctx


@dataclass(frozen=True)
class Atom(Syn):
    value: Any
    ctx: Ctx


@dataclass(frozen=True)
class Lst(Syn):
    items: Tuple[Syn, ...]
    ctx: Ctx


def _map_ctx(syn: Syn, fn) -> Syn:
    if isinstance(syn, Id):
        return Id(syn.name, fn(syn.ctx))
    if isinstance(syn, Atom):
        return Atom(syn.value, fn(syn.ctx))
    if isinstance(syn, Lst):
        return Lst(tuple(_map_ctx(i, fn) for i in syn.items), fn(syn.ctx))
    raise TypeError(f"unknown syntax node: {syn!r}")


def add_scope(syn: Syn, scope: str) -> Syn:
    """Add a scope identity to every node of the tree (binders/bodies/args)."""
    return _map_ctx(syn, lambda c: Ctx(c.span, c.scopes | {scope}, c.origin, c.step))


def replace_scopes(syn: Syn, scopes: frozenset, origin: Optional[str] = None, step: int = 0) -> Syn:
    """Replace the scope set of every node (datum->syntax context selection)."""
    return _map_ctx(
        syn,
        lambda c: Ctx(c.span, frozenset(scopes), origin if origin is not None else c.origin, step),
    )


def datum_to_syntax(datum: Any, ctx: Ctx) -> Syn:
    """Convert a raw datum to syntax using an explicitly chosen context.

    The context choice is what makes datum->syntax visible and is recorded by
    the expander as a first-class graph node.
    """
    if isinstance(datum, Syn):
        return replace_scopes(datum, ctx.scopes, origin=ctx.origin, step=ctx.step)
    if isinstance(datum, str):
        return Id(datum, ctx)
    if isinstance(datum, (int, float, bool)) or datum is None:
        return Atom(datum, ctx)
    if isinstance(datum, (list, tuple)):
        return Lst(tuple(datum_to_syntax(d, ctx) for d in datum), ctx)
    raise TypeError(f"cannot convert datum: {datum!r}")


def syn_to_datum(syn: Syn) -> Any:
    if isinstance(syn, Id):
        return ("id", syn.name)
    if isinstance(syn, Atom):
        return syn.value
    if isinstance(syn, Lst):
        return [syn_to_datum(i) for i in syn.items]
    raise TypeError(syn)


def label(syn: Syn) -> str:
    if isinstance(syn, Id):
        return syn.name
    if isinstance(syn, Atom):
        return repr(syn.value)
    if isinstance(syn, Lst):
        return "(" + " ".join(label(i) for i in syn.items) + ")"
    return "?"


def to_data(syn: Syn) -> dict:
    if isinstance(syn, Id):
        return {"t": "id", "name": syn.name, "ctx": syn.ctx.to_data()}
    if isinstance(syn, Atom):
        return {"t": "atom", "value": syn.value, "ctx": syn.ctx.to_data()}
    if isinstance(syn, Lst):
        return {"t": "list", "items": [to_data(i) for i in syn.items], "ctx": syn.ctx.to_data()}
    raise TypeError(syn)


def from_data(d: dict) -> Syn:
    ctx = Ctx.from_data(d["ctx"])
    if d["t"] == "id":
        return Id(d["name"], ctx)
    if d["t"] == "atom":
        return Atom(d["value"], ctx)
    if d["t"] == "list":
        return Lst(tuple(from_data(i) for i in d["items"]), ctx)
    raise ValueError(d)
