"""Syntax objects: datum + source span + scope set + provenance.

A ``Syntax`` never loses where it came from: ``origin`` is the id of the
token it was parsed from, or the id of the transformation step that
produced it.  ``datum_to_syntax`` makes the *context choice* (which
scope set a fresh identifier receives) an explicit, inspectable value.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .lexer import Span, Token, Diagnostic


class StrLit(str):
    """A string literal, distinguished from an identifier."""


class Syntax:
    __slots__ = ("sid", "datum", "span", "scopes", "origin")

    def __init__(self, sid: str, datum: Any, span: Optional[Span],
                 scopes: frozenset, origin: Optional[str]):
        self.sid = sid
        self.datum = datum            # str | StrLit | int | float | list[Syntax]
        self.span = span
        self.scopes = frozenset(scopes)
        self.origin = origin

    @property
    def is_list(self) -> bool:
        return isinstance(self.datum, list)

    @property
    def is_ident(self) -> bool:
        return isinstance(self.datum, str) and not isinstance(self.datum, StrLit)

    def to_json(self) -> dict:
        if self.is_list:
            datum = [c.to_json() for c in self.datum]
        elif isinstance(self.datum, StrLit):
            datum = {"string": str(self.datum)}
        else:
            datum = self.datum
        return {"sid": self.sid, "datum": datum,
                "span": self.span.to_json() if self.span else None,
                "scopes": sorted(self.scopes), "origin": self.origin}

    def __repr__(self) -> str:  # pragma: no cover
        return f"Syntax({self.sid}, {self.datum!r}, scopes={sorted(self.scopes)})"


class SyntaxFactory:
    """Mints syntax objects with deterministic ids (fixed-seed friendly)."""

    def __init__(self, prefix: str = "syn"):
        self.prefix = prefix
        self.counter = 0

    def new(self, datum: Any, span: Optional[Span], scopes: Iterable[int],
            origin: Optional[str]) -> Syntax:
        sid = f"{self.prefix}_{self.counter:05d}"
        self.counter += 1
        return Syntax(sid, datum, span, frozenset(scopes), origin)


def datum_to_syntax(factory: SyntaxFactory, datum: Any,
                    context: Optional[Syntax], span: Optional[Span],
                    origin: Optional[str],
                    scopes: Optional[Iterable[int]] = None) -> Syntax:
    """Wrap a raw datum as syntax.

    The context choice is explicit: either an existing ``context`` syntax
    object whose scope set is inherited, or an explicit ``scopes`` set.
    Callers must record which alternative they picked (the expander logs
    it as a machine-readable step).
    """
    if scopes is None:
        if context is None:
            raise ValueError("datum_to_syntax requires a context or explicit scopes")
        scopes = context.scopes
    if isinstance(datum, list):
        datum = [datum_to_syntax(factory, d, None, span, origin, scopes)
                 for d in datum]
    return factory.new(datum, span, scopes, origin)


def parse_tokens(tokens: list[Token], factory: SyntaxFactory
                 ) -> tuple[list[Syntax], list[Diagnostic]]:
    """Parse a flat token stream into top-level syntax forms."""
    diags: list[Diagnostic] = []
    stack: list[tuple[Token, list[Syntax]]] = []
    roots: list[Syntax] = []

    def atom(tok: Token) -> Syntax:
        if tok.kind == "number":
            val: Any = float(tok.text) if "." in tok.text else int(tok.text)
        elif tok.kind == "string":
            val = StrLit(tok.text)
        else:
            val = tok.text
        return factory.new(val, tok.span, frozenset(), tok.tid)

    for tok in tokens:
        if tok.kind == "lparen":
            stack.append((tok, []))
        elif tok.kind == "rparen":
            if not stack:
                diags.append(Diagnostic(
                    "unbalanced-parens",
                    f"unexpected ')' at line {tok.span.line} col {tok.span.col}",
                    tok.span.to_json()))
                continue
            open_tok, children = stack.pop()
            span = Span(open_tok.span.start, tok.span.end,
                        open_tok.span.line, open_tok.span.col)
            form = factory.new(children, span, frozenset(), open_tok.tid)
            if stack:
                stack[-1][1].append(form)
            else:
                roots.append(form)
        else:
            a = atom(tok)
            if stack:
                stack[-1][1].append(a)
            else:
                roots.append(a)
    for open_tok, _ in stack:
        diags.append(Diagnostic(
            "unbalanced-parens",
            f"'(' at line {open_tok.span.line} col {open_tok.span.col} is never closed",
            open_tok.span.to_json()))
    return roots, diags
