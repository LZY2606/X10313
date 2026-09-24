"""Syntax objects: datum + scope set + source location.

Identity is stable: every syntax object gets a deterministic sid from a
seeded counter, so exported/imported archives keep the same identities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .lexer import Span, Token, LexDiagnostic


class IdGen:
    """Deterministic id generator (fixed seed => stable identities)."""

    def __init__(self, seed: int = 0, prefix_salt: str = ""):
        self.seed = seed
        self.counters: dict[str, int] = {}
        self.salt = prefix_salt

    def next(self, kind: str) -> str:
        self.counters[kind] = self.counters.get(kind, 0) + 1
        return f"{self.salt}{kind}-{self.counters[kind]:04d}"


@dataclass
class Syntax:
    sid: str
    datum: object                 # str (identifier), float/int, or list[Syntax]
    scopes: frozenset             # set[str] of scope ids
    span: Span | None = None      # original source span (None if generated)
    introduced_by: str | None = None   # expansion node id that introduced it
    generated_name: str | None = None  # e.g. gensym result name

    @property
    def is_ident(self) -> bool:
        return isinstance(self.datum, str)

    @property
    def is_list(self) -> bool:
        return isinstance(self.datum, list)

    def to_dict(self) -> dict:
        if self.is_list:
            datum = [s.to_dict() for s in self.datum]
        else:
            datum = self.datum
        return {
            "sid": self.sid,
            "datum": datum,
            "scopes": sorted(self.scopes),
            "span": self.span.to_dict() if self.span else None,
            "introduced_by": self.introduced_by,
            "generated_name": self.generated_name,
        }


def datum_to_syntax(datum, context: Syntax | None, idgen: IdGen,
                    span: Span | None = None,
                    introduced_by: str | None = None) -> Syntax:
    """Wrap a raw datum as syntax. The *context* syntax object supplies the
    scope set; the choice of context is explicit and recorded by callers
    (datum->syntax context selection must be visible)."""
    scopes = context.scopes if context is not None else frozenset()
    if isinstance(datum, list):
        items = [datum_to_syntax(d, context, idgen, span=None,
                                 introduced_by=introduced_by) for d in datum]
        return Syntax(idgen.next("stx"), items, scopes, span, introduced_by)
    return Syntax(idgen.next("stx"), datum, scopes, span, introduced_by)


def tokens_to_syntax(tokens: list[Token], idgen: IdGen,
                     file: str = "<input>") -> tuple[list[Syntax], list[LexDiagnostic]]:
    """Parse a token stream into a list of top-level syntax objects."""
    diags: list[LexDiagnostic] = []
    pos = 0

    def parse_one() -> Syntax | None:
        nonlocal pos
        if pos >= len(tokens):
            return None
        tok = tokens[pos]
        if tok.kind == "lparen":
            pos += 1
            items: list[Syntax] = []
            while pos < len(tokens) and tokens[pos].kind != "rparen":
                item = parse_one()
                if item is not None:
                    items.append(item)
            if pos >= len(tokens):
                diags.append(LexDiagnostic(
                    "unclosed-list",
                    f"unclosed '(' opened at line {tok.span.line}", tok.span))
            else:
                pos += 1  # consume rparen
            return Syntax(idgen.next("stx"), items, frozenset(), tok.span)
        if tok.kind == "rparen":
            diags.append(LexDiagnostic("stray-rparen",
                                       f"stray ')' at line {tok.span.line}", tok.span))
            pos += 1
            return None
        pos += 1
        if tok.kind == "number":
            try:
                value: object = int(tok.text)
            except ValueError:
                try:
                    value = float(tok.text)
                except ValueError:
                    diags.append(LexDiagnostic("bad-number",
                                               f"cannot parse number {tok.text!r}",
                                               tok.span))
                    value = tok.text  # unconvertible value stays raw, with diagnostic
            return Syntax(idgen.next("stx"), value, frozenset(), tok.span)
        if tok.kind == "string":
            return Syntax(idgen.next("stx"), ("string", tok.text), frozenset(), tok.span)
        return Syntax(idgen.next("stx"), tok.text, frozenset(), tok.span)

    forms: list[Syntax] = []
    while pos < len(tokens):
        stx = parse_one()
        if stx is not None:
            forms.append(stx)
    return forms, diags
