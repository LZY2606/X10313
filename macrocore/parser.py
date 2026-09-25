"""Parser: tokens -> syntax objects, preserving token identity on every node."""
from __future__ import annotations

from typing import List, Optional

from .lexer import Token
from .syntax import Atom, Ctx, Id, Lst, Span, Syn


class ParseError(Exception):
    def __init__(self, message: str, span: Optional[Span]):
        super().__init__(message)
        self.message = message
        self.span = span

    def to_data(self) -> dict:
        return {
            "message": self.message,
            "span": self.span.to_data() if self.span else None,
            "phase": "parse",
        }


def _atom(tok: Token) -> Syn:
    ctx = Ctx(tok.span, frozenset(), tok.id, 0)
    if tok.kind == "number":
        text = tok.text
        value = float(text) if ("." in text or "e" in text.lower()) else int(text)
        return Atom(value, ctx)
    if tok.kind == "bool":
        return Atom(tok.text == "#t", ctx)
    if tok.kind == "string":
        return Atom(tok.text, ctx)
    return Id(tok.text, ctx)


def parse_program(tokens: List[Token]) -> Lst:
    pos = 0

    def parse_one() -> Syn:
        nonlocal pos
        if pos >= len(tokens):
            raise ParseError("unexpected end of input", tokens[-1].span if tokens else None)
        tok = tokens[pos]
        pos += 1
        if tok.kind == "lparen":
            items: List[Syn] = []
            while True:
                if pos >= len(tokens):
                    raise ParseError("unclosed '('", tok.span)
                if tokens[pos].kind == "rparen":
                    end = tokens[pos]
                    pos += 1
                    span = Span(tok.span.start, end.span.end, tok.span.line, tok.span.col)
                    return Lst(tuple(items), Ctx(span, frozenset(), tok.id + ":" + end.id, 0))
                items.append(parse_one())
        if tok.kind == "rparen":
            raise ParseError("unexpected ')'", tok.span)
        if tok.kind == "quote":
            inner = parse_one()
            inner_end = inner.ctx.span.end if inner.ctx.span else tok.span.end
            span = Span(tok.span.start, inner_end, tok.span.line, tok.span.col)
            return Lst(
                (Id("quote", Ctx(tok.span, frozenset(), tok.id, 0)), inner),
                Ctx(span, frozenset(), tok.id, 0),
            )
        return _atom(tok)

    forms: List[Syn] = []
    while pos < len(tokens):
        forms.append(parse_one())
    if not forms:
        raise ParseError("empty program", None)
    span = Span(
        forms[0].ctx.span.start,
        forms[-1].ctx.span.end,
        forms[0].ctx.span.line,
        forms[0].ctx.span.col,
    )
    return Lst(tuple(forms), Ctx(span, frozenset(), "prog", 0))
