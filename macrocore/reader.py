"""Reader: tokens -> syntax objects (datum->syntax with empty scope set).

The reader is the first place where a *context choice* is made: top-level
forms are read with the empty scope set.  Later datum->syntax coercions
(macro templates, explicit capture) choose other contexts and record an
event each time.
"""
from __future__ import annotations

from typing import List, Optional

from .syntax import IdGen, Span, Syntax, Token


class ReadError(Exception):
    def __init__(self, message: str, span: Optional[Span]):
        super().__init__(message)
        self.message = message
        self.span = span


def datum_to_syntax(datum, ctx_scopes: frozenset, idgen: IdGen, span: Optional[Span] = None) -> Syntax:
    """Coerce a plain datum to a syntax object carrying ``ctx_scopes``.

    The choice of context is explicit and recorded by callers as a
    ``datum_to_syntax`` event.
    """
    if isinstance(datum, list):
        return Syntax(
            sid=idgen.next("n"),
            datum=[datum_to_syntax(d, ctx_scopes, idgen) for d in datum],
            scopes=ctx_scopes,
            span=span,
            kind="list",
        )
    if isinstance(datum, bool):
        raise ReadError("booleans unsupported", span)
    if isinstance(datum, int):
        return Syntax(idgen.next("n"), datum, ctx_scopes, span, kind="int")
    return Syntax(idgen.next("n"), datum, ctx_scopes, span, kind="ident")


def read_all(tokens: List[Token], idgen: IdGen) -> List[Syntax]:
    pos = 0

    def read_one() -> Syntax:
        nonlocal pos
        if pos >= len(tokens):
            raise ReadError("unexpected end of input", None)
        tok = tokens[pos]
        pos += 1
        if tok.kind == "lparen":
            children: List[Syntax] = []
            start_span = tok.span
            while True:
                if pos >= len(tokens):
                    raise ReadError("unclosed '('", start_span)
                if tokens[pos].kind == "rparen":
                    end_span = tokens[pos].span
                    pos += 1
                    return Syntax(
                        sid=idgen.next("n"),
                        datum=children,
                        scopes=frozenset(),
                        span=Span(start_span.start, end_span.end, start_span.line, start_span.col),
                        kind="list",
                    )
                children.append(read_one())
        if tok.kind == "rparen":
            raise ReadError("unexpected ')'", tok.span)
        if tok.kind == "int":
            return Syntax(idgen.next("n"), int(tok.text), frozenset(), tok.span, kind="int")
        if tok.kind == "string":
            return Syntax(idgen.next("n"), tok.text, frozenset(), tok.span, kind="string")
        return Syntax(idgen.next("n"), tok.text, frozenset(), tok.span, kind="ident")

    forms: List[Syntax] = []
    while pos < len(tokens):
        forms.append(read_one())
    return forms
