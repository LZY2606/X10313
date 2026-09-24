"""S-expression parser producing Syntax objects with stable token identities."""
from __future__ import annotations

from typing import List, Optional

from .lexer import Token, lex
from .syntax import Span, Syntax


class ParseError(Exception):
    def __init__(self, message: str, span: Optional[Span] = None):
        super().__init__(message)
        self.span = span


def _atom(tok: Token) -> Syntax:
    if tok.kind == "number":
        value = float(tok.text)
        datum = int(value) if value.is_integer() else value
    elif tok.kind == "bool":
        datum = tok.text == "#t"
    else:
        datum = tok.text
    return Syntax(datum=datum, scopes=frozenset({0}), span=tok.span, sid=f"tok:{tok.span.start}")


def parse_tokens(tokens: List[Token], source_id: str) -> List[Syntax]:
    """Parse a token stream into a list of top-level Syntax forms."""
    pos = 0
    counter = [0]

    def next_sid():
        counter[0] += 1
        return f"{source_id}/n{counter[0]}"

    def parse_one() -> Syntax:
        nonlocal pos
        if pos >= len(tokens):
            raise ParseError("unexpected end of input")
        tok = tokens[pos]
        if tok.kind == "lparen":
            pos += 1
            children: List[Syntax] = []
            while pos < len(tokens) and tokens[pos].kind != "rparen":
                children.append(parse_one())
            if pos >= len(tokens):
                raise ParseError("unclosed '('", tok.span)
            close = tokens[pos]
            pos += 1
            span = Span(source_id, tok.span.start, close.span.end,
                        tok.span.line, tok.span.col, tok.span.basis)
            return Syntax(datum=children, scopes=frozenset({0}), span=span,
                          sid=next_sid())
        if tok.kind == "rparen":
            raise ParseError("unexpected ')'", tok.span)
        pos += 1
        return _atom(tok)

    forms: List[Syntax] = []
    while pos < len(tokens):
        forms.append(parse_one())
    return forms


def parse_program(source_id: str, text: str, basis: str = "utf8-offset") -> List[Syntax]:
    return parse_tokens(lex(source_id, text, basis), source_id)
