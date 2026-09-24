"""Lexer + parser for the tiny s-expression language.

Every token keeps its exact source span; every parsed syntax node keeps the
token id it came from, so provenance is never lost downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .syntax import SIdent, SInt, SList, Span, Syn


class LexError(Exception):
    def __init__(self, message: str, span: Span):
        super().__init__(message)
        self.message = message
        self.span = span

    def to_diag(self) -> dict:
        return {"kind": "lex", "message": self.message, "span": self.span.to_dict()}


class ParseError(Exception):
    def __init__(self, message: str, span: Span):
        super().__init__(message)
        self.message = message
        self.span = span

    def to_diag(self) -> dict:
        return {"kind": "parse", "message": self.message, "span": self.span.to_dict()}


@dataclass(frozen=True)
class Token:
    kind: str  # lparen | rparen | ident | int
    text: str
    span: Span
    tid: str

    def to_dict(self) -> dict:
        return {"tid": self.tid, "kind": self.kind, "text": self.text, "span": self.span.to_dict()}


_IDENT_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_+-*/<>=!?")
_IDENT_CONT = _IDENT_START | set("0123456789")


def lex(src: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    line, col = 1, 1
    n = 0

    def span(start, l0, c0):
        return Span(start, i, l0, c0)

    while i < len(src):
        ch = src[i]
        if ch in " \t\r\n":
            if ch == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1
            continue
        if ch == ";":
            while i < len(src) and src[i] != "\n":
                i += 1
                col += 1
            continue
        s0, l0, c0 = i, line, col
        if ch == "(":
            tokens.append(Token("lparen", ch, span(s0, l0, c0), f"t{n}"))
            n += 1
            i += 1
            col += 1
            continue
        if ch == ")":
            tokens.append(Token("rparen", ch, span(s0, l0, c0), f"t{n}"))
            n += 1
            i += 1
            col += 1
            continue
        if ch.isdigit() or (ch == "-" and i + 1 < len(src) and src[i + 1].isdigit()):
            j = i + 1
            while j < len(src) and src[j].isdigit():
                j += 1
            text = src[i:j]
            tokens.append(Token("int", text, Span(s0, j, l0, c0), f"t{n}"))
            n += 1
            col += j - i
            i = j
            continue
        if ch in _IDENT_START:
            j = i + 1
            while j < len(src) and src[j] in _IDENT_CONT:
                j += 1
            text = src[i:j]
            tokens.append(Token("ident", text, Span(s0, j, l0, c0), f"t{n}"))
            n += 1
            col += j - i
            i = j
            continue
        raise LexError(f"unexpected character {ch!r}", span(s0, l0, c0))
    return tokens


def parse(src: str) -> List[Syn]:
    """Parse a source file into a list of top-level syntax objects."""
    tokens = lex(src)
    pos = 0
    counter = 0

    def next_nid():
        nonlocal counter
        counter += 1
        return f"n{counter - 1}"

    def parse_one() -> Syn:
        nonlocal pos
        if pos >= len(tokens):
            raise ParseError("unexpected end of input", tokens[-1].span if tokens else Span(0, 0, 1, 1))
        tok = tokens[pos]
        pos += 1
        if tok.kind == "int":
            return SInt(int(tok.text), frozenset({"src"}), tok.span, "src", next_nid())
        if tok.kind == "ident":
            return SIdent(tok.text, frozenset({"src"}), tok.span, "src", next_nid())
        if tok.kind == "lparen":
            items = []
            start_span = tok.span
            while True:
                if pos >= len(tokens):
                    raise ParseError("unclosed '('", start_span)
                if tokens[pos].kind == "rparen":
                    end = tokens[pos].span
                    pos += 1
                    return SList(
                        tuple(items),
                        frozenset({"src"}),
                        Span(start_span.start, end.end, start_span.line, start_span.col),
                        "src",
                        next_nid(),
                    )
                items.append(parse_one())
        raise ParseError("unexpected ')'", tok.span)

    forms = []
    while pos < len(tokens):
        forms.append(parse_one())
    if not forms:
        raise ParseError("empty source", Span(0, 0, 1, 1))
    return forms
