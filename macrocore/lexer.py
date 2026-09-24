"""Lexer + parser for the small S-expression language.

Produces tokens with exact source spans (offsets are utf-8 codepoint offsets,
line/col are 1-based) and parses them into Syntax objects whose provenance
points back to the originating token index.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .syntax import SOURCE_BASIS, Span, Syntax

_TOKEN_RE = re.compile(r"[^\s()]+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class FormatError(Exception):
    """Explicit, reportable format error with a source location."""

    def __init__(self, message: str, line: int, col: int, start: int, end: int):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col
        self.start = start
        self.end = end

    def diagnostic(self) -> dict:
        return {
            "message": self.message,
            "span": {
                "start": self.start,
                "end": self.end,
                "line": self.line,
                "col": self.col,
                "basis": dict(SOURCE_BASIS),
            },
        }


@dataclass
class Token:
    index: int
    kind: str           # lparen | rparen | ident | number | bool
    text: str
    span: Span

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "text": self.text,
            "span": self.span.to_json(),
        }


def lex(source: str) -> List[Token]:
    tokens: List[Token] = []
    i, line, col, index = 0, 1, 1, 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in " \t\r":
            i += 1
            col += 1
            continue
        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if ch == ";":
            while i < n and source[i] != "\n":
                i += 1
                col += 1
            continue
        sline, scol, start = line, col, i
        if ch == "(":
            tokens.append(Token(index, "lparen", "(", Span(i, i + 1, line, col, dict(SOURCE_BASIS))))
            index += 1
            i += 1
            col += 1
            continue
        if ch == ")":
            tokens.append(Token(index, "rparen", ")", Span(i, i + 1, line, col, dict(SOURCE_BASIS))))
            index += 1
            i += 1
            col += 1
            continue
        m = _TOKEN_RE.match(source[i:])
        if not m:  # pragma: no cover - regex always matches at least one char
            raise FormatError(f"unexpected character {ch!r}", line, col, i, i + 1)
        text = m.group(0)
        end = i + len(text)
        if _NUMBER_RE.fullmatch(text):
            kind = "number"
        elif text in ("#t", "#f"):
            kind = "bool"
        else:
            kind = "ident"
        tokens.append(Token(index, kind, text, Span(start, end, sline, scol, dict(SOURCE_BASIS))))
        index += 1
        i = end
        col += len(text)
    return tokens


def parse(tokens: List[Token]) -> List[Syntax]:
    pos = 0

    def parse_one() -> Syntax:
        nonlocal pos
        if pos >= len(tokens):
            raise FormatError("unexpected end of input: missing ')'", 0, 0, 0, 0)
        tok = tokens[pos]
        if tok.kind == "lparen":
            pos += 1
            items: List[Syntax] = []
            while pos < len(tokens) and tokens[pos].kind != "rparen":
                items.append(parse_one())
            if pos >= len(tokens):
                open_tok = tok
                raise FormatError(
                    "unclosed list: missing ')'",
                    open_tok.span.line,
                    open_tok.span.col,
                    open_tok.span.start,
                    open_tok.span.end,
                )
            close = tokens[pos]
            pos += 1
            span = Span(tok.span.start, close.span.end, tok.span.line, tok.span.col, dict(SOURCE_BASIS))
            return Syntax(items, frozenset(), span, (("input", tok.index),))
        if tok.kind == "rparen":
            raise FormatError(
                "unexpected ')'", tok.span.line, tok.span.col, tok.span.start, tok.span.end
            )
        pos += 1
        if tok.kind == "number":
            value: object = float(tok.text) if "." in tok.text else int(tok.text)
        elif tok.kind == "bool":
            value = tok.text == "#t"
        else:
            value = tok.text
        return Syntax(value, frozenset(), tok.span, (("input", tok.index),))

    forms: List[Syntax] = []
    while pos < len(tokens):
        forms.append(parse_one())
    return forms
