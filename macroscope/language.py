"""Lexer and reader for the small source language.

Produces tokens with spans and syntax objects with deterministic identities
(`{sid_prefix}-{n}`), so re-reading the same fragment yields the same ids.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .syntax import Span, Syntax

TOKEN_RE = re.compile(
    r"(?P<ws>\s+)"
    r"|(?P<comment>;[^\n]*)"
    r"|(?P<lpar>\()"
    r"|(?P<rpar>\))"
    r"|(?P<bool>\#t|\#f)"
    r"|(?P<int>[+-]?\d+)"
    r"|(?P<ident>[^\s()]+)"
)


class LanguageError(Exception):
    def __init__(self, message: str, span: Optional[Span] = None):
        super().__init__(message)
        self.message = message
        self.span = span

    def to_diagnostic(self) -> dict:
        return {
            "kind": "format-error",
            "message": self.message,
            "span": self.span.to_obj() if self.span else None,
        }


@dataclass
class Token:
    kind: str
    text: str
    span: Span

    def to_obj(self) -> dict:
        return {"kind": self.kind, "text": self.text, "span": self.span.to_obj()}


def lex(text: str, file: str = "<input>") -> List[Token]:
    tokens: List[Token] = []
    pos, line, col = 0, 1, 1
    n = len(text)
    while pos < n:
        m = TOKEN_RE.match(text, pos)
        if not m:
            span = Span(file, line, col, line, col + 1, pos, pos + 1)
            raise LanguageError(f"unlexable character {text[pos]!r}", span)
        kind = m.lastgroup
        t = m.group()
        s_line, s_col, s_off = line, col, pos
        nls = t.count("\n")
        if nls:
            line += nls
            col = len(t) - t.rfind("\n")
        else:
            col += len(t)
        pos = m.end()
        if kind in ("ws", "comment"):
            continue
        span = Span(file, s_line, s_col, line, col, s_off, pos)
        tokens.append(Token(kind, t, span))
    return tokens


class _Reader:
    def __init__(self, tokens: List[Token], file: str, sid_prefix: str):
        self.tokens = tokens
        self.i = 0
        self.file = file
        self.sid_prefix = sid_prefix
        self.counter = 0

    def _sid(self) -> str:
        self.counter += 1
        return f"{self.sid_prefix}-{self.counter}"

    def read(self) -> Syntax:
        if self.i >= len(self.tokens):
            raise LanguageError("unexpected end of input")
        tok = self.tokens[self.i]
        self.i += 1
        if tok.kind == "lpar":
            children: List[Syntax] = []
            while True:
                if self.i >= len(self.tokens):
                    raise LanguageError("unclosed '('", tok.span)
                if self.tokens[self.i].kind == "rpar":
                    end = self.tokens[self.i].span
                    self.i += 1
                    span = Span(self.file, tok.span.start_line, tok.span.start_col,
                                end.end_line, end.end_col, tok.span.start_offset, end.end_offset)
                    return Syntax(self._sid(), children, span)
                children.append(self.read())
        if tok.kind == "rpar":
            raise LanguageError("unexpected ')'", tok.span)
        if tok.kind == "int":
            return Syntax(self._sid(), int(tok.text), tok.span)
        if tok.kind == "bool":
            return Syntax(self._sid(), tok.text == "#t", tok.span)
        return Syntax(self._sid(), tok.text, tok.span)


def parse(text: str, file: str = "<input>", sid_prefix: str = "syn") -> List[Syntax]:
    """Parse all top-level forms; raises LanguageError on format errors."""
    tokens = lex(text, file)
    reader = _Reader(tokens, file, sid_prefix)
    forms = []
    while reader.i < len(tokens):
        forms.append(reader.read())
    return forms


def parse_one(text: str, file: str = "<input>", sid_prefix: str = "syn") -> Syntax:
    forms = parse(text, file, sid_prefix)
    if len(forms) != 1:
        raise LanguageError(f"expected exactly one form, got {len(forms)}")
    return forms[0]
