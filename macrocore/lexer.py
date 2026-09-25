"""Tokenizer for the small language. Every token keeps a stable id and span."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .syntax import Span


class LexError(Exception):
    def __init__(self, message: str, span: Span):
        super().__init__(message)
        self.message = message
        self.span = span

    def to_data(self) -> dict:
        return {"message": self.message, "span": self.span.to_data(), "phase": "lex"}


@dataclass(frozen=True)
class Token:
    id: str
    kind: str
    text: str
    span: Span

    def to_data(self) -> dict:
        return {"id": self.id, "kind": self.kind, "text": self.text, "span": self.span.to_data()}


_DELIMS = set("()'\" \t\n\r")


def tokenize(src: str, prefix: str = "t") -> List[Token]:
    tokens: List[Token] = []
    i = 0
    line, col = 1, 1
    n = 0

    def advance(k: int = 1) -> None:
        nonlocal i, line, col
        for _ in range(k):
            if i < len(src) and src[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    def emit(kind: str, text: str, start: int, sl: int, sc: int) -> None:
        nonlocal n
        n += 1
        tokens.append(Token(f"{prefix}{n}", kind, text, Span(start, i, sl, sc)))

    while i < len(src):
        ch = src[i]
        if ch in " \t\n\r":
            advance()
            continue
        if ch == ";":
            while i < len(src) and src[i] != "\n":
                advance()
            continue
        start, sl, sc = i, line, col
        if ch == "(":
            advance()
            emit("lparen", ch, start, sl, sc)
        elif ch == ")":
            advance()
            emit("rparen", ch, start, sl, sc)
        elif ch == "'":
            advance()
            emit("quote", ch, start, sl, sc)
        elif ch == '"':
            advance()
            buf = []
            while i < len(src) and src[i] != '"':
                if src[i] == "\n":
                    raise LexError("unterminated string literal", Span(start, i, sl, sc))
                buf.append(src[i])
                advance()
            if i >= len(src):
                raise LexError("unterminated string literal", Span(start, i, sl, sc))
            advance()
            emit("string", "".join(buf), start, sl, sc)
        else:
            buf = []
            while i < len(src) and src[i] not in _DELIMS and src[i] != ";":
                buf.append(src[i])
                advance()
            text = "".join(buf)
            if text in ("#t", "#f"):
                emit("bool", text, start, sl, sc)
            else:
                try:
                    float(text)
                    emit("number", text, start, sl, sc)
                except ValueError:
                    emit("ident", text, start, sl, sc)
    return tokens
