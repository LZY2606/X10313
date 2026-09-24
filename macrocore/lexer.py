"""Lexer for the little language.  Every token keeps its raw source span."""
from __future__ import annotations

from typing import List

from .syntax import Span, Token

_IDENT_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_?!*+-/<>=:")
_IDENT_CONT = _IDENT_START | set("0123456789.")


class LexError(Exception):
    def __init__(self, message: str, span: Span):
        super().__init__(message)
        self.message = message
        self.span = span


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    line, col = 1, 1
    n = len(text)

    def span(start: int, l: int, c: int) -> Span:
        return Span(start, i, l, c)

    while i < n:
        ch = text[i]
        if ch in " \t\r":
            i += 1
            col += 1
            continue
        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if ch == ";" or ch == "#":
            while i < n and text[i] != "\n":
                i += 1
                col += 1
            continue
        start, l, c = i, line, col
        if ch == "(":
            i += 1
            col += 1
            tokens.append(Token("lparen", "(", span(start, l, c)))
            continue
        if ch == ")":
            i += 1
            col += 1
            tokens.append(Token("rparen", ")", span(start, l, c)))
            continue
        if ch == '"':
            i += 1
            col += 1
            buf = []
            while i < n and text[i] != '"':
                if text[i] == "\n":
                    raise LexError("newline in string literal", span(start, l, c))
                buf.append(text[i])
                i += 1
                col += 1
            if i >= n:
                raise LexError("unterminated string literal", span(start, l, c))
            i += 1
            col += 1
            tokens.append(Token("string", "".join(buf), span(start, l, c)))
            continue
        if ch.isdigit() or (ch == "-" and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and text[j].isdigit():
                j += 1
            tok = Token("int", text[i:j], span(start, l, c))
            col += j - i
            i = j
            tokens.append(tok)
            continue
        if ch in _IDENT_START:
            j = i
            while j < n and text[j] in _IDENT_CONT:
                j += 1
            tok = Token("ident", text[i:j], span(start, l, c))
            col += j - i
            i = j
            tokens.append(tok)
            continue
        raise LexError(f"unexpected character {ch!r}", span(start, l, c))
    return tokens
