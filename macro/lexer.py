"""Tokenizer. Every token keeps its source span with a coordinate basis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .syntax import Span


@dataclass
class Token:
    kind: str          # 'lparen' | 'rparen' | 'symbol' | 'number' | 'bool'
    text: str
    span: Span


class LexError(Exception):
    def __init__(self, message: str, span: Span):
        super().__init__(message)
        self.span = span


def lex(source_id: str, text: str, basis: str = "utf8-offset") -> List[Token]:
    tokens: List[Token] = []
    i, line, col = 0, 1, 1
    n = len(text)

    def span(start, end, ln, cl):
        return Span(source_id, start, end, ln, cl, basis)

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
        if ch == ";":                      # line comment
            while i < n and text[i] != "\n":
                i += 1
            continue
        start, ln, cl = i, line, col
        if ch == "(":
            tokens.append(Token("lparen", "(", span(start, start + 1, ln, cl)))
            i += 1; col += 1
            continue
        if ch == ")":
            tokens.append(Token("rparen", ")", span(start, start + 1, ln, cl)))
            i += 1; col += 1
            continue
        # atom: symbol or number
        j = i
        while j < n and text[j] not in " \t\r\n();":
            j += 1
        word = text[i:j]
        kind = "symbol"
        if word in ("#t", "#f"):
            kind = "bool"
        else:
            try:
                float(word)
                kind = "number"
            except ValueError:
                kind = "symbol"
        tokens.append(Token(kind, word, span(start, j, ln, cl)))
        col += j - i
        i = j
    return tokens
