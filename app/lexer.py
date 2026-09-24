"""Lexer for the small Scheme-flavoured language.

Every token keeps its original textual representation and a precise source
span.  Numeric-looking tokens are kept as raw strings here; conversion only
happens in the derived layer (``app.interpreter``), where a value that cannot
be converted stays in place with a diagnostic instead of being dropped.
"""
from __future__ import annotations

from .models import Span, Token

SUGAR = {"'": "quote", "`": "quasiquote", ",": "unquote", ",@": "unquote-splicing"}
TERMINATORS = set("() \t\r\n")


class LexError(Exception):
    def __init__(self, message: str, span: Span):
        super().__init__(message)
        self.message = message
        self.span = span


def _span(file_id: str, src: str, start: int, end: int) -> Span:
    line = src.count("\n", 0, start) + 1
    nl = src.rfind("\n", 0, start)
    col = start - nl if nl < 0 else start - nl
    return Span(file_id, start, end, line, col)


def tokenize(src: str, file_id: str = "input") -> list[Token]:
    out: list[Token] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == ";":
            while i < n and src[i] != "\n":
                i += 1
            continue
        start = i
        if c in "()":
            out.append(Token("lparen" if c == "(" else "rparen", c,
                             _span(file_id, src, start, i + 1)))
            i += 1
            continue
        if c == "'" or c == "`":
            out.append(Token("sugar", c, _span(file_id, src, start, i + 1),
                             raw=SUGAR[c]))
            i += 1
            continue
        if c == ",":
            if i + 1 < n and src[i + 1] == "@":
                out.append(Token("sugar", ",@", _span(file_id, src, start, i + 2),
                                 raw=SUGAR[",@"]))
                i += 2
            else:
                out.append(Token("sugar", ",", _span(file_id, src, start, i + 1),
                                 raw=SUGAR[","]))
                i += 1
            continue
        if c == ".":
            j = i + 1
            if j < n and src[j] not in TERMINATORS and src[j] != ";":
                while j < n and src[j] not in TERMINATORS and src[j] != ";":
                    j += 1
                text = src[i:j]
                out.append(Token("symbol", text, _span(file_id, src, i, j), raw=text))
            else:
                out.append(Token("dot", ".", _span(file_id, src, i, j)))
            i = j
            continue
        if c == '"':
            i += 1
            body: list[str] = []
            closed = False
            while i < n:
                ch = src[i]
                if ch == "\\":
                    body.append(src[i:i + 2])
                    i += 2
                    continue
                if ch == '"':
                    closed = True
                    i += 1
                    break
                if ch == "\n":
                    sp = _span(file_id, src, start, i)
                    raise LexError("unterminated string: newline before closing quote", sp)
                body.append(ch)
                i += 1
            if not closed:
                raise LexError("unterminated string", _span(file_id, src, start, i))
            raw = src[start:i]
            out.append(Token("string", "".join(body),
                             _span(file_id, src, start, i), raw=raw))
            continue
        if c in "#" and i + 1 < n and src[i + 1] in "tf":
            out.append(Token("boolean", src[i:i + 2],
                             _span(file_id, src, i, i + 2), raw=src[i:i + 2]))
            i += 2
            continue
        # generic atom run
        j = i
        while j < n and src[j] not in TERMINATORS and src[j] not in "'\",;" and not (
                src[j] == "#" and j > i and src[j - 1] in TERMINATORS):
            j += 1
        text = src[i:j]
        if not text:
            raise LexError(f"unexpected character {c!r}", _span(file_id, src, i, i + 1))
        kind = _classify(text)
        out.append(Token(kind, text, _span(file_id, src, i, j), raw=text))
        i = j
    return out


def _classify(text: str) -> str:
    if text[:2] in ("#t", "#f") and (len(text) == 2 or not text[2].isdigit()):
        return "boolean"
    first = text[0]
    if first.isdigit() or (first in "+-" and len(text) > 1 and text[1].isdigit()) \
            or (first == "." and len(text) > 1 and text[1].isdigit()):
        return "number"
    return "symbol"
