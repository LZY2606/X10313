"""Tokenizer for the MGL macro language.

Every token keeps its source span with an explicit coordinate basis
(``basis``).  Raw representations are preserved at import time; any
normalisation happens only in the derived layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BASIS_CHAR = "char-offset"       # 0-based unicode code-point offset, 1-based line/col
BASIS_BYTE = "byte-offset"       # utf-8 byte offset
KNOWN_BASES = {BASIS_CHAR, BASIS_BYTE}


@dataclass
class Span:
    start: int
    end: int
    line: int
    col: int
    basis: str = BASIS_CHAR

    def to_json(self) -> dict:
        return {"start": self.start, "end": self.end, "line": self.line,
                "col": self.col, "basis": self.basis}

    @staticmethod
    def from_json(d: dict) -> "Span":
        return Span(d["start"], d["end"], d["line"], d["col"],
                    d.get("basis", BASIS_CHAR))


@dataclass
class Token:
    tid: str
    kind: str            # lparen | rparen | ident | number | string
    text: str
    span: Span

    def to_json(self) -> dict:
        return {"tid": self.tid, "kind": self.kind, "text": self.text,
                "span": self.span.to_json()}


@dataclass
class Diagnostic:
    code: str
    message: str
    span: dict | None = None
    severity: str = "error"

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message,
                "span": self.span, "severity": self.severity}


_IDENT_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_+-*/!?<>=%$&^~:")
_IDENT_CONT = _IDENT_START | set("0123456789.#@")


def tokenize(source: str) -> tuple[list[Token], list[Diagnostic]]:
    tokens: list[Token] = []
    diags: list[Diagnostic] = []
    i, line, col, n = 0, 1, 1, 0
    L = len(source)

    def emit(kind: str, text: str, start: int, sl: int, sc: int) -> None:
        nonlocal n
        tokens.append(Token(f"tok_{n:04d}", kind, text,
                            Span(start, i, sl, sc)))
        n += 1

    while i < L:
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
        if ch == ";":  # comment to end of line
            while i < L and source[i] != "\n":
                i += 1
            continue
        start, sl, sc = i, line, col
        if ch == "(":
            i += 1
            col += 1
            emit("lparen", ch, start, sl, sc)
            continue
        if ch == ")":
            i += 1
            col += 1
            emit("rparen", ch, start, sl, sc)
            continue
        if ch == '"':
            i += 1
            col += 1
            buf = []
            closed = False
            while i < L:
                c = source[i]
                if c == "\\" and i + 1 < L:
                    buf.append(source[i + 1])
                    i += 2
                    col += 2
                    continue
                if c == '"':
                    closed = True
                    i += 1
                    col += 1
                    break
                if c == "\n":
                    line += 1
                    col = 1
                    i += 1
                    buf.append(c)
                    continue
                buf.append(c)
                i += 1
                col += 1
            if not closed:
                diags.append(Diagnostic(
                    "unterminated-string",
                    f"string literal opened at line {sl} col {sc} is never closed",
                    Span(start, i, sl, sc).to_json()))
                continue
            emit("string", "".join(buf), start, sl, sc)
            continue
        if ch.isdigit() or (ch == "-" and i + 1 < L and source[i + 1].isdigit()):
            j = i
            while j < L and (source[j].isdigit() or source[j] in ".-"):
                j += 1
            text = source[i:j]
            col += j - i
            i = j
            emit("number", text, start, sl, sc)
            continue
        if ch in _IDENT_START:
            j = i
            while j < L and source[j] in _IDENT_CONT:
                j += 1
            text = source[i:j]
            col += j - i
            i = j
            emit("ident", text, start, sl, sc)
            continue
        diags.append(Diagnostic(
            "illegal-character",
            f"illegal character {ch!r} at line {line} col {col}",
            Span(start, start + 1, line, col).to_json()))
        i += 1
        col += 1
    return tokens, diags
