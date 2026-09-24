"""Lexer for the tiny macro language.

Every token keeps its *raw* text plus a source span.  Spans carry an explicit
coordinate basis (``line/col/offset``) so downstream layers never have to
guess how to interpret positions.  Tokens that look numeric but cannot be
normalized stay in place as ``raw`` tokens with an attached diagnostic; the
raw representation is never discarded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SPAN_BASIS = "line/col/offset"


@dataclass
class Span:
    source: str
    line: int
    col: int
    offset: int
    length: int
    basis: str = SPAN_BASIS

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "line": self.line,
            "col": self.col,
            "offset": self.offset,
            "length": self.length,
            "basis": self.basis,
        }


@dataclass
class Token:
    tid: str
    kind: str  # lparen | rparen | symbol | number | raw
    text: str  # raw representation, always preserved
    span: Span
    value: object = None  # normalized value (int) when kind == "number"
    diagnostics: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "tid": self.tid,
            "kind": self.kind,
            "text": self.text,
            "span": self.span.to_json(),
            "value": self.value,
            "diagnostics": list(self.diagnostics),
        }


def tokenize(source_id: str, text: str) -> tuple[list[Token], list[dict]]:
    tokens: list[Token] = []
    diagnostics: list[dict] = []
    line, col = 1, 1
    i, n = 0, len(text)
    counter = 0

    def advance(ch: str) -> None:
        nonlocal line, col
        if ch == "\n":
            line += 1
            col = 1
        else:
            col += 1

    while i < n:
        ch = text[i]
        if ch.isspace():
            advance(ch)
            i += 1
            continue
        if ch == ";":  # line comment
            while i < n and text[i] != "\n":
                advance(text[i])
                i += 1
            continue
        start_line, start_col, start_off = line, col, i
        if ch in "()":
            counter += 1
            kind = "lparen" if ch == "(" else "rparen"
            tokens.append(
                Token(f"t{counter}", kind, ch, Span(source_id, start_line, start_col, start_off, 1))
            )
            advance(ch)
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace() and text[j] not in "();":
            j += 1
        raw = text[i:j]
        for k in range(i, j):
            advance(text[k])
        i = j
        counter += 1
        span = Span(source_id, start_line, start_col, start_off, len(raw))
        tok = Token(f"t{counter}", "symbol", raw, span)
        if raw[0].isdigit() or (raw[0] == "-" and len(raw) > 1 and raw[1].isdigit()):
            try:
                tok.value = int(raw)
                tok.kind = "number"
            except ValueError:
                tok.kind = "raw"
                diag = {
                    "code": "cannot-normalize-number",
                    "message": f"token {raw!r} looks numeric but cannot be converted; kept raw in place",
                    "span": span.to_json(),
                }
                tok.diagnostics.append(diag)
                diagnostics.append(diag)
        tokens.append(tok)
    return tokens, diagnostics
