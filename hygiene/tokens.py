"""Lexical layer.

The original token stream is the first immutable material layer: every token
keeps its raw text and a character span over the submitted source. Nothing in
later layers ever mutates a token; syntax objects point back at tokens by
stable id. Malformed input raises LexError / ParseError carrying a span, and
the importing layer turns those into explicit format diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    lo: int
    hi: int
    line: int
    col: int

    def as_dict(self) -> dict:
        return {"lo": self.lo, "hi": self.hi, "line": self.line, "col": self.col}


@dataclass(frozen=True)
class Token:
    tid: int
    kind: str          # lparen rparen lbracket rbracket quote boolean number string symbol dot ellipsis comment
    text: str
    span: Span

    def as_dict(self) -> dict:
        return {"tid": self.tid, "kind": self.kind, "text": self.text, "span": self.span.as_dict()}


class LexError(Exception):
    def __init__(self, msg: str, span: Span):
        super().__init__(msg)
        self.msg = msg
        self.span = span


_DELIMS = set("()[]'`\",;\n")


def _span_at(src: str, pos: int) -> Span:
    line = src.count("\n", 0, pos) + 1
    last_nl = src.rfind("\n", 0, pos)
    col = pos - last_nl
    return Span(pos, pos, line, col)


def tokenize(src: str) -> list[Token]:
    toks: list[Token] = []
    i, n = 0, len(src)
    tid = 0
    while i < n:
        c = src[i]
        if c in " \t\r\n":
            i += 1
            continue
        start = i
        sp = _span_at(src, i)
        if c == ";":
            j = i + 1
            while j < n and src[j] != "\n":
                j += 1
            toks.append(Token(tid, "comment", src[i:j], Span(start, j, sp.line, sp.col)))
            tid += 1
            i = j
            continue
        if c == "'":
            toks.append(Token(tid, "quote", "'", Span(start, i + 1, sp.line, sp.col)))
            tid += 1
            i += 1
            continue
        if c in "([":
            toks.append(Token(tid, "lparen" if c == "(" else "lbracket", c,
                              Span(start, i + 1, sp.line, sp.col)))
            tid += 1
            i += 1
            continue
        if c in ")]":
            toks.append(Token(tid, "rparen" if c == ")" else "rbracket", c,
                              Span(start, i + 1, sp.line, sp.col)))
            tid += 1
            i += 1
            continue
        if c == '"':
            j = i + 1
            buf = ['"']
            while j < n and src[j] != '"':
                if src[j] == "\\" and j + 1 < n:
                    buf.append(src[j:j + 2])
                    j += 2
                    continue
                if src[j] == "\n":
                    raise LexError("unterminated string literal (newline before closing quote)",
                                   Span(start, j, sp.line, sp.col))
                buf.append(src[j])
                j += 1
            if j >= n:
                raise LexError("unterminated string literal (no closing quote)",
                               Span(start, n, sp.line, sp.col))
            buf.append('"')
            j += 1
            toks.append(Token(tid, "string", "".join(buf), Span(start, j, sp.line, sp.col)))
            tid += 1
            i = j
            continue
        j = i
        while j < n and src[j] not in " \t\r\n()[]';\"":
            j += 1
        word = src[i:j]
        if word == "...":
            kind = "ellipsis"
        elif word in ("#t", "#f"):
            kind = "boolean"
        elif word == ".":
            kind = "dot"
        elif _is_number(word):
            kind = "number"
        else:
            kind = "symbol"
            _validate_symbol(word, sp)
        toks.append(Token(tid, kind, word, Span(start, j, sp.line, sp.col)))
        tid += 1
        i = j
    return toks


def _is_number(word: str) -> bool:
    try:
        int(word)
        return True
    except ValueError:
        try:
            float(word)
            return True
        except ValueError:
            return False


def _validate_symbol(word: str, sp: Span) -> None:
    if word.startswith("#"):
        raise LexError(f"unknown token {word!r} (only #t / #f booleans are supported)", sp)
    if word.startswith('"'):
        raise LexError("malformed string token", sp)
    for ch in word:
        if ch in "`," or ord(ch) < 0x20:
            raise LexError(f"illegal character {ch!r} in symbol", sp)


@dataclass
class ParseError(Exception):
    msg: str
    span: Span
    tokens_seen: int = field(default=0)

    def __str__(self) -> str:
        return f"{self.msg} at line {self.span.line}:{self.span.col}"
