"""Syntax objects: tokens, source spans and scope-carrying syntax nodes.

Every datum read from source becomes a Syntax node carrying:
  - a stable node id (stable identity used by the UI and the archive)
  - a source span (offset + line/col coordinate basis)
  - a scope set (frozenset of scope ids) used for hygienic binding
  - an origin tag ('source' | 'macro' | 'generated')
  - a provenance pointer (id of the expansion step that introduced it)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Span:
    """Source span. Coordinate basis: char offsets plus 1-based line/col."""
    start: int
    end: int
    line: int
    col: int

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end,
                "line": self.line, "col": self.col, "basis": "offset+linecol"}


@dataclass(frozen=True)
class Token:
    kind: str  # 'lparen' | 'rparen' | 'ident' | 'number' | 'string'
    text: str
    span: Span

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "span": self.span.to_dict()}


@dataclass(frozen=True)
class Syntax:
    id: int
    datum: Any  # str | int | float | bool | list[Syntax]
    scopes: frozenset
    span: Optional[Span]
    origin: str
    prov: Optional[int]  # expansion step id that introduced this node

    @property
    def is_ident(self) -> bool:
        return isinstance(self.datum, str)


class LexError(Exception):
    def __init__(self, message: str, span: Optional[Span] = None):
        super().__init__(message)
        self.message = message
        self.span = span


class NodeIds:
    """Deterministic node id allocator (no randomness; fixed-seed friendly)."""

    def __init__(self, start: int = 1):
        self.next = start

    def take(self) -> int:
        value = self.next
        self.next += 1
        return value


_ATOM_END = set("()\"; \t\r\n")


def lex(source: str) -> list[Token]:
    tokens: list[Token] = []
    i, line, col = 0, 1, 1
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
        if ch == ";":  # comment to end of line
            while i < n and source[i] != "\n":
                i += 1
                col += 1
            continue
        if ch in "()":
            tokens.append(Token("lparen" if ch == "(" else "rparen", ch,
                                Span(i, i + 1, line, col)))
            i += 1
            col += 1
            continue
        if ch == '"':
            j = i + 1
            buf: list[str] = []
            while j < n and source[j] != '"':
                if source[j] == "\\" and j + 1 < n:
                    j += 1
                if source[j] == "\n":
                    raise LexError("newline in string literal",
                                   Span(i, j, line, col))
                buf.append(source[j])
                j += 1
            if j >= n:
                raise LexError("unterminated string literal",
                               Span(i, j, line, col))
            tokens.append(Token("string", "".join(buf),
                                Span(i, j + 1, line, col)))
            col += (j + 1 - i)
            i = j + 1
            continue
        j = i
        while j < n and source[j] not in _ATOM_END:
            j += 1
        text = source[i:j]
        kind = "ident"
        try:
            int(text)
            kind = "number"
        except ValueError:
            try:
                float(text)
                kind = "number"
            except ValueError:
                pass
        tokens.append(Token(kind, text, Span(i, j, line, col)))
        col += j - i
        i = j
    return tokens


def _atom_value(tok: Token) -> Any:
    if tok.kind == "number":
        try:
            return int(tok.text)
        except ValueError:
            return float(tok.text)
    if tok.kind == "string":
        return tok.text
    if tok.text == "#t":
        return True
    if tok.text == "#f":
        return False
    return tok.text


def read(source: str, ids: Optional[NodeIds] = None):
    """Read source into a list of toplevel Syntax forms.

    Returns (forms, tokens, ids). All nodes get origin='source', empty scope
    set and prov=None; the expander re-allocates ids for introduced nodes.
    """
    ids = ids or NodeIds()
    tokens = lex(source)
    pos = 0

    def parse() -> Syntax:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok.kind == "lparen":
            items: list[Syntax] = []
            while True:
                if pos >= len(tokens):
                    raise LexError("unclosed '('", tok.span)
                if tokens[pos].kind == "rparen":
                    end = tokens[pos].span
                    pos += 1
                    return Syntax(ids.take(), items, frozenset(),
                                  Span(tok.span.start, end.end,
                                       tok.span.line, tok.span.col),
                                  "source", None)
                items.append(parse())
        if tok.kind == "rparen":
            raise LexError("unexpected ')'", tok.span)
        return Syntax(ids.take(), _atom_value(tok), frozenset(),
                      tok.span, "source", None)

    forms: list[Syntax] = []
    while pos < len(tokens):
        forms.append(parse())
    return forms, tokens, ids
