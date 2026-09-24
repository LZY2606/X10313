"""Tokenizer for the small language.

Every original token, its raw text and its source span are retained. Malformed
input produces ``error`` tokens with diagnostics rather than being silently
dropped, so the caller can still import and review it.
"""
from __future__ import annotations

from .syntax import Span, Token

DELIMITERS = set("()'")
SEPARATORS = set(" \t\r\n,;\"`")


class LexError(Exception):
    def __init__(self, message: str, span: Span):
        super().__init__(message)
        self.message = message
        self.span = span


def _is_number(text: str) -> bool:
    body = text[1:] if text[:1] in "+-" else text
    if not body or not body.replace(".", "", 1).isdigit():
        return False
    return body.count(".") <= 1


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    lines = source.splitlines()
    line_starts: list[int] = []
    pos = 0
    for line in source.split("\n"):
        line_starts.append(pos)
        pos += len(line) + 1

    def span_at(start: int, end: int) -> Span:
        sl = source.count("\n", 0, start)
        prefix = source[line_starts[sl]:start]
        el = source.count("\n", 0, max(start, end - 1))
        eprefix = source[line_starts[el]:end]
        return Span(sl + 1, len(prefix), el + 1, len(eprefix),
                    start, max(1, end - start))

    i = 0
    index = 0
    n = len(source)
    while i < n:
        char = source[i]
        if char in " \t\r\n":
            i += 1
            continue
        if char == ";":
            while i < n and source[i] != "\n":
                i += 1
            continue
        start = i
        if char in "()'":
            kind = {"(": "lparen", ")": "rparen", "'": "quote"}[char]
            tokens.append(Token(char, kind, span_at(start, i + 1), index))
            index += 1
            i += 1
            continue
        if char in '",`':
            tokens.append(Token(char, "error", span_at(start, i + 1), index,
                                diagnostic=(
                                    f"unsupported literal character {char!r}; "
                                    "only ( ) ' and bare atoms are allowed")))
            index += 1
            i += 1
            continue
        while i < n and source[i] not in SEPARATORS and source[i] not in DELIMITERS:
            i += 1
        text = source[start:i]
        kind = "number" if _is_number(text) else "atom"
        tokens.append(Token(text, kind, span_at(start, i), index))
        index += 1
    return tokens


def parse(tokens: list[Token]):
    """Parse tokens into nested syntax lists. Returns (forms, diagnostics)."""
    diagnostics: list[dict] = []
    cursor = 0

    def take_form() -> list | Token:
        nonlocal cursor
        if cursor >= len(tokens):
            raise LexError("unexpected end of input: missing a closing ')'",
                           tokens[-1].span if tokens else None)
        tok = tokens[cursor]
        if tok.kind == "error":
            diagnostics.append({
                "severity": "error", "message": tok.diagnostic or "bad token",
                "span": tok.span.to_json(), "token_index": tok.index,
                "raw": tok.raw,
            })
            cursor += 1
            return tok
        if tok.kind == "lparen":
            open_tok = tok
            cursor += 1
            items = []
            while cursor < len(tokens) and tokens[cursor].kind != "rparen":
                if tokens[cursor].kind == "lparen":
                    items.append(take_form())
                else:
                    items.append(tokens[cursor])
                    cursor += 1
            if cursor >= len(tokens):
                diagnostics.append({
                    "severity": "error",
                    "message": "unbalanced parentheses: missing ')'",
                    "span": open_tok.span.to_json(),
                    "token_index": open_tok.index, "raw": open_tok.raw,
                })
                return items
            cursor += 1  # consume rparen
            return items
        if tok.kind == "rparen":
            raise LexError("unexpected ')'", tok.span)
        if tok.kind == "quote":
            cursor += 1
            quoted = take_form()
            return [Token("quote", "atom", tok.span, tok.index), quoted]
        cursor += 1
        return tok

    forms = []
    while cursor < len(tokens):
        try:
            forms.append(take_form())
        except LexError as exc:
            diagnostics.append({
                "severity": "error", "message": exc.message,
                "span": exc.span.to_json() if exc.span else None,
            })
            cursor += 1
    return forms, diagnostics
