"""Lexer + s-expression reader for the tiny language.

Accepted concrete syntax (plain text)::

    <program>     := <form>*
    <form>        := ( <form>* [. <form>] )
                    | id | int | float | "string" | #t | #f
    ints like 1, -3, 999999999999999999999 (overflow stays at origin w/ diag)

Every token keeps its raw representation and a span in
``codepoint-offset:0;line:1;column:1``. Nothing is dropped to make the
picture nicer: unconvertible numbers remain ``?`` atoms with a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .syntax import OriginLink, SObj, Span, Token

INT64_MAX = 2 ** 63 - 1
INT64_MIN = -(2 ** 63)


class FormatError(Exception):
    def __init__(self, message: str, line: int, col: int, offset: int,
                 code: str = "format-error"):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col
        self.offset = offset
        self.code = code

    def to_json(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "col": self.col,
            "offset": self.offset,
        }


@dataclass
class ReadResult:
    forms: list[SObj]
    tokens: list[Token]
    diagnostics: list[dict]


_SIMPLE = set("()'.`,")


def _line_col(text: str, off: int) -> tuple[int, int]:
    line = text.count("\n", 0, off) + 1
    last_nl = text.rfind("\n", 0, off)
    return line, off - last_nl  # 1-based column


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch == ";":
            while i < n and text[i] != "\n":
                i += 1
            continue
        line, col = _line_col(text, i)
        start = i
        if ch == "(":
            tokens.append(Token("punct", "(", Span(start, i + 1, line, col)))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token("punct", ")", Span(start, i + 1, line, col)))
            i += 1
            continue
        if ch == ".":
            # dotted-pair dot only when standalone
            j = i + 1
            if j >= n or text[j] in " \t\r\n()":
                tokens.append(Token("punct", ".", Span(start, i + 1, line, col)))
                i += 1
                continue
        if ch == '"':
            i += 1
            buf: list[str] = []
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    esc = text[i + 1]
                    buf.append({"n": "\n", "t": "\t", '"': '"',
                                "\\": "\\"}.get(esc, esc))
                    i += 2
                else:
                    if text[i] == "\n":
                        l2, c2 = _line_col(text, i)
                        raise FormatError(
                            "unterminated string literal", l2, c2, i,
                            code="unterminated-string")
                    buf.append(text[i])
                    i += 1
            if i >= n:
                l2, c2 = _line_col(text, start)
                raise FormatError(
                    "unterminated string literal", l2, c2, start,
                    code="unterminated-string")
            i += 1  # closing quote
            raw = text[start:i]
            tokens.append(Token(
                "string", "".join(buf),
                Span(start, i, line, col), unit="character-sequence",
                raw=raw))
            continue
        if ch == "#" and i + 1 < n and text[i + 1] in "tf":
            i += 2
            raw = text[start:i]
            tokens.append(Token(
                "bool", raw == "#t", Span(start, i, line, col),
                unit="boolean", raw=raw))
            continue
        # generic atom: read until delimiter
        while i < n and text[i] not in " \t\r\n();":
            i += 1
        raw = text[start:i]
        kind, unit = _classify(raw)
        tokens.append(Token(kind, raw, Span(start, i, line, col),
                            unit=unit, raw=raw))
    return tokens


def _classify(raw: str) -> tuple[str, str]:
    body = raw[1:] if raw[:1] in "+-" and len(raw) > 1 else raw
    if body and body.isdigit():
        return "int", "integer:codepoint"
    if body.replace(".", "", 1).isdigit() and body.count(".") == 1:
        return "float", "float:decimal"
    return "id", "identifier"


def read_program(text: str) -> ReadResult:
    if "\x00" in text:
        raise FormatError("null byte in input", 1, 1, text.index("\x00"),
                          code="illegal-character")
    tokens = tokenize(text)
    diagnostics: list[dict] = []
    pos = 0

    def parse() -> SObj:
        nonlocal pos
        if pos >= len(tokens):
            raise FormatError("unexpected end of input: missing ')'", 0, 0,
                              len(text), code="unexpected-eof")
        tok = tokens[pos]
        if tok.kind == "punct" and tok.text == ")":
            raise FormatError("unexpected ')'", tok.span.line, tok.span.col,
                              tok.span.start, code="unexpected-rparen")
        if tok.kind != "punct" or tok.text != "(":
            pos += 1
            return _atom(tok, diagnostics)
        # list
        pos += 1
        open_tok = tok
        items: list[SObj] = []
        tail: SObj | None = None
        while True:
            if pos >= len(tokens):
                raise FormatError(
                    "unterminated list: missing ')'", open_tok.span.line,
                    open_tok.span.col, open_tok.span.start,
                    code="unterminated-list")
            t = tokens[pos]
            if t.kind == "punct" and t.text == ")":
                pos += 1
                break
            if t.kind == "punct" and t.text == ".":
                if tail is not None or not items:
                    raise FormatError(
                        "misplaced dot", t.span.line, t.span.col,
                        t.span.start, code="misplaced-dot")
                pos += 1
                tail = parse()
                if pos >= len(tokens) or tokens[pos].text != ")":
                    cur = tokens[min(pos, len(tokens) - 1)]
                    raise FormatError(
                        "expected ')' after dotted tail", cur.span.line,
                        cur.span.col, cur.span.start, code="dotted-tail")
                continue
            items.append(parse())
        last = tokens[pos - 1]
        span = Span(open_tok.span.start, last.span.end,
                    open_tok.span.line, open_tok.span.col)
        return SObj.form(items, span,
                         origin=OriginLink("reader"), tail=tail)

    forms: list[SObj] = []
    while pos < len(tokens):
        forms.append(parse())
    return ReadResult(forms=forms, tokens=tokens, diagnostics=diagnostics)


def _atom(tok: Token, diagnostics: list[dict]) -> SObj:
    origin = OriginLink("reader")
    if tok.kind == "int":
        val: object
        try:
            iv = int(tok.text)
            if iv > INT64_MAX or iv < INT64_MIN:
                raise ValueError("overflow")
            val = iv
        except ValueError:
            diagnostics.append({
                "code": "unconvertible-value",
                "message": ("integer literal outside signed 64-bit range; "
                            "kept at origin as unknown"),
                "span": tok.span.to_json(),
                "raw": tok.raw,
            })
            return SObj.leaf("?", "unknown", tok.span, origin)
        return SObj.leaf(val, "int", tok.span, origin)
    if tok.kind == "float":
        return SObj.leaf(float(tok.text), "float", tok.span, origin)
    if tok.kind == "string":
        return SObj.leaf(tok.text, "string", tok.span, origin)
    if tok.kind == "bool":
        return SObj.leaf(tok.text, "bool", tok.span, origin)
    return SObj.leaf(tok.text, "id", tok.span, origin)
