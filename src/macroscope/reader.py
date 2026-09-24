"""Tokenizer and reader for the small language.

Preserves *every* original token with an explicit :class:`Span`.  Values
that cannot be converted (for example ``12abc``) stay in place as ``raw``
nodes carrying a diagnostic rather than being dropped to make a prettier
picture.  Format errors are reported with token-level spans.
"""
from __future__ import annotations

from typing import Any, Optional

from .syntax import Diagnostic, GraphBuilder, Span, Stx, Token

_SPECIAL = {"(": "(", ")": ")", "{": "{", "}": "}",
            "'": "'", "`": "`", ",": ", ", ",@": ",@"}

_PREFIX_WRAP = {"'": "quote", "`": "quasiquote", ",": "unquote",
                ",@": "unquote-splicing"}


class ReaderError(Exception):
    def __init__(self, message: str, diagnostics: list[Diagnostic]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _span(file: str, text: str, start: int, length: int) -> Span:
    line = text.count("\n", 0, start) + 1
    last_nl = text.rfind("\n", 0, start)
    col = start - last_nl
    return Span(file=file, offset=start, line=line, col=col, length=length)


def tokenize(text: str, file: str = "input") -> tuple[list[Token], list[Diagnostic]]:
    tokens: list[Token] = []
    diags: list[Diagnostic] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == ";":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c in "(){}":
            kind = {"(": "(", ")": ")", "{": "{", "}": "}"}[c]
            tokens.append(Token(len(tokens), kind, c, _span(file, text, i, 1)))
            i += 1
            continue
        if c == ",":
            if i + 1 < n and text[i + 1] == "@":
                tokens.append(Token(len(tokens), ",@", ",@",
                                    _span(file, text, i, 2)))
                i += 2
            else:
                tokens.append(Token(len(tokens), ",", ",",
                                    _span(file, text, i, 1)))
                i += 1
            continue
        if c in "'`":
            tokens.append(Token(len(tokens), c, c, _span(file, text, i, 1)))
            i += 1
            continue
        if c == '"':
            start = i
            i += 1
            buf: list[str] = []
            closed = False
            while i < n:
                ch = text[i]
                if ch == "\\" and i + 1 < n:
                    buf.append(ch + text[i + 1])
                    i += 2
                    continue
                if ch == '"':
                    closed = True
                    i += 1
                    break
                if ch == "\n":
                    break
                buf.append(ch)
                i += 1
            raw = text[start:i]
            sp = _span(file, text, start, len(raw))
            if not closed:
                diags.append(Diagnostic("E101", "unterminated string literal",
                                        span=sp))
            tokens.append(Token(len(tokens), "atom", raw, sp))
            continue
        if c in "[]":
            diags.append(Diagnostic("E102",
                                    f"square brackets are not supported: {c!r} "
                                    "(use () or {})", "error",
                                    _span(file, text, i, 1)))
            i += 1
            continue
        # generic atom
        start = i
        while i < n and text[i] not in ' \t\r\n()[]{}\'"`,;':
            i += 1
        if i == start:
            diags.append(Diagnostic("E100", f"unexpected character {c!r}",
                                    "error", _span(file, text, i, 1)))
            i += 1
            continue
        word = text[start:i]
        tokens.append(Token(len(tokens), "atom", word,
                            _span(file, text, start, len(word))))
    return tokens, diags


def _atom(builder: GraphBuilder, tok: Token, diags: list[Diagnostic]) -> Stx:
    w = tok.value
    if w.startswith('"'):
        inner = w[1:-1] if len(w) >= 2 and w.endswith('"') else w[1:]
        try:
            value = bytes(inner, "utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            value = inner
        return builder.node(kind="string", value=value, lexeme=w,
                            span=tok.span, reason="read")
    if w in ("#true", "#false"):
        return builder.node(kind="bool", value=(w == "#true"), lexeme=w,
                            span=tok.span, reason="read")
    try:
        num = int(w)
        return builder.node(kind="number", value=num, lexeme=w,
                            span=tok.span, reason="read")
    except ValueError:
        pass
    try:
        num = float(w)
        return builder.node(kind="number", value=num, lexeme=w,
                            span=tok.span, reason="read")
    except ValueError:
        pass
    if w[0].isdigit():
        d = Diagnostic("E110", f"unconvertible numeric-looking atom {w!r}; "
                               "kept in place as raw (not dropped)",
                       "warning", tok.span)
        diags.append(d)
        return builder.node(kind="raw", value=w, lexeme=w, span=tok.span,
                            reason="read-raw", flags=("diagnostic",))
    return builder.node(kind="id", value=w, lexeme=w, span=tok.span,
                        reason="read")


def read_program(text: str, builder: GraphBuilder, file: str = "input"
                 ) -> dict[str, Any]:
    tokens, diags = tokenize(text, file)

    pos = 0

    def parse_delimited(open_tok: Token) -> Stx:
        nonlocal pos
        close = ")" if open_tok.kind == "(" else "}"
        bracket = "()" if close == ")" else "{}"
        items: list[Stx] = []
        while pos < len(tokens):
            tok = tokens[pos]
            if tok.kind in (")", "}"):
                if tok.kind != close:
                    diags.append(Diagnostic(
                        "E120", f"mismatched bracket: expected {close}, "
                                f"got {tok.kind}", span=tok.span))
                pos += 1
                return builder.node(
                    kind="seq", lexeme=bracket, span=open_tok.span,
                    reason="read-seq", seq=tuple(items), bracket=bracket)
            items.append(parse_form())
        diags.append(Diagnostic(
            "E121", f"unbalanced {open_tok.kind}: missing {close}",
            span=open_tok.span))
        return builder.node(kind="seq", lexeme=bracket, span=open_tok.span,
                            reason="read-seq-incomplete", seq=tuple(items),
                            bracket=bracket, flags=("diagnostic",))

    def parse_form() -> Stx:
        nonlocal pos
        tok = tokens[pos]
        if tok.kind in ("(", "{"):
            pos += 1
            return parse_delimited(tok)
        if tok.kind in (")", "}"):
            diags.append(Diagnostic("E122",
                                    f"unexpected {tok.kind}", span=tok.span))
            pos += 1
            return builder.node(kind="raw", value=tok.value, lexeme=tok.value,
                                span=tok.span, reason="read-stray-close",
                                flags=("diagnostic",))
        if tok.kind in ("'", "`", ",", ",@"):
            pos += 1
            if pos >= len(tokens) or tokens[pos].kind in (")", "}", ",", ",@"):
                diags.append(Diagnostic(
                    "E123", f"{tok.value!r} is missing its operand",
                    span=tok.span))
                return builder.node(kind="raw", value=tok.value,
                                    lexeme=tok.value, span=tok.span,
                                    reason="read-dangling-prefix",
                                    flags=("diagnostic",))
            operand = parse_form()
            wrap = _PREFIX_WRAP[tok.kind]
            head = builder.node(kind="id", value=wrap, lexeme=wrap,
                                span=tok.span, reason="read-prefix")
            return builder.node(kind="seq", lexeme="()", span=tok.span,
                                reason=f"read-prefix {tok.kind}",
                                seq=(head, operand))
        pos += 1
        return _atom(builder, tok, diags)

    forms: list[Stx] = []
    while pos < len(tokens):
        forms.append(parse_form())

    root = builder.node(kind="seq", lexeme="<program>", span=None,
                        reason="read-program", seq=tuple(forms), bracket="()")
    if diags:
        builder.log_step("reader-diagnostics",
                         diagnostics=[d.as_dict() for d in diags])
    return {"root": root, "tokens": tokens, "diagnostics": diags}
