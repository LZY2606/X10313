"""Tokenizer + s-expression reader.

Preserves *every* original token with its span; unconvertible tokens are kept in
place carrying a diagnostic instead of being dropped to make the graph drawable.
"""
from __future__ import annotations

from typing import Optional

from .model import Span, Token, Syntax, symbol_syntax, cid

_DELIMITERS = set("()'")
_TERM_CHARS = set("()';\"`,") | set(" \t\r\n")


class LexError(Exception):
    def __init__(self, message: str, span: Span):
        super().__init__(message)
        self.span = span


def tokenize(text: str, basis: str) -> list[Token]:
    tokens: list[Token] = []
    index = 0
    line = 1
    col = 1

    def advance(ch: str):
        nonlocal index, line, col
        index += 1
        if ch == "\n":
            line += 1
            col = 1
        else:
            col += 1

    n = len(text)
    while index < n:
        ch = text[index]
        if ch in " \t\r\n":
            advance(ch)
            continue
        if ch == ";":
            while index < n and text[index] != "\n":
                advance(text[index])
            continue
        start, sline, scol = index, line, col
        if ch in "()'":
            kind = {"(": "lparen", ")": "rparen", "'": "quote"}[ch]
            tokens.append(Token(cid("tok", (basis, start, end0 := index + 1)),
                                ch, kind, Span(start, start + 1, sline, scol, basis)))
            advance(ch)
            continue
        if ch == '"':
            # string literal with basic escapes
            advance(ch)
            closed = False
            while index < n:
                if text[index] == "\\" and index + 1 < n:
                    advance(text[index]); advance(text[index])
                elif text[index] == '"':
                    advance(text[index]); closed = True
                    break
                else:
                    advance(text[index])
            diag = None if closed else "unterminated string literal"
            tokens.append(Token(cid("tok", (basis, start, index)), text[start:index],
                                "atom" if closed else "error",
                                Span(start, index, sline, scol, basis), diag))
            continue
        # atom: read until whitespace/delimiter
        while index < n and text[index] not in _TERM_CHARS:
            if text[index] == "\\" and index + 1 < n:
                advance(text[index]); advance(text[index])
            else:
                advance(text[index])
        raw = text[start:index]
        tokens.append(Token(cid("tok", (basis, start, index)), raw, "atom",
                            Span(start, index, sline, scol, basis)))
    return tokens


def _atom_datum(raw: str):
    if raw in ("#t", "true"):
        return True
    if raw in ("#f", "false"):
        return False
    try:
        if raw.startswith("0x"):
            return int(raw, 16)
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return ("__string__", bytes(raw[1:-1], "utf-8").decode("unicode_escape"))
    return ("__symbol__", raw)


def read(tokens: list[Token]) -> tuple[list[Syntax], list[Token], list[dict]]:
    """Return (top-level syntax forms, raw tokens in original order, diagnostics).

    Every token is returned even when it forms an error node. Reader diagnostics
    are collected rather than raised so partial input still imports.
    """
    diagnostics: list[dict] = []
    pos = 0

    def err_token(token: Token, message: str):
        diagnostics.append({"kind": "lex", "message": message,
                            "span": token.span.as_dict(), "tid": token.tid})

    for token in tokens:
        if token.kind == "error" and token.diagnostic:
            err_token(token, token.diagnostic)

    def parse_list(open_tok: Token) -> Syntax:
        nonlocal pos
        children: list[Syntax] = []
        while True:
            if pos >= len(tokens):
                diag = (f"unclosed list opened at "
                        f"{open_tok.span.line}:{open_tok.span.col}")
                diagnostics.append({"kind": "read", "message": diag,
                                    "span": open_tok.span.as_dict()})
                return _error_syntax(open_tok, [open_tok], diag)
            token = tokens[pos]
            if token.kind == "rparen":
                pos += 1
                tids = (open_tok.tid, token.tid)
                span = Span(open_tok.span.start, token.span.end,
                            open_tok.span.line, open_tok.span.col,
                            open_tok.span.basis)
                return Syntax(
                    sid=cid("sx", [open_tok.tid, [c.sid for c in children]]),
                    is_pair=True, children=children, token_ids=tids, span=span)
            if token.kind == "lparen":
                pos += 1
                children.append(parse_list(token))
                continue
            if token.kind == "quote":
                pos += 1
                children.append(parse_quote(token))
                continue
            if token.kind == "error":
                pos += 1
                children.append(_error_syntax(token, [token],
                                              token.diagnostic or "unconvertible token"))
                continue
            # atom
            pos += 1
            children.append(_atom_syntax(token))

    def parse_quote(quote_tok: Token) -> Syntax:
        nonlocal pos
        if pos >= len(tokens):
            return _error_syntax(quote_tok, [quote_tok], "quote with no datum")
        token = tokens[pos]
        if token.kind == "lparen":
            pos += 1
            inner = parse_list(token)
        elif token.kind in ("atom", "error"):
            pos += 1
            inner = (_error_syntax(token, [token], token.diagnostic or "bad quote datum")
                     if token.kind == "error" else _atom_syntax(token))
        else:
            pos += 1
            return _error_syntax(quote_tok, [quote_tok], "quote followed by delimiter")
        quoted = symbol_syntax(cid("sx", ["quote", quote_tok.tid]), "quote",
                               token_ids=(quote_tok.tid,), span=quote_tok.span)
        return Syntax(sid=cid("sx", ["quote-list", quote_tok.tid, inner.sid]),
                      is_pair=True, children=[quoted, inner],
                      token_ids=(quote_tok.tid,) + inner.token_ids, span=quote_tok.span)

    forms: list[Syntax] = []
    while pos < len(tokens):
        token = tokens[pos]
        if token.kind == "rparen":
            err_token(token, "unexpected closing parenthesis")
            pos += 1
            forms.append(_error_syntax(token, [token], "unexpected ')'"))
        elif token.kind == "lparen":
            pos += 1
            forms.append(parse_list(token))
        elif token.kind == "quote":
            pos += 1
            forms.append(parse_quote(token))
        elif token.kind == "error":
            pos += 1
            forms.append(_error_syntax(token, [token],
                                       token.diagnostic or "unconvertible token"))
        else:
            pos += 1
            forms.append(_atom_syntax(token))
    return forms, tokens, diagnostics


def _atom_syntax(token: Token) -> Syntax:
    parsed = _atom_datum(token.text)
    origin_step_kind = "token"
    if isinstance(parsed, tuple) and parsed[0] == "__symbol__":
        return symbol_syntax(
            cid("sx", ["id", token.tid]), parsed[1],
            token_ids=(token.tid,), span=token.span,
            origin=())
    value = parsed[1] if isinstance(parsed, tuple) else parsed
    return Syntax(sid=cid("sx", ["lit", token.tid]), value=value,
                  token_ids=(token.tid,), span=token.span)


def _error_syntax(anchor: Token, tokens: list[Token], message: str) -> Syntax:
    end = max((t.span.end for t in tokens), default=anchor.span.end)
    return Syntax(
        sid=cid("sx", ["error", anchor.tid, message]),
        value=None, is_id=False, is_pair=False,
        token_ids=tuple(t.tid for t in tokens),
        span=Span(anchor.span.start, end, anchor.span.line, anchor.span.col,
                  anchor.span.basis),
        context=None)
