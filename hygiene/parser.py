"""Token stream -> syntax objects (s-expressions).

Every parsed object is registered with a ``parsed`` birth certificate that
points at its original token; list spans cover the whole form. A leading
program scope is stamped on every object so that definition-site references
inside macro templates resolve the way the macro author wrote them.
"""
from __future__ import annotations

from .syntax import Syntax, Birth, SyntaxRegistry
from .tokens import Token, tokenize, ParseError, Span


def parse(src: str, reg: SyntaxRegistry, program_scope: str):
    toks = tokenize(src)
    pos = 0
    forms: list[Syntax] = []
    while pos < len(toks):
        if toks[pos].kind == "comment":
            pos += 1
            continue
        node, pos = _parse_one(toks, pos, reg)
        forms.append(node)
    _stamp_scope(forms, program_scope)
    return toks, forms


def _stamp_scope(forms, scope_id: str) -> None:
    stack = list(forms)
    while stack:
        s = stack.pop()
        s.add_scope(scope_id)
        if s.is_list:
            stack.extend(s.val)


def _parse_one(toks: list[Token], pos: int, reg: SyntaxRegistry):
    t = toks[pos]
    if t.kind in ("lparen", "lbracket"):
        close = "rparen" if t.kind == "lparen" else "rbracket"
        start = t.span
        elems: list[Syntax] = []
        pos += 1
        while pos < len(toks) and toks[pos].kind != close:
            if toks[pos].kind in ("rparen", "rbracket"):
                bad = toks[pos]
                raise ParseError("mismatched closing bracket", bad.span, pos)
            if toks[pos].kind == "comment":
                pos += 1
                continue
            elem, pos = _parse_one(toks, pos, reg)
            elems.append(elem)
        if pos >= len(toks):
            raise ParseError("unclosed list: missing closing bracket", start, pos)
        end = toks[pos]
        pos += 1
        node = Syntax(elems, True, Span(start.lo, end.span.hi, start.line, start.col),
                      birth=Birth("parsed", detail="list", token_id=t.tid))
        reg.record(node)
        return node, pos
    if t.kind in ("rparen", "rbracket"):
        raise ParseError("unexpected closing bracket", t.span, pos)
    if t.kind == "quote":
        if pos + 1 >= len(toks):
            raise ParseError("quote without an expression", t.span, pos)
        quoted, newpos = _parse_one(toks, pos + 1, reg)
        node = Syntax([_atom("quote", t, reg), quoted], True,
                      Span(t.span.lo, quoted.span.hi, t.span.line, t.span.col),
                      birth=Birth("parsed", detail="quote", token_id=t.span.lo))
        reg.record(node)
        return node, newpos
    if t.kind == "dot":
        raise ParseError("illegal '.' token outside dotted lists (unsupported)", t.span, pos)
    if t.kind == "ellipsis":
        raise ParseError("'...' is only allowed inside macro patterns/templates", t.span, pos)
    node = _atom_from_token(t, reg)
    return node, pos + 1
def _atom_from_token(t: Token, reg: SyntaxRegistry) -> Syntax:
    if t.kind == "symbol":
        val = t.text
    elif t.kind == "number":
        val = int(t.text) if _is_int(t.text) else float(t.text)
    elif t.kind == "boolean":
        val = t.text == "#t"
    elif t.kind == "string":
        val = t.text[1:-1]
    elif t.kind == "comment":
        raise ParseError("comment is not an expression", t.span, t.tid)
    else:
        raise ParseError(f"unexpected token {t.text!r}", t.span, t.tid)
    s = Syntax(val, False, t.span, birth=Birth("parsed", detail=f"token:{t.kind}",
                                               token_id=t.tid))
    reg.record(s)
    return s


def _atom(name: str, t: Token, reg: SyntaxRegistry) -> Syntax:
    s = Syntax(name, False, t.span, birth=Birth("parsed", detail="quote-symbol",
                                                token_id=t.tid))
    reg.record(s)
    return s


def _is_int(text: str) -> bool:
    try:
        int(text)
        return True
    except ValueError:
        return False
