"""Core data model: tokens, source spans, syntax objects, domain quantities.

Every domain quantity carries a unit or a coordinate basis.  The import layer
keeps the raw representation; only the derived layer normalizes.  Values that
cannot be converted stay in place with a diagnostic attached.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

SPAN_BASIS = "char@line1col1"
SCOPE_BASIS = "scope-id@expansion-step"
STEP_UNIT = "expansion-step"
TICK_UNIT = "virtual-tick"


# ---------------------------------------------------------------- spans
@dataclass(frozen=True)
class Span:
    offset: int
    line: int
    col: int
    basis: str = SPAN_BASIS

    def to_dict(self):
        return {"offset": self.offset, "line": self.line,
                "col": self.col, "basis": self.basis}

    @staticmethod
    def from_dict(d):
        return Span(int(d["offset"]), int(d["line"]), int(d["col"]),
                    d.get("basis", SPAN_BASIS))


def span_dict(span):
    return span.to_dict() if span is not None else None


# ---------------------------------------------------------------- tokens
@dataclass(frozen=True)
class Token:
    tid: str
    text: str
    span: Span

    def to_dict(self):
        return {"tid": self.tid, "text": self.text, "span": self.span.to_dict()}


class LexError(Exception):
    def __init__(self, message, span):
        super().__init__(message)
        self.span = span


class ParseError(Exception):
    def __init__(self, message, span):
        super().__init__(message)
        self.span = span


def lex(source: str):
    """Tokenize, keeping the original text and span of every token."""
    tokens = []
    i, line, col = 0, 1, 1
    counter = itertools.count()
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
        if ch == ";":
            while i < n and source[i] != "\n":
                i += 1
                col += 1
            continue
        span = Span(i, line, col)
        if ch in "()":
            tokens.append(Token(f"t{next(counter)}", ch, span))
            i += 1
            col += 1
            continue
        j = i
        while j < n and source[j] not in " \t\r\n();":
            j += 1
        tokens.append(Token(f"t{next(counter)}", source[i:j], span))
        col += j - i
        i = j
    return tokens


# ---------------------------------------------------------------- syntax
class Syntax:
    """A syntax object: datum + scope set + source span + provenance chain."""

    __slots__ = ("datum", "scopes", "span", "prov", "sid")

    def __init__(self, datum, scopes, span, prov, sid):
        self.datum = datum            # str | int | list[Syntax]
        self.scopes = frozenset(scopes)
        self.span = span              # Span | None
        self.prov = tuple(prov)       # tuple of provenance dicts
        self.sid = sid                # stable identity

    def with_scopes(self, scopes):
        return Syntax(self.datum, scopes, self.span, self.prov, self.sid)

    def __repr__(self):
        return f"Syntax({self.datum!r}, scopes={sorted(self.scopes)}, sid={self.sid})"


def add_scope(syn, scope):
    """Return a copy of `syn` with `scope` added to every identifier inside."""
    if isinstance(syn.datum, list):
        datum = [add_scope(x, scope) for x in syn.datum]
    else:
        datum = syn.datum
    return Syntax(datum, syn.scopes | {scope}, syn.span, syn.prov, syn.sid)


def parse(source: str):
    """Parse source text into a list of top-level syntax objects."""
    tokens = lex(source)
    pos = 0
    counter = itertools.count()

    def parse_one():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok.text == "(":
            items = []
            while True:
                if pos >= len(tokens):
                    raise ParseError("unclosed '('", tok.span)
                if tokens[pos].text == ")":
                    pos += 1
                    break
                items.append(parse_one())
            return Syntax(items, frozenset(), tok.span, (), f"s{next(counter)}")
        if tok.text == ")":
            raise ParseError("unexpected ')'", tok.span)
        text = tok.text
        try:
            val = int(text)
        except ValueError:
            val = text
        return Syntax(val, frozenset(), tok.span, (), f"s{next(counter)}")

    forms = []
    while pos < len(tokens):
        forms.append(parse_one())
    if not forms:
        raise ParseError("empty program", Span(0, 1, 1))
    return forms


# ---------------------------------------------------------------- quantities
KNOWN_BASES = {
    "char@line1col1": {"canonical": "offset@char", "kind": "span"},
    "scope-id@expansion-step": {"canonical": "scope-set@sorted-id", "kind": "scope-set"},
    "expansion-step": {"canonical": "count@step", "kind": "count"},
    "virtual-tick": {"canonical": "tick@monotonic", "kind": "time"},
}


def normalize_quantity(q):
    """Normalize one imported quantity into the canonical derived form.

    The raw representation is always preserved.  Unknown bases are not
    dropped: the value stays in place with a diagnostic.
    """
    basis = q.get("basis") or q.get("unit")
    qid = q.get("id", "?")
    if basis in KNOWN_BASES:
        spec = KNOWN_BASES[basis]
        return {
            "id": qid,
            "status": "normalized",
            "raw": q,
            "canonical": {
                "kind": spec["kind"],
                "basis": spec["canonical"],
                "value": q.get("value"),
            },
        }
    return {
        "id": qid,
        "status": "unconvertible",
        "raw": q,
        "diagnostic": f"unknown basis/unit: {basis!r}; raw value kept in place",
    }


def normalize_quantities(quantities):
    return [normalize_quantity(q) for q in (quantities or [])]
