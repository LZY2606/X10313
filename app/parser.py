"""Parser: tokens -> syntax objects (``Node`` trees).

Identities are deterministic path coordinates over the *submitted* token
stream (``<file>::l<open-token-index>`` for lists, ``<file>::a<token-index>``
for atoms, ``<file>::q<token-index>`` for reader-sugar wrappers), so the same
submission always yields the same ids, independent of clocks or randomness.
"""
from __future__ import annotations

from .lexer import tokenize, LexError
from .models import Node, Origin, Span, Token

SUGAR_NAME = {"'": "quote", "`": "quasiquote", ",": "unquote",
              ",@": "unquote-splicing"}


class ParseError(Exception):
    def __init__(self, message: str, span: Span):
        super().__init__(message)
        self.message = message
        self.span = span


def parse_source(src: str, file_id: str = "input") -> tuple[list[Node], list[Token]]:
    tokens = tokenize(src, file_id)
    cursor = _Cursor(tokens, file_id)
    forms: list[Node] = []
    while not cursor.at_end():
        forms.append(cursor.parse_form())
    return forms, tokens


class _Cursor:
    def __init__(self, tokens: list[Token], file_id: str):
        self.toks = tokens
        self.file = file_id
        self.i = 0

    def at_end(self) -> bool:
        return self.i >= len(self.toks)

    def _peek(self) -> Token:
        return self.toks[self.i]

    def parse_form(self) -> Node:
        tok = self._peek()
        if tok.kind == "rparen":
            raise ParseError("unexpected ')'", tok.span)
        if tok.kind == "lparen":
            return self.parse_list()
        if tok.kind == "sugar":
            return self.parse_sugar()
        return self.parse_atom()

    def parse_atom(self) -> Node:
        tok = self._peek()
        idx = self.i
        self.i += 1
        lit = {"symbol": "symbol", "number": "number", "string": "string",
               "boolean": "boolean"}[tok.kind]
        value = tok.text
        return Node("atom", f"{self.file}::a{idx}", frozenset(), tok.span,
                    (), value, lit, Origin("parse"))

    def parse_sugar(self) -> Node:
        tok = self._peek()
        idx = self.i
        self.i += 1
        inner = self.parse_form()
        name = SUGAR_NAME[tok.text]
        head = Node("atom", f"{self.file}::q{idx}#h", frozenset(), tok.span,
                    (), name, "symbol", Origin("sugar", basis=f"{self.file}::a{idx}"))
        return Node("list", f"{self.file}::q{idx}", frozenset(), tok.span,
                    (head, inner), None, "symbol", Origin("sugar"))

    def parse_list(self) -> Node:
        open_tok = self._peek()
        open_idx = self.i
        self.i += 1
        children: list[Node] = []
        dotted = False
        while not self.at_end() and self._peek().kind != "rparen":
            tok = self._peek()
            if tok.kind == "dot":
                if dotted or not children:
                    raise ParseError("malformed dotted pair", tok.span)
                self.i += 1
                tail = self.parse_form()
                children.append(tail)
                dotted = True
                continue
            children.append(self.parse_form())
        if self.at_end():
            raise ParseError("unclosed '(' (missing ')')", open_tok.span)
        close = self._peek()
        self.i += 1
        return Node("list", f"{self.file}::l{open_idx}", frozenset(), open_tok.span,
                    tuple(children), None, "symbol", Origin("parse"),
                    **({"dot": True} if dotted else {}))
