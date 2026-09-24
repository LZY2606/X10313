"""Parser: significant tokens -> tree of original syntax objects.

Dotted pairs and quote abbreviations are supported. Every node carries the
span of its first/last token so errors can map back to submitted source.
"""
from .errors import ParseError
from .syntax import SynFactory

QUOTE_FORMS = {"'": "quote", "`": "quasiquote", ",": "unquote", ",@": "unquote-splicing"}


class Parser:
    def __init__(self, tokens, factory=None):
        self.tokens = [t for t in tokens if t["type"] not in ("ws", "comment")]
        self.pos = 0
        self.fac = factory or SynFactory()

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self):
        t = self.peek()
        if t is None:
            raise ParseError("unexpected end of input")
        self.pos += 1
        return t

    def parse_program(self):
        forms = []
        while self.peek() is not None:
            forms.append(self.parse_form())
        return forms

    def parse_form(self):
        t = self.peek()
        if t is None:
            raise ParseError("unexpected end of input")
        if t["type"] == "quote":
            self.next()
            inner = self.parse_form()
            return self._wrap(QUOTE_FORMS[t["value"]], t, inner)
        if t["type"] == "paren":
            if t["value"] == ")":
                raise ParseError("unexpected ')'", span=t["span"])
            return self.parse_list()
        return self.parse_atom(t)

    def _wrap(self, head, quote_tok, inner):
        head_syn = self.fac.mk_original("id", head, [], quote_tok["span"])
        span = {"file": quote_tok["span"]["file"],
                "start": quote_tok["span"]["start"],
                "end": inner.span["end"] if inner.span else quote_tok["span"]["end"]}
        return self.fac.mk_original("list", None, [head_syn, inner], span)

    def parse_atom(self, t):
        self.next()
        kind = {"symbol": "id", "number": "num", "string": "str", "bool": "bool"}[t["type"]]
        if kind == "id" and t["value"] == ".":
            raise ParseError("'.' is only allowed inside a list", span=t["span"])
        return self.fac.mk_original(kind, t["value"], [], t["span"])

    def parse_list(self):
        open_tok = self.next()  # '('
        kids = []
        dotted = None
        while True:
            t = self.peek()
            if t is None:
                raise ParseError("unterminated list: missing ')'",
                                 span=open_tok["span"])
            if t["type"] == "paren" and t["value"] == ")":
                close = self.next()
                break
            form = self.parse_form()
            # dot handling
            nt = self.peek()
            if (nt is not None and nt["type"] == "symbol" and nt["value"] == "."):
                self.next()
                dotted = self.parse_form()
                closer = self.peek()
                if closer is None or not (closer["type"] == "paren" and closer["value"] == ")"):
                    raise ParseError("expected ')' after dotted tail",
                                     span=nt["span"])
                close = self.next()
                break
            kids.append(form)
        end_span = close["span"]["end"]
        span = {"file": open_tok["span"]["file"],
                "start": open_tok["span"]["start"], "end": end_span}
        lst = self.fac.mk_original("list", None, kids, span)
        if dotted is not None:
            lst.flags.add("dotted")
            lst.children.append(dotted)
        return lst


def parse(tokens, factory=None):
    p = Parser(tokens, factory)
    return p.parse_program(), p.fac
