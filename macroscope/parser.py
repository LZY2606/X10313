from __future__ import annotations

from .lexer import SourceImportError

OPEN = {"lparen": "(", "lbracket": "[", "lbrace": "{"}
CLOSE = {"rparen": ")", "rbracket": "]", "rbrace": "}"}
PAIR = {"(": ")", "[": "]", "{": "}"}


def parse(tokens):
    significant = [token for token in tokens if token["kind"] != "comment"]
    forms, cursor = [], 0

    def parse_form():
        nonlocal cursor
        if cursor >= len(significant):
            raise SourceImportError([{"code": "unexpected_end", "message": "expected a form but reached end of input", "span": None}])
        token = significant[cursor]
        if token["kind"] in CLOSE:
            raise SourceImportError([{"code": "unexpected_closing_delimiter", "message": f"unexpected {token['raw']!r}", "span": token["span"]}])
        if token["kind"] in OPEN:
            return parse_list(token)
        cursor += 1
        return {
            "id": f"S:{token['id']}",
            "type": "atom",
            "value": _atom_value(token),
            "token_ids": [token["id"]],
            "span": token["span"],
            "origin_span": token["span"],
            "synthetic": False,
            "scopes": [],
            "introduced_scope": None,
        }

    def parse_list(open_token):
        nonlocal cursor
        opening = open_token["raw"]
        cursor += 1
        children, token_ids = [], [open_token["id"]]
        while cursor < len(significant):
            token = significant[cursor]
            if token["kind"] in CLOSE:
                if token["raw"] != PAIR[opening]:
                    raise SourceImportError([{"code": "mismatched_delimiter", "message": f"expected {PAIR[opening]!r} but found {token['raw']!r}", "span": token["span"]}])
                token_ids.append(token["id"])
                cursor += 1
                return {
                    "id": f"S:{open_token['id']}:{token['id']}",
                    "type": "list",
                    "value": children,
                    "token_ids": token_ids,
                    "span": {"start": open_token["span"]["start"], "end": token["span"]["end"]},
                    "origin_span": {"start": open_token["span"]["start"], "end": token["span"]["end"]},
                    "synthetic": False,
                    "scopes": [],
                    "introduced_scope": None,
                    "delimiter": opening + PAIR[opening],
                }
            children.append(parse_form())
        raise SourceImportError([{"code": "unterminated_list", "message": f"missing closing {PAIR[opening]!r}", "span": open_token["span"]}])

    while cursor < len(significant):
        forms.append(parse_form())
    return forms


def _atom_value(token):
    if token["kind"] == "symbol":
        return token["raw"]
    if token["kind"] == "number":
        return {"number": token["numeric_value"], "unit": token.get("unit", "dimensionless"), "raw": token["raw"]}
    if token["kind"] == "string":
        return {"string": token["raw"][1:-1], "raw": token["raw"]}
    if token["kind"] == "boolean":
        return {"boolean": token.get("value"), "raw": token["raw"]}
    return token["raw"]
