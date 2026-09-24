from __future__ import annotations

import re

from . import COORDINATE_BASIS

ATOM_RE = re.compile(r"[^\s()\[\]{};\"']+")
NUMBER_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)([A-Za-z%][A-Za-z0-9%_-]*)?$")


class SourceImportError(ValueError):
    def __init__(self, errors):
        super().__init__("source import failed")
        self.errors = errors


def _span(content, start, end, line, column):
    return {
        "start": {"offset": start, "line": line, "column": column, "basis": COORDINATE_BASIS},
        "end": {
            "offset": end,
            "line": line if "\n" not in content[start:end] else line + content[start:end].count("\n"),
            "column": column + (end - start) if "\n" not in content[start:end] else len(content[start:end].splitlines()[-1]),
            "basis": COORDINATE_BASIS,
        },
    }


def tokenize(content):
    tokens = []
    errors = []
    offset = 0
    line = 1
    column = 0
    index = 0

    def advance(text):
        nonlocal offset, line, column
        lines = text.split("\n")
        if len(lines) > 1:
            line += len(lines) - 1
            column = len(lines[-1])
        else:
            column += len(text)
        offset += len(text)

    while offset < len(content):
        char = content[offset]
        if char in " \t\r\n":
            advance(char)
            continue
        start = offset
        start_line, start_column = line, column

        if char == ";":
            end_offset = content.find("\n", offset)
            if end_offset == -1:
                end_offset = len(content)
            raw = content[start:end_offset]
            advance(raw)
            tokens.append(_token(index, "comment", raw, content, start, start_line, start_column))
            index += 1
            continue

        if char == '"':
            end_offset = offset + 1
            while end_offset < len(content) and content[end_offset] != '"':
                if content[end_offset] == "\n":
                    errors.append(
                        {
                            "code": "unterminated_string",
                            "message": "string literal contains a raw newline before a closing quote",
                            "span": _span(content, start, min(end_offset + 1, len(content)), start_line, start_column),
                        }
                    )
                    break
                end_offset += 1
            if end_offset >= len(content) or content[end_offset] != '"':
                errors.append(
                    {
                        "code": "unterminated_string",
                        "message": "reached end of input while scanning string literal",
                        "span": _span(content, start, len(content), start_line, start_column),
                    }
                )
                break
            raw = content[start : end_offset + 1]
            advance(raw)
            tokens.append(_token(index, "string", raw, content, start, start_line, start_column))
            index += 1
            continue

        if char in "()[]{}":
            raw = char
            kind = {"(": "lparen", ")": "rparen", "[": "lbracket", "]": "rbracket", "{": "lbrace", "}": "rbrace"}[char]
            advance(raw)
            tokens.append(_token(index, kind, raw, content, start, start_line, start_column))
            index += 1
            continue

        match = ATOM_RE.match(content, offset)
        raw = match.group(0) if match else char
        advance(raw)
        number_match = NUMBER_RE.match(raw)
        if raw == "#t" or raw == "#true":
            kind, value = "boolean", True
        elif raw == "#f" or raw == "#false":
            kind, value = "boolean", False
        elif number_match and not (raw.startswith(("+", "-")) and not raw[1:2].isdigit()):
            kind = "number"
            value = None
        else:
            kind, value = "symbol", None
        token = _token(index, kind, raw, content, start, start_line, start_column)
        if kind == "number":
            token["numeric_value"] = float(number_match.group(1))
            token["unit"] = number_match.group(2) or "dimensionless"
        elif kind == "boolean":
            token["value"] = value
        elif kind == "string":
            token["value"] = raw[1:-1]
        tokens.append(token)
        index += 1

    if errors:
        raise SourceImportError(errors)
    return tokens


def _token(index, kind, raw, content, start, line, column):
    end = start + len(raw)
    return {
        "id": f"T{index + 1:04d}",
        "kind": kind,
        "raw": raw,
        "span": _span(content, start, end, line, column),
        "coordinate_basis": COORDINATE_BASIS,
    }
