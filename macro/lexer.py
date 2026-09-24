"""Lexer for the small macro language.

Every token keeps its source span. Spans are domain quantities: they carry a
unit ("char") and a coordinate basis (file + line/col + absolute offset).
The raw representation is always preserved; normalisation happens only in
derived layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Span:
    file: str
    offset: int          # absolute char offset, unit: char
    length: int          # unit: char
    line: int            # 1-based, coordinate basis: file text
    col: int             # 1-based, unit: char
    unit: str = "char"
    basis: str = "file+offset"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Token:
    tid: str
    kind: str            # lparen | rparen | ident | number | string
    text: str
    span: Span

    def to_dict(self) -> dict:
        return {"tid": self.tid, "kind": self.kind, "text": self.text,
                "span": self.span.to_dict()}


@dataclass
class LexDiagnostic:
    code: str
    message: str
    span: Span | None
    severity: str = "error"

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message,
                "span": self.span.to_dict() if self.span else None,
                "severity": self.severity}


class LexError(Exception):
    def __init__(self, diagnostics: list[LexDiagnostic]):
        super().__init__(diagnostics[0].message if diagnostics else "lex error")
        self.diagnostics = diagnostics


def lex(source: str, file: str = "<input>") -> tuple[list[Token], list[LexDiagnostic]]:
    tokens: list[Token] = []
    diags: list[LexDiagnostic] = []
    i, line, col = 0, 1, 1
    n = len(source)
    counter = 0

    def span(start: int, length: int, sl: int, sc: int) -> Span:
        return Span(file=file, offset=start, length=length, line=sl, col=sc)

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
        if ch == ";":  # comment to end of line
            while i < n and source[i] != "\n":
                i += 1
                col += 1
            continue
        counter += 1
        tid = f"tok-{counter:04d}"
        if ch == "(":
            tokens.append(Token(tid, "lparen", ch, span(i, 1, line, col)))
            i += 1; col += 1
        elif ch == ")":
            tokens.append(Token(tid, "rparen", ch, span(i, 1, line, col)))
            i += 1; col += 1
        elif ch == '"':
            sl, sc, start = line, col, i
            i += 1; col += 1
            buf = []
            closed = False
            while i < n:
                c = source[i]
                if c == '"':
                    closed = True
                    i += 1; col += 1
                    break
                if c == "\n":
                    line += 1; col = 1; i += 1
                    buf.append(c)
                    continue
                buf.append(c)
                i += 1; col += 1
            text = "".join(buf)
            if not closed:
                diags.append(LexDiagnostic(
                    "unterminated-string",
                    f"unterminated string literal starting at line {sl}",
                    span(start, i - start, sl, sc)))
            tokens.append(Token(tid, "string", text, span(start, i - start, sl, sc)))
        elif ch.isdigit() or (ch == "-" and i + 1 < n and source[i + 1].isdigit()):
            start, sc = i, col
            while i < n and (source[i].isdigit() or source[i] in ".-"):
                i += 1; col += 1
            tokens.append(Token(tid, "number", source[start:i], span(start, i - start, line, sc)))
        elif ch.isalpha() or ch in "_+*/<>=!?$%&|~^-":
            start, sc = i, col
            while i < n and (source[i].isalnum() or source[i] in "_+*/<>=!?$%&|~^-."):
                i += 1; col += 1
            tokens.append(Token(tid, "ident", source[start:i], span(start, i - start, line, sc)))
        else:
            diags.append(LexDiagnostic(
                "bad-character", f"unexpected character {ch!r}",
                span(i, 1, line, col)))
            i += 1; col += 1
    return tokens, diags
