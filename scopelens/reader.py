"""导入层：可复制的小型样例文本 -> 原 token（保留原始表示）-> syntax object。

- token 保留原始文本、source span（codepoint 坐标基准）、token 类型与顺序；
- 明确的格式错误带 source span（错误来源回映的基础）；
- datum 到 syntax 的初始上下文可见：整棵树携带初始 scope 集 {1(top)}，
  核心原语绑定在独立 scope {0(core)}。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .model import Bool, IdGen, Num, Span, Stx, Sym

SCOPE_CORE = 0
SCOPE_TOP = 1
INITIAL_SCOPES = frozenset({SCOPE_CORE, SCOPE_TOP})


class ReaderError(Exception):
    def __init__(self, message: str, span: Optional[Span] = None):
        super().__init__(message)
        self.message = message
        self.span = span


@dataclass(frozen=True)
class Token:
    kind: str          # lparen | rparen | atom
    text: str          # 原始表示（不规范化）
    start: int
    end: int
    line: int
    col: int

    @property
    def span(self) -> Span:
        return Span(self.start, self.end, self.line, self.col)

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text,
                "span": self.span.to_json(), "unit": "token"}


_DELIMS = set("()")
_TERMINATORS = set('();"') | {" ", "\t", "\n", "\r"}


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i, line, col = 0, 1, 1
    n = len(source)

    def advance(ch: str) -> None:
        nonlocal i, line, col
        i += 1
        if ch == "\n":
            line += 1
            col = 1
        else:
            col += 1

    while i < n:
        ch = source[i]
        if ch in " \t\n\r":
            advance(ch)
            continue
        if ch == ";":  # 行注释
            while i < n and source[i] != "\n":
                advance(source[i])
            continue
        start, sline, scol = i, line, col
        if ch in _DELIMS:
            tokens.append(Token("lparen" if ch == "(" else "rparen",
                                ch, start, i + 1, sline, scol))
            advance(ch)
            continue
        if ch == '"':
            raise ReaderError("字符串字面量不是本语言支持的数据表示",
                              Span(start, start + 1, sline, scol))
        # atom
        while i < n and source[i] not in _TERMINATORS:
            advance(source[i])
        text = source[start:i]
        tokens.append(Token("atom", text, start, i, sline, scol))
    return tokens


def parse_atom(text: str) -> Sym | Num | Bool:
    if text == "#t":
        return Bool(True)
    if text == "#f":
        return Bool(False)
    try:
        if text.lstrip("-").isdigit():
            return Num(int(text))
    except ValueError:
        pass
    return Sym(text)


def read_program(source: str, idgen: IdGen) -> tuple[list[Stx], list[Token]]:
    """返回顶层 syntax 列表与全部 token。原子 scope 初始化为 {core,top}。"""
    tokens = tokenize(source)
    forms: list[Stx] = []
    pos = 0

    def parse_list(open_tok: Token) -> Stx:
        nonlocal pos
        items: list[Stx] = []
        while True:
            if pos >= len(tokens):
                raise ReaderError("缺少右括号 ')'（EOF 前列表未闭合）",
                                  open_tok.span)
            tok = tokens[pos]
            if tok.kind == "rparen":
                pos += 1
                span = Span(open_tok.start, tok.end, open_tok.line, open_tok.col)
                return idgen.list_stx(items, span, INITIAL_SCOPES)
            if tok.kind == "lparen":
                pos += 1
                items.append(parse_list(tok))
            else:
                pos += 1
                items.append(idgen.atom_stx(parse_atom(tok.text), tok.span,
                                           INITIAL_SCOPES))

    while pos < len(tokens):
        tok = tokens[pos]
        if tok.kind == "rparen":
            raise ReaderError("遇到未匹配的右括号 ')'", tok.span)
        if tok.kind == "lparen":
            pos += 1
            forms.append(parse_list(tok))
        else:
            pos += 1
            forms.append(idgen.atom_stx(parse_atom(tok.text), tok.span,
                                       INITIAL_SCOPES))
    return forms, tokens
