"""Syntax objects: datum + scope set + provenance + stable identity.

A syntax object wraps an s-expression atom/pair and carries the set of lexical
scopes attached to that occurrence. Capture is decided by scope-set
containment, never by string renaming, so identical spellings with different
scope sets are distinct occurrences with distinct stable identities.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any

NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def stable_id(*parts: Any) -> str:
    payload = canonical(parts)
    return uuid.uuid5(NAMESPACE, payload).hex[:16]


def canonical(value: Any) -> str:
    if isinstance(value, frozenset) or isinstance(value, set):
        return "[" + ",".join(canonical(v) for v in sorted(value, key=str)) + "]"
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(canonical(v) for v in value) + ")"
    if isinstance(value, dict):
        return "{" + ",".join(f"{canonical(k)}={canonical(value[k])}"
                              for k in sorted(value)) + "}"
    return f"{type(value).__name__}:{value}"


@dataclass(frozen=True)
class Span:
    """Source span. line is 1-based, column is 0-based UTF-16 code units."""

    start_line: int
    start_col: int
    end_line: int
    end_col: int
    offset: int
    length: int

    def to_json(self) -> dict:
        return {
            "start_line": self.start_line, "start_col": self.start_col,
            "end_line": self.end_line, "end_col": self.end_col,
            "offset": self.offset, "length": self.length,
            "coordinate_basis": "line:1:column:0:utf16",
        }

    @staticmethod
    def from_json(data: dict) -> "Span":
        return Span(
            data["start_line"], data["start_col"],
            data["end_line"], data["end_col"],
            data["offset"], data["length"],
        )


@dataclass(frozen=True)
class OriginFrame:
    """One edge in the provenance chain (a transformation / macro frame)."""

    kind: str                 # token | macro | transform | core | error
    label: str
    span: Span | None = None
    invocation_id: str | None = None
    step: int | None = None
    macro_name: str | None = None
    detail: str = ""

    def to_json(self) -> dict:
        return {
            "kind": self.kind, "label": self.label,
            "span": self.span.to_json() if self.span else None,
            "invocation_id": self.invocation_id, "step": self.step,
            "macro_name": self.macro_name, "detail": self.detail,
        }

    @staticmethod
    def from_json(data: dict) -> "OriginFrame":
        return OriginFrame(
            kind=data["kind"], label=data["label"],
            span=Span.from_json(data["span"]) if data.get("span") else None,
            invocation_id=data.get("invocation_id"),
            step=data.get("step"), macro_name=data.get("macro_name"),
            detail=data.get("detail", ""),
        )


@dataclass
class Token:
    """An original lexical token; raw text is never overwritten."""

    text: str
    kind: str                 # lparen | rparen | quote | atom | number | error
    span: Span
    index: int
    raw: str = ""
    diagnostic: str | None = None

    def to_json(self) -> dict:
        return {
            "text": self.text, "kind": self.kind,
            "span": self.span.to_json(), "index": self.index,
            "raw": self.raw or self.text, "diagnostic": self.diagnostic,
        }

    @staticmethod
    def from_json(data: dict) -> "Token":
        return Token(
            text=data["text"], kind=data["kind"],
            span=Span.from_json(data["span"]), index=data["index"],
            raw=data.get("raw", data["text"]),
            diagnostic=data.get("diagnostic"),
        )


class Syn:
    """A syntax object: either an atom or a pair/list.

    `datum` is the plain s-expression spelling; `scopes` is the scope set;
    `origin` is the full provenance chain (oldest frame first).
    """

    def __init__(self, datum: Any, scopes: frozenset[str] = frozenset(),
                 origin: list[OriginFrame] | None = None,
                 token: Token | None = None, sid: str | None = None,
                 quoted: bool = False):
        self.datum = datum
        self.scopes = frozenset(scopes)
        self.origin = list(origin or [])
        self.token = token
        self.quoted = quoted
        self.children: list[Syn] = []
        self._sid = sid

    # ---- structural helpers -------------------------------------------------
    def is_pair(self) -> bool:
        return isinstance(self.datum, list)

    def atom(self) -> Any:
        return None if self.is_pair() else self.datum

    def add_scope(self, scope: str) -> "Syn":
        self.scopes = self.scopes | {scope}
        for child in self.children:
            child.add_scope(scope)
        self._sid = None
        return self

    def with_scope(self, scope: str) -> "Syn":
        result = self._clone()
        result.add_scope(scope)
        return result

    def push_origin(self, frame: OriginFrame) -> None:
        self.origin.append(frame)
        for child in self.children:
            child.push_origin(frame)
        self._sid = None

    def _clone(self) -> "Syn":
        cloned = Syn(self.datum, self.scopes, self.origin, self.token,
                     self._sid, self.quoted)
        cloned.children = [c._clone() for c in self.children]
        return cloned

    @property
    def sid(self) -> str:
        if self._sid is None:
            self._sid = self._compute_sid()
        return self._sid

    def _compute_sid(self) -> str:
        if self.is_pair():
            return stable_id("pair", [c.sid for c in self.children],
                             sorted(self.scopes))
        return stable_id("atom", self.datum, sorted(self.scopes),
                         self.span_key())

    def span_key(self) -> tuple:
        span = self.token.span if self.token else (
            self.origin[-1].span if self.origin and self.origin[-1].span else None)
        if span is None:
            return ()
        return (span.offset, span.length)

    def first_span(self) -> Span | None:
        if self.token is not None:
            return self.token.span
        for frame in reversed(self.origin):
            if frame.span is not None:
                return frame.span
        for child in self.children:
            found = child.first_span()
            if found is not None:
                return found
        return None

    # ---- serialization ------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "sid": self.sid, "datum": self.datum,
            "scopes": sorted(self.scopes), "quoted": self.quoted,
            "token": self.token.to_json() if self.token else None,
            "origin": [f.to_json() for f in self.origin],
            "children": [c.to_json() for c in self.children],
        }

    @staticmethod
    def from_json(data: dict) -> "Syn":
        token = Token.from_json(data["token"]) if data.get("token") else None
        node = Syn(
            datum=data["datum"], scopes=frozenset(data.get("scopes", [])),
            origin=[OriginFrame.from_json(f) for f in data.get("origin", [])],
            token=token, sid=data.get("sid"), quoted=data.get("quoted", False),
        )
        node.children = [Syn.from_json(c) for c in data.get("children", [])]
        return node
