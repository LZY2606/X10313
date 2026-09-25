"""Core domain model: tokens, spans, scopes and syntax objects.

Every position keeps its *basis* (offset/line/column origin) and every identifier
keeps an explicit set of scopes. Hygiene is decided by scope-set relations, never
by renaming strings.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional


# --------------------------------------------------------------------------- #
# stable identity helpers
# --------------------------------------------------------------------------- #

def cid(prefix: str, parts: Iterable[Any]) -> str:
    """Content-addressed stable id: ``prefix`` + 16 hex chars of a hash."""
    payload = json.dumps(_normal(parts), ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _normal(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(_normal(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    if isinstance(value, dict):
        return {k: _normal(v) for k, v in sorted(value.items())}
    return value


# --------------------------------------------------------------------------- #
# coordinates / tokens
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Span:
    """Half-open character span with an explicit coordinate basis.

    ``basis`` names the source fragment whose first character is offset 0,
    line 1, column 1. Positions are never silently normalised between fragments.
    """
    start: int
    end: int
    line: int
    col: int
    basis: str

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "line": self.line,
                "col": self.col, "basis": self.basis}


@dataclass(frozen=True)
class Token:
    tid: str
    text: str
    kind: str  # lparen | rparen | quote | atom | error
    span: Span
    diagnostic: Optional[str] = None

    def as_dict(self) -> dict:
        data = {"tid": self.tid, "text": self.text, "kind": self.kind,
                "span": self.span.as_dict()}
        if self.diagnostic:
            data["diagnostic"] = self.diagnostic
        return data


# --------------------------------------------------------------------------- #
# scopes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Scope:
    """A lexical / macro-introduction / gensym scope.

    ``gid`` is deterministic given (reason, counter, parent/frame identity),
    so replaying with a fixed seed reproduces every scope id.
    """
    gid: str
    reason: str          # module | lambda | let | macro-intro | gensym | branch | local
    label: str           # human label, e.g. "lambda@3:1"
    origin: Optional[str] = None  # node id that created the scope

    def as_dict(self) -> dict:
        return {"gid": self.gid, "reason": self.reason, "label": self.label,
                "origin": self.origin}


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OriginStep:
    """One step in the datum -> syntax provenance chain."""
    kind: str          # token | copied | macro-call | gensym | datum->syntax | macro-output | wrap
    detail: dict
    node: Optional[str] = None  # related expansion node (macro call frame, ...)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "node": self.node}


# --------------------------------------------------------------------------- #
# syntax object
# --------------------------------------------------------------------------- #

@dataclass
class Syntax:
    """A syntax object: datum plus scopes plus provenance.

    Three datum shapes:
      * symbol identifier -> ``is_id`` true, ``value`` is the symbol string
      * leaf datum        -> number/string/bool, ``is_id`` false
      * pair/list         -> ``is_pair`` true, ``children`` is a list
    """
    sid: str
    value: Any = None
    is_id: bool = False
    is_pair: bool = False
    children: list["Syntax"] = field(default_factory=list)
    scopes: frozenset[str] = frozenset()
    token_ids: tuple[str, ...] = ()
    span: Optional[Span] = None
    origin: tuple[OriginStep, ...] = ()
    context: Optional[str] = None  # visible datum->syntax context choice, when relevant

    # -- scope editing (the only mutation of binding identity) -------------- #
    def with_scope(self, gid: str) -> "Syntax":
        if gid in self.scopes:
            return self
        new = replace(self, scopes=frozenset(self.scopes | {gid}))
        new.origin = self.origin + (OriginStep("wrap", {"add": gid}),)
        return new

    def without_scope(self, gid: str) -> "Syntax":
        if gid not in self.scopes:
            return self
        new = replace(self, scopes=frozenset(self.scopes - {gid}))
        new.origin = self.origin + (OriginStep("wrap", {"remove": gid}),)
        return new

    # -- traversal ---------------------------------------------------------- #
    def walk(self):
        yield self
        if self.is_pair:
            for child in self.children:
                yield from child.walk()

    def as_dict(self) -> dict:
        data: dict[str, Any] = {
            "sid": self.sid,
            "is_id": self.is_id,
            "is_pair": self.is_pair,
            "scopes": sorted(self.scopes),
            "token_ids": list(self.token_ids),
            "origin": [step.as_dict() for step in self.origin],
        }
        if self.span is not None:
            data["span"] = self.span.as_dict()
        if self.is_pair:
            data["children"] = [c.as_dict() for c in self.children]
        else:
            data["value"] = self.value
        if self.context is not None:
            data["context"] = self.context
        return data


def symbol_syntax(sid: str, name: str, *, token_ids=(), span: Optional[Span] = None,
                  scopes: Iterable[str] = (), origin: Iterable[OriginStep] = (),
                  context: Optional[str] = None) -> Syntax:
    return Syntax(sid=sid, value=name, is_id=True, token_ids=tuple(token_ids),
                  span=span, scopes=frozenset(scopes), origin=tuple(origin),
                  context=context)
